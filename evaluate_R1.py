# Copyright 2024-2025 Daniel Franzen, Jan Disselhoff and David Hartmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import bz2
import pickle
from tqdm import tqdm
from datasets import Dataset
from diskcache import Cache

import torch
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments

from arc_loader import ArcDataset
from model_tools import (
    InputMaskingDataCollator,
    keep_single_char_tokens,
    load_peft_state,
    merge_peft_into_base,
    save_model_and_tokenizer,
)
from inference_tools import inference_run
from selection import EvalTool
from arc_downloader import download_arc_data
import argparse

parser = argparse.ArgumentParser(description="Evaluate ARC model.")
parser.add_argument("use_symbolic", type=str,
                    help="Use symbolic mode: 'true' or 'false'.")
parser.add_argument("min_prob", type=float,
                    help="Minimum probability threshold (e.g. 0.2).")
args = parser.parse_args()

# convert string to boolean
use_symbolic = args.use_symbolic.lower() in ("true", "1", "t", "yes", "y")
min_prob = args.min_prob

# =========================================================
# Paths / Config
# =========================================================
BASE_MODEL_NAME   = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TRAINED_LORA_DIR  = "pretrained_models/R1-1.5B-ARChitects-ReArc369-bnb-4bit-lora"  # your trained LoRA
MERGED_FP_DIR     = "pretrained_models/R1-1.5B-ARChitects-ReArc369-pruned-merged-fp16"  # will be created

arc_data_path     = os.path.join("input", "arc-prize-2024")  # auto-downloaded
output_path = f'output/R1_dfs{min_prob}_{"sym" if use_symbolic else "neuro"}'
ttt_target_size   = 1

# One LoRA at runtime: we’ll attach only this adapter for TTT.
TTT_LORA_R        = 32
TTT_LORA_ALPHA    = 16
TTT_LORA_DROPOUT  = 0.0

MAX_SEQ_LEN       = 16384
ATTN_IMPL         = "flash_attention_2"  # avoids 'eager' sliding-window warning

# =========================================================
# One-time: build PRUNED + MERGED FP checkpoint (no adapters)
# =========================================================
def ensure_pruned_merged_fp_checkpoint():
    """Create a pruned+merged FP16/BF16 checkpoint once, then reuse it."""
    if os.path.isdir(MERGED_FP_DIR) and os.path.exists(os.path.join(MERGED_FP_DIR, "config.json")):
        print(f"[prep] Found existing merged FP checkpoint at: {MERGED_FP_DIR}")
        return

    print("[prep] Creating pruned+merged FP checkpoint (no adapters)...")

    # 1) Load base in FULL precision (no 4-bit). Use Unsloth's loader.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL_NAME,
        max_seq_length = MAX_SEQ_LEN,
        dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        load_in_4bit = False,            # FULL precision here
        device_map = "auto",
        attn_implementation = ATTN_IMPL,
        trust_remote_code = True,
    )

    # 2) Replay EXACT pruning from training
    keep_tok = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=") + tokenizer.tokenize("\n")
    keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)

    # 3) Recreate training-time LoRA topology (r=256) and load trained adapter
    lora_layers = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','embed_tokens','lm_head']
    model = FastLanguageModel.get_peft_model(
        model = model,
        target_modules = lora_layers,
        r = 256,
        lora_alpha = 24,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = False,
        random_state = 42,
        use_rslora = True,
        loftq_config = None,
    )
    load_peft_state(model, TRAINED_LORA_DIR)

    # 4) Merge LoRA into base weights (no adapters remain)
    model = merge_peft_into_base(model)

    # 5) Save the pruned+merged FP checkpoint
    os.makedirs(os.path.dirname(MERGED_FP_DIR), exist_ok=True)
    save_model_and_tokenizer(MERGED_FP_DIR, model, tokenizer)
    print(f"[prep] Saved pruned+merged FP checkpoint to: {MERGED_FP_DIR}")

    # Free GPU RAM
    del model
    torch.cuda.empty_cache()

# =========================================================
# Loader used during evaluation: load merged FP in 4-bit and attach ONE TTT adapter
# =========================================================
def load_4bit_base_and_attach_single_ttt_adapter():
    """
    Load pruned+merged FP checkpoint in 4-bit, then attach ONE LoRA adapter (TTT) for on-the-fly training.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MERGED_FP_DIR,
        max_seq_length = MAX_SEQ_LEN,
        dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        load_in_4bit = True,             # quantize on load (bnb)
        device_map = "auto",
        attn_implementation = ATTN_IMPL,
        trust_remote_code = True,
    )

    # Attach ONLY ONE adapter (TTT). This is the *only* LoRA present at runtime.
    ttt_layers = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','embed_tokens','lm_head']
    model = FastLanguageModel.get_peft_model(
        model = model,
        target_modules = ttt_layers,
        r = TTT_LORA_R,
        lora_alpha = TTT_LORA_ALPHA,
        lora_dropout = TTT_LORA_DROPOUT,
        bias = "none",
        use_gradient_checkpointing = False,
        random_state = 42,
        use_rslora = True,
        loftq_config = None,
    )

    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    return model, tokenizer

# =========================================================
# Build ARC dataset + fmt
# =========================================================
print("downloading arc...")
download_arc_data(arc_data_path)
eval_dataset = ArcDataset.load_from_json(os.path.join(arc_data_path, "arc-agi_evaluation_challenges.json"))
eval_dataset = eval_dataset.load_solutions(os.path.join(arc_data_path, "arc-agi_evaluation_solutions.json"))
print("Done")

# Make sure the pruned+merged FP checkpoint exists
print("merguing models...")
ensure_pruned_merged_fp_checkpoint()
print("Done")

# We can pull tokenizer from the merged FP dir for EOS + formatting
_, tokenizer_for_fmt = FastLanguageModel.from_pretrained(
    model_name = MERGED_FP_DIR,
    max_seq_length = MAX_SEQ_LEN,
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    load_in_4bit = True,                 # tiny load only to read eos token; will be GC'd
    device_map = "auto",
    attn_implementation = ATTN_IMPL,
    trust_remote_code = True,
)
eos_tok = tokenizer_for_fmt.eos_token
del tokenizer_for_fmt
torch.cuda.empty_cache()

fmt_opts = dict(
    preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
    query_beg='I',
    reply_beg='\n+/-=O',
    reply_end='\n' + eos_tok,
    lines_sep='\n',
    max_tokens=MAX_SEQ_LEN,
)

# =========================================================
# Evaluation loop (ONE adapter at runtime)
# =========================================================
inference_keys = {}
inference_results = {}
all_eval_tools  = [
    EvalTool(n_guesses = 2, specific_constraint = "only_one_color_changes"),
    EvalTool(n_guesses = 2, specific_constraint = "a_specific_color_does_not_change"),
    EvalTool(n_guesses = 2, specific_constraint = "output_is_different_to_input"),
    EvalTool(n_guesses = 2, specific_constraint = "color_histogramm_stays_the_same_except_one"),
    EvalTool(n_guesses = 2, specific_constraint = "count_of_a_specific_color_does_not_change"),
    EvalTool(n_guesses = 2, specific_constraint = "no_lonely_pixels_in_output"),
    EvalTool(n_guesses = 2, specific_constraint = "background_color_decreases"),
    EvalTool(n_guesses = 2, specific_constraint = "output_has_specific_set_of_colors"),
    EvalTool(n_guesses = 2, specific_constraint = "count_of_a_specific_color_changes_by_specific_amount"),
    EvalTool(n_guesses = 2, specific_constraint = "output_is_horizontally_mirrored"),
    EvalTool(n_guesses = 2, specific_constraint = "output_is_vertically_mirrored"),
    EvalTool(n_guesses = 2, specific_constraint = "color_histogramm_stays_the_same"),
    EvalTool(n_guesses = 2, specific_constraint = None),
]

os.makedirs(output_path, exist_ok=True)

print("start inference")
with tqdm(eval_dataset.split(n=len(eval_dataset.challenge)//ttt_target_size, split_seed=123), desc='inference') as pbar:
    for i, eval_dataset_part in enumerate(pbar):

        # Cache per split: build the 4-bit merged base + attach single TTT LoRA; TTT it; return for inference
        def get_model_and_tokenizer(cache=[None]):
            if cache[0] is None:
                # 1) Load 4-bit merged base and attach single TTT adapter
                model, tok = load_4bit_base_and_attach_single_ttt_adapter()

                # 2) Augment training data for TTT
                train_aug_opts = dict(tp='all', rt='all', shfl_keys=True, perm=True, shfl_ex=True, seed=i)
                train_dataset_aug = eval_dataset_part.remove_test_data().augment(n=8, **train_aug_opts)
                train_dataset_as_list = train_dataset_aug.as_list(len_name='text', **fmt_opts)

                # 3) TTT: train ONLY that TTT adapter
                FastLanguageModel.for_training(model)
                trainer = Trainer(
                    model = model,
                    tokenizer = tok,
                    train_dataset = Dataset.from_list(train_dataset_as_list),
                    dataset_text_field = "text",
                    max_seq_length = fmt_opts['max_tokens'],
                    data_collator = InputMaskingDataCollator(
                        instruction_template = fmt_opts['query_beg'],
                        response_template = fmt_opts['reply_beg'],
                        mlm = False,
                        tokenizer = tok,
                        mask_first_n_examples = 1,
                    ),
                    args = TrainingArguments(
                        per_device_train_batch_size = 1,
                        gradient_accumulation_steps = 1,
                        warmup_steps = 32,
                        num_train_epochs = 1,
                        learning_rate = 1e-4,
                        embedding_learning_rate = 1e-5,
                        fp16 = not is_bfloat16_supported(),
                        bf16 = is_bfloat16_supported(),
                        logging_steps = 8,
                        optim = "adamw_8bit",
                        weight_decay = 0.00,
                        lr_scheduler_type = 'cosine',
                        seed = 42,
                        output_dir = 'tmp_output',
                        save_strategy = 'no',
                        report_to = 'none',
                    ),
                )
                _ = unsloth_train(trainer)

                # 4) Back to inference
                FastLanguageModel.for_inference(model)
                cache[0] = (model, tok)
            return cache[0]

        # Inference over augmented eval split
        task_cache_path = os.path.join(output_path, f'{sorted(eval_dataset_part.challenge.keys())[0]}.cache')
        infer_aug_opts = dict(tp='all', rt='all', perm=True, shfl_ex=True, seed=10000 + i)
        eval_dataset_augmented = eval_dataset_part.augment(n=2, **infer_aug_opts)

        for k in eval_dataset_augmented.keys:
            base_key = k.split('.', 1)[0]
            if base_key not in inference_keys:
                inference_keys[base_key] = []
            inference_keys[base_key].append(k)
        inference_results.update(
            inference_run(
                model_tok=get_model_and_tokenizer,
                fmt_opts=fmt_opts,
                dataset=eval_dataset_augmented,
                min_prob=min_prob,
                aug_score_opts=dict(n=2, **infer_aug_opts),
                eval_tools=all_eval_tools,
                cache=Cache(task_cache_path).memoize(typed=True, ignore=set(['model_tok', 'guess'])),
                print_func=pbar.write,
                use_symbolic=use_symbolic,
            )
        )

# =========================================================
# Dump results + submission
# =========================================================
results_file = os.path.join(output_path, 'results.pickle.bz2')
with bz2.BZ2File(results_file, 'w') as f:
    pickle.dump(inference_keys, f)
    pickle.dump(inference_results, f)

submission_file = os.path.join(output_path, 'submission.json')
with open(submission_file, 'w') as f:
    json.dump(eval_dataset.get_submission(inference_results), f)
with open(submission_file, 'r') as f:
    print(f"Reload score for '{submission_file}':", eval_dataset.validate_submission(json.load(f)))
