# Copyright 2024-2025 Daniel Franzen, Jan Disselhoff and David Hartmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import torch
import hashlib
import numpy as np
import time

from symbolic_constraints import ArcSymbolicConstraints


def is_unsloth_model(model):
    return model.model_tags is not None and 'unsloth' in model.model_tags


def logits_to_score(sequence, logits):
    assert sequence.ndim == 1
    assert logits.ndim == 2
    assert len(sequence) <= len(logits)
    return -logits.log_softmax(-1)[torch.arange(len(sequence)), sequence].sum().item()


def cut_after_first_eos(sequence, eos_token_id):
    eos_positions = (sequence == eos_token_id).nonzero()[:, 0]
    return sequence[:eos_positions[0] + 1] if len(eos_positions) else sequence


def extract_sequence_and_score(sequence, logits, eos_token_id):
    sequence = cut_after_first_eos(sequence, eos_token_id=eos_token_id)
    return sequence, logits_to_score(sequence, logits)


def preprocess_generation_outputs(gen, input_len, eos_token_id):
    sequences = gen['sequences'][:, input_len:].cpu()
    logits = torch.stack(gen['logits'], axis=-2).float().cpu()
    assert sequences.ndim == 2
    assert logits.ndim == 3
    assert sequences.shape[0] == logits.shape[0]
    assert sequences.shape[1] <= logits.shape[1]
    return [extract_sequence_and_score(s, l, eos_token_id=eos_token_id) for s, l in zip(sequences, logits)]


def calc_score(input, reply, model_tok, cache=None, **_):
    if cache is not None:  # try loading result from cache
        return cache(calc_score)(input=input, reply=reply, model_tok=model_tok)

    # prepare model and tokenizer
    model, tokenizer = model_tok if isinstance(model_tok, (list, tuple)) else model_tok()

    with torch.no_grad():  # calculate score
        input_len = len(tokenizer(input)['input_ids'])
        tokenized = tokenizer([input+reply], return_tensors='pt')
        tokenized.pop('token_type_ids', None)
        sequence = tokenized['input_ids'][0][input_len:].cpu()
        logits = model(**tokenized.to(model.device))['logits'][0, input_len-1: -1].float().cpu()
        return logits_to_score(sequence, logits)


def prune_cache(cache, max_len):
    is_legacy_cache = isinstance(cache, (tuple, list))
    if max_len < (cache[0][0].shape[2] if is_legacy_cache else cache.get_seq_length()):
        if is_legacy_cache: cache = tuple(tuple(c[:, :, :max_len] for c in l) for l in cache)
        else: cache.crop(max_len)
    return cache


def explore(model, logits, path, eos, max_new_tokens, max_score, pos, cache, symbolic: ArcSymbolicConstraints, enforce_symbolic: bool, token_lookup, score=0.0, violating_constraint = None):
    first_token_logits, logits = logits[0], (logits[1:] if len(logits) > 1 else None)
    softmax = list(enumerate(-first_token_logits.detach().float().log_softmax(-1).cpu()))

    if len(path):  # follow precomputed path first
        softmax[0], softmax[path[0]], path = softmax[path[0]], softmax[0], path[1:]

    return_suffixes = []
    for i, s in softmax:  # loop over all possible tokens
        newly_violating_constraint = None
        next_score = score + s.item()
        if next_score < max_score:  # check if still below the score limit, otherwise stop exploration

            if violating_constraint == None:
                follows, newly_violating_constraint = symbolic.check_characteristics(token_lookup[i])
                #if newly_violating_constraint != None:
                #    symbolic.add_new_token(.decode(i))
                #    print(symbolic.characteristics)
                #    print(f"reject adding token {token_lookup[i]} in \ncurrent state:\n{symbolic.current_state}\ntest_input:\n{symbolic.test_input}\ncolor_map: {symbolic.current_color_map}\ncolormap_counts: {symbolic.current_color_map_counts}\nhistogram:\n{symbolic.current_histogramm}\nis_finished={symbolic.current_is_finished}")
                #    symbolic.remove_last_token()
                #    input("Confirm")
            
                if enforce_symbolic and not follows:
                    continue

            if i == eos:  # candidate found, append to suffixes (tokens are aggregated on backward pass)
                suffixes = [([], next_score, violating_constraint if violating_constraint != None else newly_violating_constraint)]
            elif max_new_tokens > 1:  # check if still below token limit, otherwise stop exploration
                if logits is None:  # if not following the initial guess, calculate logits to pass to explore function
                    logits, cache[0] = model(
                        input_ids=torch.full((1, 1), i, device=model.device),
                        position_ids=torch.full((1, 1), pos, device=model.device),
                        past_key_values=prune_cache(cache[0], pos),
                    )[:2]
                    logits = logits[0]  # unbatch
                
                # explore suffixes
                symbolic.add_new_token(token_lookup[i])
                #print(f"after adding token {token_lookup[i]} the current state is:\n{symbolic.current_state}\nis_legal = {is_legal}")
                suffixes = explore(model, logits, path, eos, max_new_tokens-1, max_score, pos+1, cache, symbolic, enforce_symbolic, token_lookup, next_score, violating_constraint if violating_constraint != None else newly_violating_constraint)
                symbolic.remove_last_token()
                #print(f"after removing token {token_lookup[i]} the current state is:\n{symbolic.current_state}")
            else: suffixes = []

            # update suffixes
            for suffix in suffixes:
                suffix[0].append(i)
            return_suffixes.extend(suffixes)

        logits = None
    return return_suffixes


def dfs(model, input_ids, eos_token_id, max_new_tokens, min_prob, symbolic, token_lookup, pos=None, attention_mask=None, enforce_symbolic = False):
    assert not torch.is_grad_enabled()
    assert attention_mask is None or attention_mask.all(), 'not implemented'
    sys.setrecursionlimit(1000 + max_new_tokens)  # avoid stack overflows

    # prepare inputs
    input_ids = torch.as_tensor(input_ids, device=model.device, dtype=int)
    if input_ids.ndim == 2:
        input_ids = input_ids.squeeze(0)
    assert input_ids.ndim == 1, 'batching not supported'

    if pos is None:
        # no guess passed, set generation starting position to length of input
        pos = len(input_ids)
    elif pos < len(input_ids):
        # if guess passed, remove final eos_token from input
        if input_ids[-1] == eos_token_id:
            input_ids = input_ids[:-1]

    # process prompt and best guess
    logits, cache = model(input_ids=input_ids[torch.newaxis])[:2]
    logits = logits[0, pos-1:]

    # run dfs
    result = explore(model, logits, input_ids[pos:], eos_token_id, max_new_tokens, -np.log(min_prob), pos, [cache], symbolic, enforce_symbolic,token_lookup)

    # return results sorted by scores
    return sorted([(np.array(suffix[::-1]), score_val, violating_cosntraint) for suffix, score_val,violating_cosntraint in result], key=lambda x: x[1])


def infer_single(prompt, model_tok, symbolic, guess=None, enforce_symbolic= False, min_prob=None, cache=None, **kwargs):
    assert len(prompt)

    if cache is not None:  # try loading result from cache
        return cache(infer_single)(prompt=prompt, model_tok=model_tok, symbolic = symbolic,enforce_symbolic = enforce_symbolic, guess=guess, min_prob=min_prob, **kwargs)

    # prepare model and tokenizer
    model, tokenizer = model_tok if isinstance(model_tok, (list, tuple)) else model_tok()

    token_lookup = []
    for i in range(100):
        try:
            token_lookup.append(tokenizer.decode(i))
        except IndentationError:
            break

    with torch.no_grad():
        # tokenize input
        tokenized = tokenizer(prompt, return_tensors='pt').to(model.device)
        input_len = tokenized['input_ids'].shape[-1]
        tokenized.pop('token_type_ids', None)

        if min_prob is not None:  # run dfs if 'min_prob' is passed
            if guess is not None:
                tokenized = tokenizer(guess, return_tensors='pt').to(model.device)
                tokenized.pop('token_type_ids', None)
            ret = dfs(model, **tokenized, pos=input_len, min_prob=min_prob, symbolic= symbolic,enforce_symbolic = enforce_symbolic, token_lookup = token_lookup, eos_token_id=tokenizer.eos_token_id, **kwargs)

        else:  # run model 'generate' function
            assert kwargs.get('num_beams', 1) == 1 or not is_unsloth_model(model)
            gen = model.generate(**tokenized, return_dict_in_generate=True, output_logits=True, use_cache=True,
                                 eos_token_id=tokenizer.eos_token_id, **kwargs)
            ret = preprocess_generation_outputs(gen, input_len=input_len, eos_token_id=tokenizer.eos_token_id)


        return [(tokenizer.decode(o), s, v_c) for o, s, v_c in ret]


def process_output(unique_results, dataset, fmt_opts, aug_score_opts, key, score, output, correct, score_args):
    # add output to results
    hashable = tuple(map(tuple, output))
    if hashable not in unique_results:
        unique_results[hashable] = dict(output=output, correct=correct, scores_inf={})
    res = unique_results[hashable]

    # store inference score
    if score is not None:
        res['scores_inf'][key] = score

    # calculate augmented score
    if aug_score_opts and 'scores_aug' not in res:
        aug_score_opts_copy = aug_score_opts.copy()
        key_hash = int(hashlib.md5(key.split('.')[0].encode('utf-8')).hexdigest()[:6], 16)
        np.random.seed(aug_score_opts_copy.pop('seed') + key_hash)
        aug_keys = dataset.augment_keys([key.split('.', 1)[0]], **aug_score_opts_copy)
        aug_key_fmt = [dataset.get_task(k, reply=output, **fmt_opts) for k in aug_keys]
        res['scores_aug'] = {key: calc_score(**fmt, **score_args) for key, fmt in aug_key_fmt}
        if res['scores_aug']:
            res['poe'] = sum(res['scores_aug'].values()) / len(res['scores_aug'])

    return res

def infer_task(
    keys,
    dataset,
    fmt_opts,
    aug_score_opts=None,
    pass_guess=True,
    print_func=print,
    use_symbolic=False,
    early_stopping=True,
    **kwargs,
):
    unique_results = {}
    best_guess = (None, float('inf'))

    poe_solutions = []
    stop_search = False

    found_non_violating = False

    def drop_violating_from_unique():
        print("Found non-violating solution. Delete all pevious candidates that violated")
        to_delete = [h for h, r in unique_results.items() if r.get('violating', False)]
        for h in to_delete:
            del unique_results[h]

    for key in keys:
        if stop_search:
            break

        # format task
        key, fmt = dataset.get_task(key, **fmt_opts)
        input_len = dataset.count_tokens(fmt['input'])
        reply_len = dataset.count_tokens(fmt['reply']) if 'reply' in fmt else '?'

        # get current best guess
        guess = None
        if pass_guess and best_guess[0] is not None:
            guess = dataset.get_task(key, reply=best_guess[0], **fmt_opts)[1]['text']
            assert guess.startswith(fmt['input'])

        # initialize symbolic
        key_num, *tf = key.split('.')
        base_key, reply_num = dataset.get_base_key_and_reply_num(key_num)
        train_pairs = dataset.challenge[base_key]['train']
        test_input = dataset.challenge[base_key]['test'][reply_num]['input']

        augmented_example_inputs = [dataset.transform_array(x["input"], tf).tolist() for x in train_pairs]
        augmented_example_outputs = [dataset.transform_array(x["output"], tf).tolist() for x in train_pairs]
        augmented_test_input = dataset.transform_array(test_input, tf).tolist()

        symbolic = ArcSymbolicConstraints(
            key_num,
            augmented_example_inputs,
            augmented_example_outputs,
            augmented_test_input,
            track_time = True
        )
        characteristics = symbolic.find_characteristics()

        data = infer_single(
            prompt=fmt['input'],
            symbolic=symbolic,
            guess=guess,
            enforce_symbolic=use_symbolic and found_non_violating,
            **kwargs
        )

        #symbolic.report_time_analysis()
        if symbolic.time_analysis["find_characteristics_total"] + symbolic.time_analysis["check_characteristics_total"] > 0.1:
            print("Unusual long time for symbolic part!")
            symbolic.report_time_analysis()

        # loop over inference outputs
        for i, (sequence, score, violating_constraint) in enumerate(data):

            if stop_search:
                break

            if use_symbolic and found_non_violating and (violating_constraint is not None):
                continue

            if 1 < kwargs.get('num_return_sequences', 1):  # recalc incorrect beam search scores
                score = calc_score(input=fmt['input'], reply=sequence, **kwargs)

            output, correct, corr_info = dataset.decode(sequence, fmt_opts['lines_sep'], key)

            res = None
            if output is not None:

                # Hashable representation of the output
                hashable = tuple(map(tuple, output))

                # If this is the first time we see this output, create entry; otherwise reuse.
                res = process_output(
                    unique_results, dataset, fmt_opts, aug_score_opts,
                    key, score, output, correct, kwargs
                )

                res['violating'] = (violating_constraint is not None)

                if use_symbolic and not found_non_violating and (violating_constraint is None):
                    found_non_violating = True
                    drop_violating_from_unique()

                # update best guess
                if pass_guess:
                    new_score = min(res['scores_inf'].values())
                    if new_score < best_guess[1]:
                        best_guess = (res['output'], new_score)

                # Early stopping logic based on PoE
                if early_stopping and 'poe' in res:
                    poe_solutions.append((hashable, res['poe']))

                    # Only consider the *first three* solutions that actually have a PoE
                    if len(poe_solutions) == 3:
                        hashes = [h for (h, _) in poe_solutions]
                        poes = [p for (_, p) in poe_solutions]

                        # All three outputs must be the same
                        same_outputs = (len(set(hashes)) == 1)

                        # Convert PoE scores to probabilities via exp(-poe)
                        high_conf = all(np.exp(-p) >= 0.5 for p in poes)

                        if same_outputs and high_conf:
                            stop_search = True
                            print("ABOARD SEARCH Because the first three candidates are the same and high confidence")

                # print some info (still show violating info)
                token_info = f" in:{input_len:>4} out:{dataset.count_tokens(sequence):>3}/{reply_len:>3}"
                shape_info = f'{output.shape[0]:>2}x{output.shape[1]:<2}' if output is not None else '--x--'
                inf_sc = f"{min(np.exp(-score), 0.99):>3.0%}"
                poe_sc = (
                    f"{min(np.exp(-res['poe']), 0.999999):>8.4%}"
                    if res and 'poe' in res else '-'*8
                )
                print_func(
                    f"{token_info} > {shape_info} {corr_info} "
                    f"p={inf_sc} poe={poe_sc} [{key}.out{i}], v_c = {violating_constraint}"
                )

    return list(unique_results.values()), list(characteristics.keys())


def inference_run(dataset, fmt_opts, max_new_tokens=None, eval_tools = [],  print_func=print, use_symbolic = False, **kwargs):
    # set token limits
    if max_new_tokens is None:
        max_new_tokens = dataset.max_new_tokens(**fmt_opts)
    if 'max_tokens' in fmt_opts:
        fmt_opts = {**fmt_opts, 'max_tokens': fmt_opts['max_tokens'] - max_new_tokens, 'len_name': 'input'}

    # iterate over dataset
    results = {}
    for base_key, tasks in dataset.grouped_keys().items():
        results[base_key] = []
        for task_num, task in enumerate(tasks):
            start = time.perf_counter()
            res, holding_constraints = infer_task(keys=task, dataset=dataset, fmt_opts=fmt_opts, max_new_tokens=max_new_tokens,
                             print_func=print_func,use_symbolic = use_symbolic, **kwargs)
            required_time = time.perf_counter() - start
            results[base_key].append(res)

            for eval_tool in eval_tools:
                if eval_tool.specific_constraint == None or eval_tool.specific_constraint in holding_constraints:
                    eval_tool.total_time += required_time
                    eval_tool.process_result(res, name=f'{base_key}_{task_num}', value=1/len(tasks), print_func=print_func)
    return results
