
from collections import Counter
import numpy as np

import time


class ArcSymbolicConstraints:
    def __init__(self,key, example_inputs, example_outputs, test_input, track_time = False):

        """
        This class is created for a specific problem.
        It first find a list of characteristics (constraints) that hold for this problem
        Later it can be checked if the input and (potentially unfinished) output violate 
        the found constraints.
        """

        self.example_inputs = example_inputs
        self.example_outputs = example_outputs
        self.key = key
        self.test_input = test_input

        self.current_state = [[]]
        self.current_histogramm = {i:0 for i in range(10)}
        self.current_color_map = dict()
        self.current_color_map_counts = {i: {j: 0 for j in range(10)} for i in range(10)}
        self.current_is_finished = False

        # work with lists, not nupy
        if isinstance(self.test_input, np.ndarray):
            self.test_input = self.test_input.tolist()
        for i in range(len(self.example_inputs)):
            if isinstance(self.example_inputs[i], np.ndarray):
                self.example_inputs[i] = self.example_inputs[i].tolist()
            if isinstance(self.example_outputs[i], np.ndarray):
                self.example_outputs[i] = self.example_outputs[i].tolist()

        self.histogramm_input = self.get_color_histogram(self.test_input)
        self.characteristics = dict()

        # Make more efficient by computing histogram and color map only once for examples
        self.color_maps = []
        self.histogramms = []
        for example_input, example_output in zip(self.example_inputs, self.example_outputs):

            input_histogramm = self.get_color_histogram(example_input)
            output_histogramm = self.get_color_histogram(example_output)
            self.histogramms.append({'input' : input_histogramm, 'output' : output_histogramm})

            # only works if they have the same size
            if len(example_input) != len(example_output) or len(example_input[0]) != len(example_output[0]):
                self.color_maps.append(None)
            else:
                self.color_maps.append( self.get_color_map(example_input, example_output) )

        self.all_constraints = [
            self.only_one_color_changes, # 1
            self.a_specific_color_does_not_change, # 2
            self.output_is_different_to_input, # 3
            self.color_histogramm_stays_the_same_except_one, # 4
            self.count_of_a_specific_color_does_not_change, # 5
            self.no_lonely_pixels_in_output, # 6
            self.background_color_decreases, # 7
            self.output_has_specific_set_of_colors, # 8
            self.count_of_a_specific_color_changes_by_specific_amount, # 9
            self.output_is_horizontally_mirrored, # 10
            self.output_is_vertically_mirrored, # 11
            self.color_histogramm_stays_the_same, # 12
        ]

        self.all_constraint_checks = [
            self.check_only_one_color_changes, # 1
            self.check_a_specific_color_does_not_change, # 2
            self.check_output_is_different_to_input, # 3
            self.check_color_histogramm_stays_the_same_except_one, # 4
            self.check_count_of_a_specific_color_does_not_change, # 5
            self.check_no_lonely_pixels_in_output, # 6
            self.check_background_color_decreases, # 7
            self.check_output_has_specific_set_of_colors, # 8
            self.check_count_of_a_specific_color_changes_by_specific_amount, # 9
            self.check_output_is_horizontally_mirrored, # 10
            self.check_output_is_vertically_mirrored, # 11
            self.check_color_histogramm_stays_the_same, # 12
            self.check_is_rectangle # always
        ]

        self.track_time = track_time
        #track the time spend in diffent parts of the code
        self.time_analysis = {
            "find_characteristics_total" : 0,
            "check_characteristics_total" : 0,
            "add_new_token" : 0,
            "remove_last_token" : 0,
        }
        for c in self.all_constraints + self.all_constraint_checks:
            self.time_analysis[c.__name__] = 0

    def report_time_analysis(self):

        if not self.track_time:
            print("Time tracking is disabled (track_time = False).")
            return


        # Print all entries, sorted by time (largest first)
        print("\nTime analysis (per key):")
        for name, t in sorted(self.time_analysis.items(),
                              key=lambda kv: kv[1],
                              reverse=True):
            print(f" {t:.6f}s -> {name:35s}")

        # Compute a grand total
        total_keys = [k for k in self.time_analysis.keys() if k.endswith("_total")]
        grand_total = sum(self.time_analysis[k] for k in total_keys)

        print(f"  {'sum':35s}: {grand_total:.6f} s")

        print()

    def add_new_token(self, token):
        """
        Add a new token to the current state and update the color map and histogramm accordingly
        """

        if self.track_time:
            start = time.perf_counter()

        if token == '<|eot_id|>' or token == "</s>" or token == '<｜end▁of▁sentence｜>':
            # remove new list in lat line
            self.current_state.pop()
            self.current_is_finished = True
        elif token == "\n":
            self.current_state.append([])
        else:
            try:
                token = int(token)
            except ValueError:
                print(f"Error, could not interepret token {token} for current, use 0 instead")
                token = 0
            self.current_state[-1].append(token)

            # update histogram
            self.current_histogramm[token] +=1

            # update color map
            pos_x = len(self.current_state[-1]) -1
            pos_y = len(self.current_state) -1
            try:
                c_in_input = self.test_input[pos_y][pos_x]
                self.current_color_map_counts[c_in_input][token] +=1
                if self.current_color_map_counts[c_in_input][token] == 1:
                    # we have to add to the color map
                    if c_in_input in self.current_color_map:
                        self.current_color_map[c_in_input].add(token)
                    else:
                        self.current_color_map[c_in_input] = set([token])
            except IndexError:
                pass

        if self.track_time:
            self.time_analysis["add_new_token"] += time.perf_counter() - start

        return
    
    def remove_last_token(self):
        """
        Remove the last token that was added and update the color map and histogramm accordingly.
        If this function is correctly used after each time "add_new_token" is called it should not lead to any errors
        """

        if self.track_time:
            start = time.perf_counter()

        if self.current_is_finished:
            # the last token was the <eos> token
            self.current_is_finished = False
            self.current_state.append([])
        elif len(self.current_state[-1]) == 0:
            # the last token was a linebreak
            self.current_state.pop()
        else:
            token = self.current_state[-1].pop()

            pos_x = len(self.current_state[-1])
            pos_y = len(self.current_state) -1


            try:
                c_in_input = self.test_input[pos_y][pos_x]
                self.current_color_map_counts[c_in_input][token] -=1
                if self.current_color_map_counts[c_in_input][token] == 0:
                    # We have to remove from the colow map now:
                    self.current_color_map[c_in_input].remove(token)

            except IndexError:
                pass

            # update histogram
            self.current_histogramm[token] -=1

        if self.track_time:
            self.time_analysis["remove_last_token"] += time.perf_counter() - start

        return

    def find_characteristics(self):

        if self.track_time:
            start = time.perf_counter()

        # There should be at least three examples to try to detect matching characteristics
        if len(self.example_inputs) < 3:
            return self.characteristics

        for constraint in self.all_constraints:
            if self.track_time:
                start_c = time.perf_counter()
            res = constraint()
            if res != False:
                # This constraint was holding
                self.characteristics[res[0]] = res[1]
            if self.track_time:
                self.time_analysis[constraint.__name__] += time.perf_counter() - start_c

        if self.track_time:
            self.time_analysis["find_characteristics_total"] += time.perf_counter() - start

        return self.characteristics


    def check_characteristics(self, new_token):
        """
        checks if adding new_token (a color linebreak or eos) is violating a constraint
        """

        if self.track_time:
            start = time.perf_counter()

        self.add_new_token(new_token)

        color_map = None
        if self.can_complete_to_match_size(self.test_input,self.current_state):
            color_map = self.current_color_map

        background_color = None
        if self.current_is_finished:
            # we only know what the most frequent color is if the image is finished
            background_color =  max(self.current_histogramm, key=self.current_histogramm.get)

        if new_token == "\n" or new_token == "<|eot_id|>" or new_token == "</s>" or new_token == '<｜end▁of▁sentence｜>':
            new_token = None
            pixel_at_input = None
        else:
            try:
                new_token = int(new_token)
            except ValueError:
                return False, f"unknown token '{new_token}'" # unknown token
            pos_x = len(self.current_state[-1]) -1
            pos_y = len(self.current_state) -1
            try:
                pixel_at_input = self.test_input[pos_y][pos_x]
            except IndexError:
                pixel_at_input = None
        
        
        ret_value = (True, None)
        for constraint in self.all_constraint_checks:
            if self.track_time:
                start_c = time.perf_counter()

            constraint_args = {
                "inference_input" : self.test_input,
                "uncomplete_output" : self.current_state,
                "finished" : self.current_is_finished,
                "color_map" : color_map,
                "background_color" : background_color,
                "pixel_at_input" : pixel_at_input,
                "histogramm_input" : self.histogramm_input,
                "histogramm_output" : self.current_histogramm,
                "new_pixel" : new_token,

            }


            if not constraint(**constraint_args):
                ret_value = (False, constraint.__name__)
                # print(f"violates constraint: {constraint}\n{constraint_args}\n{self.key}")
                if self.track_time:
                    self.time_analysis[constraint.__name__] += time.perf_counter() - start_c
                break

            if self.track_time:
                self.time_analysis[constraint.__name__] += time.perf_counter() - start_c
        
        self.remove_last_token()

        if self.track_time:
            self.time_analysis["check_characteristics_total"] += time.perf_counter() - start

        return ret_value

    # utils
    def get_color_map(self,input_grid, output_grid):
        """
        Returns the color mapping from one grid to another.

        For example: {0 : {0,2,4}, 1 : {2}} would indicate that all pixels
        that were 0 in the input become either 0,2 or 4 and app pixels that 
        were 1 in the input become 2 in the output.

        Also works, if the output grid is smaller than the input grid
        """

        mapping = {}
        for i in range(len(output_grid)):
            for j in range(len(output_grid[i])):
                in_px = input_grid[i][j]
                out_px = output_grid[i][j]

                if in_px in mapping:
                    mapping[in_px].add(out_px)
                else:
                    mapping[in_px] = set([out_px])
        return mapping


    def can_complete_to_match_size(self,inference_input, uncomplete_output):
        """
        Checks if the output can be completed in a way that its size matches the input
        """

        if len(uncomplete_output) == 0:
            #empty so far
            return True

        n_rows = len(uncomplete_output)
        if uncomplete_output[-1] == []:
            n_rows -= 1 # empty row does not count

        if n_rows > len(inference_input):
            # the uncomplete_output has more rows than the input image
            return False
        
        len_row = len(inference_input[0])

        for i, row_output in enumerate(uncomplete_output):

            # For the last row it is fine to be shorter because it is still being generated
            if i == len(uncomplete_output) -1 and len(row_output) < len_row:
                continue

            # check that the sizes match
            if len(row_output) != len_row:
                return False
        return True
    
    def get_color_histogram(self,grid):
        flat = [item for sublist in grid for item in sublist]
        return Counter(flat)
    
    def check_is_rectangle(self, uncomplete_output, **_):
        # check if rectangle:
        if len(uncomplete_output) <= 1:
            return True

        len_first_row = len(uncomplete_output[0])
        len_current_row = len(uncomplete_output[-1])

        if len_current_row > len_first_row:
            return False

        if len_current_row == 0:
            # we just had a linebreak check the row before
            len_row_before = len(uncomplete_output[-2])
            if len_row_before < len_first_row:
                return False
        return True


    # Constraint 1:
    def most_frequent_color_does_not_change(self):
        """
        Checks if the most frequent color (the background color) does nto change
        """
        
        for color_map, hist in zip(self.color_maps,self.histogramms):

            # only works if they have the same size
            if color_map == None:
                return False
            
            histogramm_input = hist['input']
            most_frequent_color = max(histogramm_input, key=histogramm_input.get)

            if color_map[most_frequent_color] != {most_frequent_color}:
                return False

        return "most_frequent_color_does_not_change", None

    def check_most_frequent_color_does_not_change(self, histogramm_input, color_map, **_):
        """
        Checks if the most frequent color does not change

        """

        if "most_frequent_color_does_not_change"not in self.characteristics.keys():
            # nothing to do
            return True
        
        if color_map == None:
            return True # happends if sizes of input and output dont match

        most_frequent_color_input = max(histogramm_input, key=histogramm_input.get)

        if most_frequent_color_input not in color_map.keys() or len(color_map[most_frequent_color_input]) == 0:
            return True

        return color_map[most_frequent_color_input] == {most_frequent_color_input} # should only map to itself
    
    # Constraint 2:
    def only_one_color_changes(self):
        """
        Checks if only one color changes
        """

        for color_map in self.color_maps:

            # only works if they have the same size
            if color_map == None:
                return False

            changing_color = None
            for key, value in color_map.items():
                if len(value) == 0:
                    continue
                if len(value) == 1 and key == next(iter(value)):
                    # All good, This color did not change
                    continue
                if changing_color != None and changing_color != key:
                    # We found a second changing color
                    return False
                changing_color = key
                
        return "only_one_color_changes", None # no arguments needed

    def check_only_one_color_changes(self, color_map, **_):
        """
        Checks if only one color changes. 
        
        :param uncomplete_output: the output of the transition that may not be completly filled out yet
        """

        if "only_one_color_changes" not in self.characteristics.keys():
            # nothing to do
            return True

        if color_map == None:
            return True

        changing_color = None
        for key, value in color_map.items():
            if len(value) == 0:
                continue
            if len(value) == 1 and key == next(iter(value)):
                # All good, This color did not change
                continue
            if changing_color != None:
                # We found a second changing color
                return False
            changing_color = key

        return True

    # Constraint 3:
    def only_one_color_changes_strictly(self):
        """
        Checks if only one color changes, and the changing color must always change in a different color
        """

        for color_map in self.color_maps:

            # only works if they have the same size
            if color_map == None:
                return False

            changing_color = None
            for key, value in color_map.items():
                if len(value) == 1 and key == next(iter(value)):
                    # All good, This color did not change
                    continue
                if changing_color != None and changing_color != key:
                    # We found a second changing color
                    return False
                if key in value:
                    # This color does not change into something else
                    return False
                changing_color = key
                
        return "only_one_color_changes_strictly", None # no arguments needed

    def check_only_one_color_changes_strictly(self, color_map, **_):
        """
        Checks if only one color changes, and the changing color must always change in a different color
        
        :param uncomplete_output: the output of the transition that may not be completly filled out yet
        """

        if "only_one_color_changes_strictly" not in self.characteristics.keys():
            # nothing to do
            return True

        if color_map == None:
            return False

        changing_color = None
        for key, value in color_map.items():
            if len(value) == 0:
                continue # this color did not change
            if len(value) == 1 and key == next(iter(value)):
                # All good, This color did not change
                continue
            if changing_color != None and changing_color != key:
                # We found a second changing color
                return False
            if key in value:
                # This color does not change into something else
                return False
            changing_color = key

        return True

    # Constraint 4:
    def a_specific_color_does_not_change(self):
        """
        Checks if a specific color does not change. And also the specific color appears in 
        all images. There should be at least one example that does not update the rule.
        """

        # colors are represented with the numbers 1 to 0
        non_changing_color = [i for i in range(10)]

        did_no_update_rule = 0

        for color_map in self.color_maps:

            # only works if they have the same size
            if color_map == None:
                return False

            update = False
            for color in list(non_changing_color):
                if color not in color_map.keys():
                    # The color does not appear, so remove
                    non_changing_color.remove(color)
                    update = True
                    continue
                if len(color_map[color]) != 1:
                    # The color changes into multiple things
                    non_changing_color.remove(color)
                    update = True
                    continue

                if next(iter(color_map[color])) != color:
                    # This color changes into something else
                    non_changing_color.remove(color)
                    update = True
            
            if not update:
                did_no_update_rule +=1

            if len(non_changing_color) == 0:
                return False
        
        if did_no_update_rule <1:
            return False

        return "a_specific_color_does_not_change", {"non_changing_colors" : non_changing_color}

    def check_a_specific_color_does_not_change(self, pixel_at_input, new_pixel, **_):
        """
        Checks if a specific color does not change.
        
        :param uncomplete_output: the output of the transition that may not be completly filled out yet
        """

        if "a_specific_color_does_not_change" not in self.characteristics.keys():
            # nothing to do
            return True

        if new_pixel == None or pixel_at_input == None:
            return True # Can not compare
        
        if pixel_at_input in self.characteristics["a_specific_color_does_not_change"]["non_changing_colors"]:
            return pixel_at_input == new_pixel
        
        return True

    # Constraint 5:
    def all_colors_are_preserved(self):
        """
        Checks if the colors in the input is also always present in the output
        """

        for hist in self.histogramms:

            histogramm_input = hist['input']
            histogramm_output = hist['output']

            for color in histogramm_input.keys():
                if color not in histogramm_output.keys():
                    return False
                
        return "all_colors_are_preserved", None

    def check_all_colors_are_preserved(self, histogramm_input,histogramm_output, finished,**_):
        """
        Checks if the colors in the input is also always present in the output
        :param uncomplete_output: the output of the transition that may not be completly filled out yet
        :param finished: If the output is complete
        """

        if "all_colors_are_preserved" not in self.characteristics.keys():
            # nothing to do
            return True

        if not finished:
            # we can not chack this if image is not finished
            return True

        for color in histogramm_input.keys():
            if color not in histogramm_output.keys():
                return False
        
        return True

    # Constraint 6:
    def output_is_different_to_input(self):
        """
        Checks if all training outputs are different from their corresponding inputs.
        If all training examples show a transformation, we expect the test case to also differ from its input.
        """
        for inp, out in zip(self.example_inputs, self.example_outputs):
            if inp == out:
                return False  # Found a training pair where input equals output

        return "output_is_different_to_input", None
    
    def check_output_is_different_to_input(self, inference_input, uncomplete_output, finished, **_):
        """
        Checks if the current output is different from the input.
        """
        if "output_is_different_to_input" not in self.characteristics.keys():
            return True

        if not finished:
            return True  # Can't say yet, output might still change

        return inference_input != uncomplete_output

    # Constraint 7:
    def a_new_color_is_added(self):
        """
        Checks if a new color is added in the output compared to the input.
        """
        for hist in self.histogramms:

            input_colors = hist['input']
            output_colors = hist['output']

            if any(color not in input_colors.keys() for color in output_colors.keys()):
                continue  # New color found, this example satisfies the constraint
            else:
                return False  # No new color added in this example, constraint fails

        return "a_new_color_is_added", None  # All examples added at least one new color

    def check_a_new_color_is_added(self, histogramm_input,histogramm_output, finished,**_):
        """
        Checks if a new color is added
        :param uncomplete_output: the output of the transition that may not be completly filled out yet
        :param finished: If the output is complete
        """

        if "a_new_color_is_added"not in self.characteristics.keys():
            # nothing to do
            return True

        if not finished:
            # we can not chack this if image is not finished
            return True

        for color in histogramm_output.keys():
            if color not in histogramm_input.keys():
                return True
        
        return False
    
    # Constraint 8:
    def color_histogramm_stays_the_same_except_one(self):
        """
        Checks if the count of colors is the same for the input and output,
        except for a single color.
        """
        for hist in self.histogramms:

            histogramm_input = hist['input']
            histogramm_output = hist['output']

            freebie = True
            for color, count in histogramm_input.items():
                if color not in histogramm_output.keys() or histogramm_output[color] != count:
                    if freebie == False:
                        return False
                    freebie = False
                
        return "color_histogramm_stays_the_same_except_one", None

    def check_color_histogramm_stays_the_same_except_one(self, histogramm_input, histogramm_output, finished, **_):
        """
        Checks if the count of colors is the same for the input and output,
        except for a single color.
        :param uncomplete_output: the output of the transition that may not be completly filled out yet
        :param finished: If the output is finished
        """

        if "color_histogramm_stays_the_same_except_one" not in self.characteristics.keys():
            # nothing to do
            return True

        freebie = True
        if not finished:
            for color, max in histogramm_input.items():

                if color in histogramm_output.keys() and histogramm_output[color] > max:
                    if freebie == False:
                        return False
                    freebie = False
            
        else:
            for color, max in histogramm_input.items():
                if color not in histogramm_output.keys() or histogramm_output[color] != max:
                    if freebie == False:
                        return False
                    freebie = False
        
        return True

    # Constraint 9:
    def count_of_a_specific_color_does_not_change(self):
        """
        Checks if the count of a specific color does not change. And this specific color
        always appears.
        """

        # colors are represented with the numbers 1 to 0
        counts_do_not_change = [i for i in range(10)]

        deleted_in_all = True

        for hist in self.histogramms:

            histogramm_input = hist['input']
            histogramm_output = hist['output']

            deleted_here = False
            for color in list(counts_do_not_change):
                if color not in histogramm_input.keys() or color not in histogramm_output.keys() or histogramm_input[color] != histogramm_output[color]:
                    counts_do_not_change.remove(color)
                    deleted_here = True
            if not deleted_here:
                deleted_in_all = False
            if len(counts_do_not_change) == 0:
                return False
            
        if deleted_in_all:
            return False # have at least one example were the rule could be applied without modifying the rule
        
        return "count_of_a_specific_color_does_not_change", {"colors" : counts_do_not_change}

    def check_count_of_a_specific_color_does_not_change(self, histogramm_input, histogramm_output, finished, **_):
        """
        Checks if the count of a specific color does not change. And this specific color
        always appears.
        :param uncomplete_output: the output of the transition that may not be completly filled out yet
        :param finished: If the output is complete
        """

        if "count_of_a_specific_color_does_not_change" not in self.characteristics.keys():
            # nothing to do
            return True
        
        colors_of_unchanging_counts = self.characteristics["count_of_a_specific_color_does_not_change"]["colors"]

        for color in colors_of_unchanging_counts:

            if color not in histogramm_input.keys():
                # could happen, in this case the constraint was wrong, let's ignore this
                continue

            if histogramm_output[color] > histogramm_input[color]:
                return False
            
            if finished and histogramm_output[color] != histogramm_input[color]:
                return False

        return True
    
    def _has_lonely_pixel(self, grid):
        """
        Returns True if some pixel is surrounded only by other-colored neighbors (up,down,left,right) or borders.
        """
        rows = len(grid)
        for i in range(rows):
            for j in range(len(grid[i])):
                if self._is_lonely_pixel(grid, i, j):
                    return True
        return False

    def _is_lonely_pixel(self, grid, i, j):
        """
        Returns True if the pixel at (i, j) is lonely (surrounded only by pixels of a different color) an not at boarder.
        """
        c = grid[i][j]

        neighbors = []

        if i > 0:
            neighbors.append(grid[i - 1][j])
        if i < len(grid) - 1:
            neighbors.append(grid[i + 1][j])
        if j > 0:
            neighbors.append(grid[i][j - 1])
        if j < len(grid[i]) - 1:
            neighbors.append(grid[i][j + 1])


        return len(neighbors) == 4 and all(n != c for n in neighbors)

    # Constraint 10:
    def no_lonely_pixels_in_output(self):
        """
        Adds constraint if:
        - No training output has a lonely pixel (a pixel only sourrounded different colours)
        and
            - test input does NOT have a lonely pixel 
            or
            - some training iputs have loneley pixels
        """

        # Check training outputs
        for output in self.example_outputs:
            if self._has_lonely_pixel(output):
                return False

        # So no training output had a loneley pixel
        if not self._has_lonely_pixel(self.test_input):
            # The test input has no loneley pixel, lets enfore the constraint
            return "no_lonely_pixels_in_output", None
        # The test input does have a loneley pixel

        for input_image in self.example_inputs:
            if self._has_lonely_pixel(input_image):
                return "no_lonely_pixels_in_output", None  # Safe to add constraint

        # No test training input had a loneley pixel but the test input had a loneley pixel, it could e a task were you have top
        # copy a lot from the input. Therefore it could be that there mut be a loneley pixel in the input. Better not enfoce the cosntraint
        return False

    def check_no_lonely_pixels_in_output(self, uncomplete_output, finished, **_):

        if "no_lonely_pixels_in_output" not in self.characteristics.keys():
            return True

        if finished:
            if self._has_lonely_pixel(uncomplete_output):
                return False
        return True
    
    # Constraint 11:
    def background_color_decreases(self):
        """
        Checks if the number of times the background color appears is less in the output
        """
        for hist in self.histogramms:
            bgc = max(hist["input"], key=hist["input"].get)
            count_input = hist["input"][bgc]
            if bgc in hist["output"] and hist["output"][bgc] >= count_input:
                #violates the constraint
                return False
        bgc_input = max(self.histogramm_input, key=self.histogramm_input.get)
        count_input = self.histogramm_input[bgc_input]
        
        return "background_color_decreases", {'color' : bgc_input, 'count' : count_input}

    def check_background_color_decreases(self, histogramm_output, **_):
        """
        Checks if the number of times the background color appears decreses in the output
        """

        if "background_color_decreases" not in self.characteristics.keys():
            # nothing to do
            return True
        
        bgc = self.characteristics["background_color_decreases"]["color"]
        count_input = self.characteristics["background_color_decreases"]["count"]
        
        return bgc not in histogramm_output or histogramm_output[bgc] < count_input
    
    # Constraint 12:
    def output_has_specific_set_of_colors(self):
        """
        A constraint that the output colors always come from the same specific set
        across all training examples.
        """
        color_set = set(self.histogramms[0]['output'].keys())
        for hist in self.histogramms[1:]:
            if set(hist['output'].keys()) != color_set:
                return False


        # Extra Rule: If all examples-inputs had the same set of color-set but the test-input has a different one it is saver to ignore this constraint
        input_color_set = set(self.histogramms[0]['input'].keys())
        all_inputs_examples_have_the_same_color_map = True
        for hist in self.histogramms[1:]:
            if set(hist['input'].keys()) != input_color_set:
                # inputs have different color sets
                all_inputs_examples_have_the_same_color_map = False
                break
        if  all_inputs_examples_have_the_same_color_map and set(self.histogramm_input.keys()) != input_color_set:
            return False
        
        return "output_has_specific_set_of_colors", {'color_set': color_set}

    def check_output_has_specific_set_of_colors(self, finished, histogramm_output, **_):
        """
        Checks that all output colors are from a specific set, and (if finished) exactly match that set.
        """
        if "output_has_specific_set_of_colors" not in self.characteristics.keys():
            return True

        color_set = self.characteristics["output_has_specific_set_of_colors"]["color_set"]
        output_colors = set([key for key,value in histogramm_output.items() if value > 0])

        if not output_colors.issubset(color_set):
            return False

        if finished and output_colors != color_set:
            return False

        return True
    
    # Constraint 13:
    def count_of_a_specific_color_changes_by_specific_amount(self):
        """
        Checks if any color consistently changes from a specific non zero valuee to a specific other non-zero value.
        There must be at least three exampels
        """

        all_color_counts = {}

        for color in range(10):
            count_input = self.histogramm_input[color]
            count_output = self.histogramms[0]['output'][color]

            if count_input == 0 or count_output == 0:
                continue

            is_consistent = True
            for hist in self.histogramms:

                if  hist['input'].get(color, 0) != count_input:
                    is_consistent = False
                    break
                if  hist['output'].get(color, 0) != count_output:
                    is_consistent = False
                    break

            if is_consistent:
                all_color_counts[color] = count_output

        return ("count_of_a_specific_color_changes_by_specific_amount", all_color_counts) if all_color_counts else False

    def check_count_of_a_specific_color_changes_by_specific_amount(self, histogramm_output, finished, **_):
        """
        Checks if color consistently changes from a specific amount to a specific amount
        """

        if "count_of_a_specific_color_changes_by_specific_amount" not in self.characteristics:
            # nothing to do
            return True
        
        all_color_counts = self.characteristics["count_of_a_specific_color_changes_by_specific_amount"]

        for color, count_output in all_color_counts.items():

            if histogramm_output.get(color, 0) > count_output:
                return False

            if finished and histogramm_output.get(color, 0) != count_output:
                return False
        return True
    
    def contains_at_least_n_pixels(self,image,n = 9):
        return len(image) * len(image[0]) >= n

    # Constraint 14:
    def output_is_horizontally_mirrored(self):
        """
        Checks if all training outputs are horizontally mirrored (symmetric about the vertical axis).
        """
        for out in self.example_outputs:
            if not self.contains_at_least_n_pixels(out):
                return False
            for row in out:
                if row != row[::-1]:
                    return False
        return "output_is_horizontally_mirrored", None
    
    def check_output_is_horizontally_mirrored(self, uncomplete_output,finished, **_):
        """
        Incremental check: only verify symmetry of the last completed row.
        """
        if "output_is_horizontally_mirrored" not in self.characteristics.keys():
            return True

        if not uncomplete_output:
            return True

        if finished:
            last_completed_row = uncomplete_output[-1]
        elif len(uncomplete_output[-1]) == 1 and len(uncomplete_output) > 1:
            last_completed_row = uncomplete_output[-2]
        else:
            return True
        
        return last_completed_row == last_completed_row[::-1]
    
    # Constraint 15:
    def output_is_vertically_mirrored(self):
        """
        Checks if all training outputs are vertically mirrored (symmetric about the horizontal axis).
        """

        n_examples = 0
        for out in self.example_outputs:
            if not self.contains_at_least_n_pixels(out):
                return False
            if out != out[::-1]:
                return False
            # do not use if output has less than 9 pixels (could be coincidence)
            if len(out) * len(out[0]) < 9:
                continue
            n_examples +=1
            

        return "output_is_vertically_mirrored", None
    
    def check_output_is_vertically_mirrored(self, uncomplete_output, finished, **_):
        """
        Full check after image is completed.
        """
        if "output_is_vertically_mirrored" not in self.characteristics.keys():
            return True

        if not finished:
            return True

        return uncomplete_output == uncomplete_output[::-1]

    # Constraint 16:
    def output_is_point_symmetric(self):
        """
        Checks if all training outputs are point symmetric (180-degree rotational symmetry).
        """
        for out in self.example_outputs:
            if not self.contains_at_least_n_pixels(out):
                return False
            if out != [row[::-1] for row in out[::-1]]:
                return False
        return "output_is_point_symmetric", None
    
    def check_output_is_point_symmetric(self, uncomplete_output, finished, **_):
        """
        Checks if the completed output is point symmetric.
        """
        if "output_is_point_symmetric" not in self.characteristics.keys():
            return True

        if not finished:
            return True

        rotated_180 = [row[::-1] for row in uncomplete_output[::-1]]
        return uncomplete_output == rotated_180

    # Constraint 17:
    def color_histogramm_stays_the_same(self):
        """
        Checks if the count of colors is the same for the input and output.
        This means the output is basically a rearrangement of pixels.
        """
        for hist in self.histogramms:

            histogramm_input = hist['input']
            histogramm_output = hist['output']

            if histogramm_input != histogramm_output:
                return False
                
        return "color_histogramm_stays_the_same", None

    def check_color_histogramm_stays_the_same(self, histogramm_input, histogramm_output, color_map, **_):
        """
        Checks if the count of colors is the same for the input and output
        :param uncomplete_output: the output of the transition that may not be completly filled out yet
        """

        if "color_histogramm_stays_the_same" not in self.characteristics.keys():
            # nothing to do
            return True

        if color_map == None:
            # sizes must match
            return False

        n_missing_pixels = sum(histogramm_input.values()) - sum(histogramm_output.values())

        for color, max in histogramm_input.items():

            if color not in histogramm_output.keys():
                n_missing_pixels -= max
                continue

            if histogramm_output[color] > max:
                return False
            n_missing_pixels -= max - histogramm_output[color]
        
        return n_missing_pixels >= 0


def main():

    import json

    # Adjust these paths if needed
    challenges_path = "input/arc-prize-2024/arc-agi_evaluation_challenges.json"
    solutions_path  = "input/arc-prize-2024/arc-agi_evaluation_solutions.json"

    with open(challenges_path, "r") as f:
        challenges = json.load(f)
    with open(solutions_path, "r") as f:
        solutions = json.load(f)

    # Dummy instance just to get the list of constraint methods
    dummy_symbolic = ArcSymbolicConstraints(
        key="",
        example_inputs=[],
        example_outputs=[],
        test_input=[],
        track_time=False,
    )

    # Use names instead of bound methods from dummy_symbolic
    constraint_names = [c.__name__ for c in dummy_symbolic.all_constraints]

    # Global histogram: per-constraint and total
    histogram = {}
    for name in constraint_names:
        histogram[name] = {"holds": 0, "exceptions": []}
    histogram["total"] = 0

    for key, task in challenges.items():
        # Training examples
        train_pairs = task["train"]
        example_inputs = [p["input"] for p in train_pairs]
        example_outputs = [p["output"] for p in train_pairs]

        # Test inputs
        test_inputs = [t["input"] for t in task["test"]]

        # Ground truth outputs – handle both list-of-dicts and raw grids
        sol_entry = solutions[key]
        test_outputs = []
        for t in sol_entry:
            if isinstance(t, dict) and "output" in t:
                test_outputs.append(t["output"])
            else:
                test_outputs.append(t)

        if len(example_inputs) < 3:
            continue

        for test_input, test_output in zip(test_inputs, test_outputs):
            histogram["total"] += 1

            for cname in constraint_names:
                # Fresh instance for this (task, test, constraint)
                symbolic = ArcSymbolicConstraints(
                    key="",
                    example_inputs=example_inputs,
                    example_outputs=example_outputs,
                    test_input=test_input,
                    track_time=False,
                )

                # Re-bind the constraint method on this instance
                constraint_method = getattr(symbolic, cname)
                symbolic.all_constraints = [constraint_method]

                # Infer whether this constraint appears to hold on training data
                res = symbolic.find_characteristics()
                if len(res) == 0:
                    continue  # constraint not inferred for this task

                # We inferred it -> count this as a case for this constraint
                histogram[cname]["holds"] += 1

                # Now check if GT test_output violates it
                ok = True
                for row_idx, row in enumerate(test_output):
                    for v in row:
                        ok = symbolic.check_characteristics(str(v))[0]
                        symbolic.add_new_token(str(v))
                        if not ok:
                            break
                    if not ok:
                        break

                    ok = symbolic.check_characteristics("\n")[0]
                    symbolic.add_new_token("\n")
                    if not ok:
                        break
                    
                    if row_idx == len(test_output) - 1:
                        ok = symbolic.check_characteristics("</s>")[0]
                        if not ok:
                            break

                if not ok:
                    histogram[cname]["exceptions"].append(key)

    # Print results
    print("Per-constraint statistics (counts over (task, test_case) pairs):")
    for c_name, stats in histogram.items():
        if c_name == "total":
            continue
        print(
            f"{c_name:55s} -> holds: {stats['holds']:6d}  "
            f"exceptions: {len(stats['exceptions'])}-> {stats['exceptions']}"
        )
    print(f"total:      {histogram['total']}")


if __name__ == "__main__":
    main()