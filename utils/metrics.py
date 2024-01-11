import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import LM_utils
import data_utils
import re

def p_value(betas):
    # calculate p-value for two-sided difference from 0 test with a bootstrapped distribution of statistics, betas
    abs_mean_beta = np.abs(np.mean(betas))
    centered_betas = betas - np.mean(betas)
    outside_prop = np.mean(centered_betas < -abs_mean_beta) + np.mean(centered_betas > abs_mean_beta)  
    return outside_prop

def grid_bootstrap(ndarray, summary_function, boot_times=10000):
    '''
    This function performs bootstrap resampling on an ndarray of dim <=2, applying the summary_function, and returns a mean estimate and 95\% CI for the estimate
    - note we filter our all-nan rows before starting the bootstrap, because those are entirely missing data
    args:
        ndarray: np.ndarray of data of up to ndim=2. Usually two-dimensional data would be of the form n_items x n_models
        summary_function: a function that converts an ndarray to a scalar value (the summary statistic of interest)
        boot_times: number of resamples on the ndarray to perform. The higher, the more precise the bootstrap
    '''
    assert ndarray.ndim <= 2
    # filter out totally missing rows
    n_cols = ndarray.shape[1]
    where_all_nan_rows = np.isnan(ndarray).sum(-1) == n_cols
    ndarray = ndarray[np.argwhere(1-where_all_nan_rows).squeeze()]
    n_rows = ndarray.shape[0]
    # collect stats
    n_observed_rows = []
    n_observations = []
    boot_stats = []
    for _ in range(boot_times):
        # pick number of columns to sample
        col_idx = np.random.choice(np.arange(n_cols), size=n_cols, replace=True)
        row_idx = np.random.choice(np.arange(n_rows), size=n_rows, replace=True)
        resampled_data = ndarray[:,col_idx]
        resampled_data = resampled_data[row_idx, :]
        n_observed_rows.append(len(resampled_data) - (np.isnan(resampled_data).sum(1) == n_cols).sum())
        n_observations.append((1-np.isnan(resampled_data)).sum())
        boot_stat = summary_function(resampled_data)
        boot_stats.append(boot_stat)
    # get mean and 95% quantiles on boot distribution
    mean_estimate = np.mean(boot_stats)
    quantiles = np.quantile(boot_stats, [0.025, .975])
    avg_diff_from_mean = np.mean(np.abs(quantiles - mean_estimate))
    if np.abs(summary_function(ndarray) - mean_estimate) > .01:
        print(f"WARNING: Bootstrap mean estimate error greater than .01, please use more boot_times")
    return_dict = {
        'mean': mean_estimate,
        'error_bar': avg_diff_from_mean,
        'str_format': f"{100*mean_estimate:5.2f} \u00b1 {100*avg_diff_from_mean:5.2f}",
        'p_value': p_value(boot_stats),
        'sample_size': n_rows,
        'effective_sample_size': f"{np.mean(n_observed_rows):.2f}",
    }
    return return_dict

def force_not_dimensionless(data):
    if type(data) is torch.Tensor:
        if data.dim()==0:
            data = data.view(1)
    return data

def safe_seq(seq):
    # filter to non -100 values in seq, which is a list. -100 is the default ignore_index in pytorch
    return [x for x in seq if x >= 0]

def em_accuracy_sum(preds, labels, return_where_correct=False):
    assert len(preds) == len(labels)
    # strict calculation of accuracy for predictions from fewshot model
    preds = np.array([x for x in preds])
    labels = np.array([label for label in labels])
    correct = (preds==labels)
    if return_where_correct:
        return correct.sum(), correct
    else:
        return correct.sum()

def standardize_preds_or_labels(data, tokenizer=None):
    """
    takes tensors, arrays, and lists, and returns standardized pred/label strs
    IF there are multiple labels per item, then we return a list of lists
    ELSE, we return an np array
    args:
        data: should be 1-d np.array, 1-d torch.tensor, or list of these things
        tokenizer: model tokenizer
    """
    # unravel data into list or list of lists
    if type(data) is list and type(data[0]) is torch.Tensor or type(data[0]) is np.ndarray:
        data = [item.tolist() for item in data]
    if type(data) is not list:
        data = data.tolist()
    if type(data) in [int, torch.int, str, np.str_]:
        data = [data]
    # decode if elements are not already strings, or lists of strings (which would suggest it had been decoded already)
    need_to_decode = not (type(data[0]) is str or type(data[0]) is np.str_ or (type(data) is list and type(data[0][0]) is str))
    if need_to_decode:
        data = [tokenizer.decode(safe_seq(seq), skip_special_tokens=True) for seq in data]
    # lowercase and strip the strs
    multiple_eligible_labels = type(data[0]) is list
    if multiple_eligible_labels:
        data = [[x.lower().strip().strip('.') for x in eligible_labels] for eligible_labels in data]
    else:
        data = [x.lower().strip().strip('.') for x in data]
    # convert to np array or list of lists
    if type(data) is torch.Tensor:
        data = data.detach().cpu().numpy()
    elif type(data) is list and type(data[0]) is list:
        data = data # skip the array formatting here as it will not be used in downstream metrics
    else:
        data = np.array(data)
    return data

def first_appearance_fewshot_accuracy_sum(preds, labels, extract_answers, trigger_phrase=None, return_vec=False):
    """
    calculated accuracy of model generations against labels, optionally given answer_choices and a 'trigger phrase' used in CoT
    an answer is 'predicted' if it appears in the pred str
    - this is VERY GENEROUS scoring for some tasks. Use generative_exact_match_accuracy for math tasks
    - this function also faces issues when labels/answers are subsets of one another
    - if multiple answers are mentioned, count which answer appears most. tie breaking is done randomly
    returns acc sum, optionally the vector of binary 0/1 accs per points
    args:
        preds and labels should be list, 1-d np.array, or 1-d torch.tensor of ints or strs
        answer_choices: optional list of answer choices to count
        trigger_phrase: a phrase that could separate e.g. reasoning from a final answer, like "Therefore, the answer is"
    """
    assert len(preds) == len(labels)
    preds = standardize_preds_or_labels(preds)
    labels = standardize_preds_or_labels(labels)
    extract_answers = standardize_preds_or_labels(extract_answers)
    if trigger_phrase is not None:
        trigger_phrase = standardize_preds_or_labels([trigger_phrase]).item()
    n_correct = 0
    use_preds = []
    correct_indicators = []
    for pred, label in zip(preds, labels):
        answer_positions = {answer : 2e8 for answer in extract_answers}
        # extract part of pred after trigger phrase
        if trigger_phrase is not None and trigger_phrase in pred:
            pred = pred.split(trigger_phrase)[1]
        else:
            pred = pred
        # take first appearance of an answer in the pred
        # note this faces difficulty when answers are subsets of one another
        for answer in extract_answers:
            if answer in pred:
                answer_positions[answer] = pred.index(answer)
        min_position = min(answer_positions.values())
        earliest_pred = list(filter(lambda tup: tup[1] == min_position, list(answer_positions.items())))
        if len(earliest_pred) == 1:
            use_pred = earliest_pred[0][0]
        else:
            use_pred = 'NA'
        correct = (use_pred == label)
        n_correct += correct
        correct_indicators.append(correct)
        use_preds.append(use_pred)
    if not return_vec:
        return n_correct
    else:
        return n_correct, use_preds, np.array(correct_indicators)

def extract_numeric(pred):
    # extracts the numeric prediction from a string (should already have string.split(trigger_phrase)[1] applied as necessary)
    pred_end = " ".join(pred.split()[-2:]) # sometimes what follows the top string is something like "2 + 3 = 5" or "5 apples"
    numeric = re.sub(r"[^0-9.]", "", pred_end)
    if numeric == "" or numeric == ".":
        for word in reversed(pred.split(" ")):
            if bool(re.search(r"\d", word)):
                numeric = re.sub(r"[^0-9.]", "", word)
    pred = numeric
    return numeric
    
def generative_exact_match_accuracy_sum(preds, labels, trigger_phrase=None, stop_string=None, numeric_filter=False, return_vec=False):
    """
    Checks that predictions exactly match label during CoT. 
    Preds are extracted by taking text after the "trigger phrase" that always prefaces answers
    args:
        preds and labels should be list, 1-d np.array, or 1-d torch.tensor of ints or strs
        trigger_phrase: a phrase that must always separate e.g. reasoning from a final answer, like "the answer is". Used in CoT
        stop_string: answers always 'end' at this string, like a line break or eos token
    """
    assert len(preds) == len(labels)
    preds = standardize_preds_or_labels(preds)
    labels = standardize_preds_or_labels(labels)
    processed_preds = []
    for pred in preds:
        # break up preds based on trigger phrase
        if trigger_phrase is None:
            pass
        elif trigger_phrase is not None:
            if trigger_phrase in pred:
                pred = pred.split(trigger_phrase)[1]
            else:
                pred = f"[PRED DID NOT HAVE TRIGGER PHRASE]: ...{pred[-30:]}"
        # stop preds at the stop_string
        if stop_string is not None and stop_string in pred:
            pred = pred[:pred.index(stop_string)]
        pred = pred.strip().strip('\n').strip().strip('.')
        if numeric_filter and not "PRED DID NOT HAVE TRIGGER PHRASE" in pred:
            pred = extract_numeric(pred)
        processed_preds.append(pred)
    # compute EM
    n_correct, binary_correct = em_accuracy_sum(processed_preds, labels, return_where_correct=return_vec)
    if not return_vec:
        return n_correct
    else:
        return n_correct, processed_preds, binary_correct
