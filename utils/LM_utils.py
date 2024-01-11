import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import transformers
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import utils
import metrics
from itertools import chain
from copy import deepcopy
import re

def str_clean(data):
    if data is not None:
        return data.strip().lower()
    else:
        return None
    
def renormalize_mc_pred_probs(pred_probs, use_softmax=False):
    # assumes pred_probs of shape num_items x num_answers
    # make sure no elements are negative for sum calculation (e.g., need to shift log_probs)
    if use_softmax:
        pred_probs = torch.softmax(pred_probs, dim=-1)
    else:
        sums = pred_probs.sum(1, keepdim=True)
        pred_probs = pred_probs / sums
    return pred_probs
    
def compute_mc_loss(pred_probs, answer_idx):
    # assumes pred_probs of shape num_items x num_answers
    # assumes answer_idx of shape [num_items], containing idx of answer in range(0,m) assuming m answer choices
    label_probs = torch.gather(pred_probs, 1, answer_idx.view(-1, 1)).squeeze(-1)
    nll = -torch.log(label_probs).mean()
    return nll, label_probs

def compute_probs_from_batch(model, batch, return_value='log_probs', pad_token_id=None, 
                             return_hidden_states=False, num_answers_list=None):
    '''
    Compute label probabilities for decoder-only model, where labels are shifted by one from input ids
    Always returns one value per sequence
    - the reason that we get and write hidden states to file through this function is to do a sanity check that zero-shot accuracy is similar to later experiments
    '''
    assert return_value in ['probs', 'log_probs', 'log_probs_token_normed', 'log_probs_char_normed'] 
    model_batch = {
        'input_ids' : batch['input_ids'],
        'attention_mask' : batch['attention_mask']
    }
    target_tokens = batch['labels']
    if 'targets_mask' in batch and batch['targets_mask'] is not None:
        target_mask = batch['targets_mask']
    else:
        target_mask = target_tokens != pad_token_id
    outputs = model(**model_batch, output_hidden_states=return_hidden_states)
    logits = outputs.logits
    loss_fct = CrossEntropyLoss(reduction='none')
    shift_logits = logits[..., :-1, :]
    shift_labels = target_tokens[..., 1:]
    shift_mask = target_mask[...,1:]
    nll = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    nll = nll.reshape(logits.size(0), -1) # batch size x num tokens
    if return_value == 'log_probs':
        nll = (shift_mask * nll).sum(-1) # sum over token dimension
        probs = -nll # one value per sequence
    elif return_value == 'probs':
        nll = (shift_mask * nll).sum(-1) # sum over token dimension
        probs = torch.exp(-nll) # one probability per input sequence
    elif return_value == 'log_probs_token_normed':
        nll = (shift_mask * nll).sum(-1) # sum over token dimension
        seq_lens = shift_mask.sum(-1) # for shifted targets
        probs = -nll / seq_lens
    elif return_value == 'log_probs_char_normed':
        nll = (shift_mask * nll).sum(-1) # sum over token dimension
        seq_num_chars = [len(answer) for answer in batch['answer_choices']]
        seq_num_chars = torch.tensor(seq_num_chars).to(nll.device)
        probs = -nll / seq_num_chars
    # reshape probs to be num_items x num_answers
    if num_answers_list is not None:
        assert not all(x==1 for x in num_answers_list), "Only single choice items passed to compute_probs as num_answers_list"
        num_items = len(num_answers_list)
        all_same_num_answers = all([num_answers_list[i] == num_answers_list[0] for i in range(num_items)])
        if all_same_num_answers:
            num_answers = num_answers_list[0]
            num_items = batch['input_ids'].size(0) // num_answers
            probs = probs.reshape(num_items, num_answers)
            probs = renormalize_mc_pred_probs(probs, use_softmax=(return_value!='probs'))
        else:
            max_num_answers = max(num_answers_list) # need to pad probs with fewer than max_num_answers
            split_probs = torch.split(probs, num_answers_list)
            all_probs = []
            for item_probs in split_probs:
                item_probs = item_probs.view(1, -1)
                _probs = renormalize_mc_pred_probs(item_probs, use_softmax=(return_value!='probs'))
                padding = torch.zeros((1,max_num_answers-item_probs.shape[1])).cuda()
                _probs = torch.concatenate([_probs, padding], dim=-1)
                all_probs.append(_probs)
            probs = torch.concatenate(all_probs)
    else:
        probs = probs.reshape(len(probs), 1)
    if return_hidden_states:
        hidden_states_dict = {states_name: getattr(outputs, states_name) for states_name in ['decoder_hidden_states', 'encoder_hidden_states', 'hidden_states'] if hasattr(outputs, states_name)}
        return probs, hidden_states_dict
    else:
        return probs

def get_hidden_states_from_batch(model, batch):
    model_batch = {
        'input_ids' : batch['input_ids'],
        'attention_mask' : batch['attention_mask']
    }
    outputs = model(**model_batch, output_hidden_states=True)
    hidden_states_dict = {states_name: getattr(outputs, states_name) for states_name in ['decoder_hidden_states', 'encoder_hidden_states', 'hidden_states'] if hasattr(outputs, states_name)}
    return hidden_states_dict
    
def get_last_token_hidden_states(hidden_states_dict, num_answers=1, num_answers_list=None, max_num_answers=None):
    '''
    This function gathers hidden states from model output, reshapes based on number answers, and stacks/concats into a single array to return
    Items could have different numbers of answers, so that is handled with num_answers_list with padding up to max_num_answers

    args:
        hidden_states_dict: output k,v pairs from model forward pass when 'hidden_states' in k
        num_answers: used to reshape, assuming that the model forward pass was 'flattened' but originally contained aa
    returns
        new_hidden_states: np ndarray of shape: bs x num_answers x enc/dec x num_layers x hidden_size
    '''
    # add num_answers dimension and stack layers
    # new shape is bs x num_answers x seq_len x num_layers x hidden_size
    if num_answers_list is None: # this way is never used in our code, because we always pass num_answers_list for generality
        for k,v in hidden_states_dict.items(): # iterate across decoder/encoder hidden states
            v = torch.stack(v, dim=-2) # stack layers of hidden states in second to last dimension
            hidden_shape = list(v.shape) # bs x seq_len x num_layers x hidden_size
            hidden_shape[0] = hidden_shape[0] // num_answers # cut batch size by num_answers
            hidden_shape.insert(1, num_answers) # insert num_answers after bs (really, now num_items rather than orig bs)
            v = v.reshape(*hidden_shape)
            hidden_states_dict[k] = v.cpu()
    else:
        for k,v in hidden_states_dict.items(): # iterate across decoder/encoder hidden states
            v = torch.stack(v, dim=-2) # stack layers of hidden states in second to last dimension
            padded_hidden_states = []
            per_item_hidden_states = torch.split(v, num_answers_list)
            for item_hidden_states in per_item_hidden_states:
                num_answers = len(item_hidden_states)
                hidden_shape = list(item_hidden_states.shape) #  item_num_answers x seq_len x num_layers x hidden_size
                hidden_shape.insert(0, 1) # insert bs dim of 1
                item_hidden_states = item_hidden_states.view(*hidden_shape)
                # need to pad v with zeros up to max_num_answers
                zeros_shape = deepcopy(hidden_shape)
                zeros_shape[1] = max_num_answers - num_answers
                zeros = torch.zeros(*zeros_shape)
                item_hidden_states = torch.concatenate((item_hidden_states.cpu(), zeros), dim=1)
                padded_hidden_states.append(item_hidden_states)
            hidden_states_dict[k] = torch.concatenate(padded_hidden_states)
    # stack enc/dec and grab last token index
    # new shape is bs x num_answers x enc/dec x num_layers x hidden_size
    if 'decoder_hidden_states' in hidden_states_dict:
        new_hidden_states = hidden_states_dict['decoder_hidden_states'][:, :, -1, :, :].unsqueeze(2).numpy()
    elif 'hidden_states' in hidden_states_dict:
        new_hidden_states = hidden_states_dict['hidden_states'][:, :, -1, :, :].unsqueeze(2).numpy()
    # stack hidden states if there is an encoder. ENCODER HIDDEN STATES WILL BE SECOND INDEX. DECODER STATES ARE ALWAYS FIRST INDEX
    if 'encoder_hidden_states' in hidden_states_dict:
        stack_hidden_states = hidden_states_dict['encoder_hidden_states'][:, :, -1, :, :].unsqueeze(2).numpy()
        new_hidden_states = np.concatenate([stack_hidden_states, new_hidden_states], axis=2) 
    # select only middle and last layer
    num_layers = new_hidden_states.shape[-2]
    middle_layer = np.ceil(num_layers/2) # the embedding layer gets counted as a layer, so round up for odd num_layers
    middle_and_last_idx = torch.tensor([middle_layer, num_layers-1]).to(torch.int)
    new_hidden_states = new_hidden_states[:, :, :, middle_and_last_idx, :]
    return new_hidden_states

def make_LM_batch(tokenizer, prompts, label_strs, label_idx=None, padding_side='left', add_eos_token=False,
                  max_len=None, generative_batch=False, reasoning_chains=None):
    '''
    This makes inputs for computing LM probabilities of labels given prompts, when generative_batch=False
    e.g. for prompts = ["I like", "I like", "I do not like", "I do not like"] and answer_choices = ["dogs", "cats, "birds", "fish]
        with labels = [1,1] (repeating of prompts and flattening of nested answer choice list expected to be done prior to this method)
    This returns a dict
    {
        "input_ids": tokenized ["I like dogs", "I like cats", "I do not like birds", "I do not like fish"]
        "attention_mask": normal attention mask for the tokenizer
        "targets_mask": tensor, 0 where a token belonged in the prompt, 1 where it belonged in answer_choice, 0 for padding
        "label_idx": indices of answers without modification, i.e. [1,1], that can index the probabilities after reshaping into orig_batch_size x num_answers
    }
    intended for use with compute_probs_from_batch
    args:
        generative_batch: when true, input_ids does not contain both prompts and answer_choices
        reasoning_chains: when provided, these are included with labels as "targets" for the batch (for finetuning a model to do CoT)
    '''
    # set pad and bos tokens as needed
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id # note we never supervise model with eos token
    if tokenizer.bos_token_id is None:
        bos_token_list = []
    else:
        bos_token_list = [tokenizer.bos_token_id]
    # tokenize inputs. DO NOT ADD SPACE BEFORE ANSWER. this is handled in prompt.py with spacing at end of prompt
    prompt_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    label_ids = [tokenizer.encode(f"{answer}", add_special_tokens=False) for answer in label_strs]
    if reasoning_chains is not None:
        reasoning_ids = [tokenizer.encode(reasoning, add_special_tokens=False) for reasoning in reasoning_chains]
    if generative_batch:
        lm_inputs = [bos_token_list + _prompt_ids for _prompt_ids in prompt_ids]
    else:
        if reasoning_chains is None:
            lm_inputs = [bos_token_list + _prompt_ids + _label_ids for _prompt_ids, _label_ids in zip(prompt_ids, label_ids)]
        else:
            lm_inputs = [bos_token_list + _prompt_ids + _reasoning_ids + _label_ids for _prompt_ids, _reasoning_ids, _label_ids in zip(prompt_ids, reasoning_ids, label_ids)]        
    # add eos tokens if requested
    if add_eos_token:
        lm_inputs = [x + [tokenizer.eos_token_id] for x in lm_inputs]
    # pad lm inputs
    if max_len is not None and max_len > 0:
        assert not max([len(input_ids) for input_ids in lm_inputs]) > max_len, f"Trying to make LM batch with inputs that are too long for max len {max_len}. Fix this in data_utils.py"
    use_max_len = max([len(input_ids) for input_ids in lm_inputs])
    # left-pad inputs to max len of batch
    for lm_input in lm_inputs:
        short_by = use_max_len - len(lm_input)
        if padding_side == 'left':
            lm_input[:0] = [tokenizer.pad_token_id]*short_by # somehow this is proper indexing...
        else:
            lm_input += [tokenizer.pad_token_id]*short_by 
    # now get label masks
    if generative_batch:
        targets_mask = None
    else:
        targets_mask = []
        reasoning_ids = [[] for _ in range(len(prompt_ids))] if reasoning_chains is None else reasoning_ids
        for _prompt_ids, _reasoning_ids, _label_ids in zip(prompt_ids, reasoning_ids, label_ids):
            num_tokens = len(_prompt_ids) + len(_reasoning_ids) + len(_label_ids) + add_eos_token + (tokenizer.bos_token_id is not None)
            num_target_tokens = len(_reasoning_ids) + len(_label_ids) + add_eos_token
            if padding_side == 'left':
                label_mask = [0]*(use_max_len-num_target_tokens) + [1]*(num_target_tokens)
            elif padding_side == 'right':
                label_mask = [0]*(num_tokens-num_target_tokens) + [1]*num_target_tokens + [0]*(use_max_len-num_tokens)
            targets_mask.append(label_mask)
        targets_mask = torch.tensor(targets_mask)
    # and an attention mask
    lm_inputs = torch.tensor(lm_inputs)
    attention_mask = lm_inputs != tokenizer.pad_token_id
    batch = {
        'input_ids': lm_inputs,
        'attention_mask': attention_mask,
        'targets_mask': targets_mask,
        'label_idx': torch.tensor(label_idx) if label_idx else None,
    }
    return batch

def postprocess_generations(tokenizer, preds, prompts):
    """
    model generations include the prompts by default. this removes these from the generation
    also checks for bad degenerations of alternating stop tokens and real tokens
    """
    if type(preds) is torch.Tensor:
        preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in preds]
    if type(prompts) is torch.Tensor:
        prompts = [tokenizer.decode(x, skip_special_tokens=True) for x in prompts]
    assert len(preds) == len(prompts)
    preds = [pred.replace(prompt, "") for pred, prompt in zip(preds, prompts)]
    return preds

def pull_prompt_from_data(data, k):
  prompt_idx = np.random.choice(np.arange(len(data)), size=k, replace=False)
  prompt_ex = data.iloc[prompt_idx]
  remaining_idx = np.setdiff1d(np.arange(len(data)), prompt_idx)
  remaining_data = data.iloc[remaining_idx]
  return prompt_ex, remaining_data

def score_seq_probs_from_strings(model, tokenizer, strings, breaking_batch_size=None):
    '''
    Helper for scoring straight from a set of strings. computes a string log prob, starting with a bos token
    '''
    all_probs = []
    if breaking_batch_size is not None:
        list_of_strings = np.array_split(strings, len(strings) // breaking_batch_size + 1)
    else:
        list_of_strings = [strings]
    print()
    for idx, _strings in enumerate(list_of_strings):
        empty_prompts = [""] * len(_strings)
        batch = make_LM_batch(tokenizer, prompts=empty_prompts, label_strs=_strings)
        model_input = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'targets_mask': batch['targets_mask'],
            'labels': batch['input_ids'],
        }
        probs = compute_probs_from_batch(model, model_input)
        all_probs.extend(probs.reshape(-1).tolist())
        del probs
        print(f" Batch: {idx}/{len(list_of_strings)} | mem use: {utils.get_mem()}", end='\r')
    return all_probs



