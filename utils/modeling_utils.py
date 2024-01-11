import torch
import numpy as np
import pandas as pd
from transformers import get_scheduler
from torch.optim import SGD, AdamW
import time

from models.probes import Probe
from utils import utils, metrics, LM_utils, data_utils
import globals

def load_optimizer_and_scheduler(args, model, num_training_steps):
    named_parameters = model.named_parameters()
    optimizer_grouped_parameters = [
        {"params": [p for n, p in named_parameters if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": args.weight_decay,
            'lr' : args.lr},
        {"params": [p for n, p in named_parameters if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": 0.0,
            'lr' : args.lr}
    ]
    optimizer_to_class = {'adamw' : AdamW, 'sgd' : SGD}
    optimizer_class = optimizer_to_class[args.optimizer]
    optimizer = optimizer_class(optimizer_grouped_parameters)
    if args.lr_decay == "constant":
        scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    elif args.lr_decay in ['linear', '10-percent']:
        percent_of_orig_value = .1 if args.lr_decay == '10-percent' else 0
        multiplier = 1 / (1-percent_of_orig_value)
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=multiplier*num_training_steps)
    return (optimizer, scheduler)

def evaluate_model(args, 
                log,
                model, 
                dataloader, 
                tokenizer,
                return_hidden_states=False,
                calibrate_probe=False,
                verbose=False):
    '''
    main train_and_eval function that evaluates models on data from a MCTextDataset
    returns eval_stats, which may include the hidden_states as requested
    '''
    # condition names
    gathering_question_end_states = return_hidden_states and args.probing_token_state == 'question_end_token'
    MC_or_classification = not args.generative_eval and not gathering_question_end_states
    # init stats dicts. epochs_stats will be running statistics, used to compute values for eval_stats
    eval_stats = {
        'n_batches': 0,
        'forward_time_sum' : 0,
        'acc': -1,
        'loss': -1,
        'probe_loss': -1, # use for model selection, may be unsupervised objective
        'modal_label': '',
    }
    epoch_stats = {
        'acc_sum': 0,
        'loss_sum': 0,
        'probe_loss_sum': 0, # used for model selection, may be unsupervised objective
        'n_data_points': 0,
    }
    start_time = time.time()
    model.eval()
    total_batches = len(dataloader)
    # set other generative eval args
    if args.generative_eval and hasattr(dataloader, 'dataset'):
        trigger_phrase = 'the answer is' if args.use_cot else None
        if args.probing_method == 'decoding':
            stop_string = "\n"
        if args.probing_method == 'finetuned':
            stop_string = None # would be tokenizer.eos_token, but tokenizer.decode should stop decoding at the eos_token
    all_pd_index = []
    all_probs = []
    all_preds = []
    all_labels = []
    all_binary_correct = []
    label_confidence = []

    # prepare collection of hidden states
    if return_hidden_states:
        # if doing probing like scoring f(x,a) pairs, store one hidden states per "x a" input. But when doing classifation, we classify f(x)
        answers_dim = 1 if args.probing_token_state == 'question_end_token' else data_utils.get_max_num_answers(dataloader.dataset.dataframe)
        hidden_size = globals.model_to_hidden_size[args.short_model_name]
        num_layers = 2 # including last layer and a middle layer. middle layer is idx 0, last layer is idx 1
        encoder_decoder = 2 if args.model_type == 'encoder_decoder' else 1 # will stack encoder/decoder representations
        running_hidden_states = np.ndarray((0, answers_dim, encoder_decoder, num_layers, hidden_size))
    for batch_num, batch in enumerate(dataloader):
        running_time = (time.time()-start_time)
        est_run_time = (running_time/(batch_num if batch_num > 0 else 1)*total_batches)
        forward_time = (eval_stats['forward_time_sum'] / eval_stats['n_batches'] if batch_num > 0 else 0)
        if verbose:
            gpu_mem = utils.get_gpu_utilization() if "cuda" in str(args.device) else None
            log.print_training_prog(eval_stats, 'EVAL', 'EVAL', batch_num, total_batches, running_time, est_run_time, forward_time, gpu_mem=gpu_mem)
        # forward pass
        with torch.no_grad():
            if args.generative_eval:
                _model = getattr(model, 'model', None) if isinstance(model, Probe) else model 
                assert args.padding_side == 'left'
                main_kwargs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                    'do_sample': False,
                    'max_new_tokens': args.max_gen_len,
                    'pad_token_id': tokenizer.pad_token_id,
                    'eos_token_id': tokenizer.eos_token_id
                }
                utils.move_kwargs_to_gpu(main_kwargs)
                forward_begin = time.time()
                preds = _model.generate(**main_kwargs)
                eval_stats['forward_time_sum'] += (time.time() - forward_begin)
                preds = LM_utils.postprocess_generations(tokenizer, preds, main_kwargs['input_ids'])
                labels = batch['label_strs']
                n_correct, trimmed_preds, binary_correct = metrics.generative_exact_match_accuracy_sum(preds, labels, 
                                                                                                       trigger_phrase=trigger_phrase, 
                                                                                                       stop_string=stop_string, 
                                                                                                       numeric_filter=('gsm8k' in args.dataset),
                                                                                                       return_vec=True)
                all_preds.extend(trimmed_preds)
                all_labels.extend(labels)
                all_pd_index.extend(batch['pd_index'])
                all_binary_correct.extend(binary_correct.tolist())
            # in this condition, we're just gathering hidden states from the ends of the questions
            elif gathering_question_end_states:
                main_kwargs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }
                hidden_states_dict = LM_utils.get_hidden_states_from_batch(model, main_kwargs)
            # in this condition, we evaluate the model in a multiple choice or classification fashion
            elif MC_or_classification:
                if isinstance(model, Probe):
                    forward_begin = time.time()
                    num_answers_list = batch['num_answers_list']
                    compute_mc_probs = not all(x==1 for x in num_answers_list)
                    answer_probs = model(batch, compute_mc_probs=compute_mc_probs) # handle cuda() inside forward pass due to moving different data to gpu
                    eval_stats['forward_time_sum'] += (time.time() - forward_begin)
                    assert not return_hidden_states, "If collecting hidden states, use model as an arg and not Probe class as arg"
                else:
                    forward_begin = time.time()
                    main_kwargs = {
                        'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        'labels': batch['input_ids'],
                        'targets_mask': batch['targets_mask'],
                        'answer_choices': batch['answer_choices'],
                    }
                    if return_hidden_states:
                        answer_probs, hidden_states_dict = LM_utils.compute_probs_from_batch(model, main_kwargs, return_value=args.answer_scoring, num_answers_list=batch['num_answers_list'],
                                                                                            return_hidden_states=True)
                    elif not return_hidden_states:
                        answer_probs = LM_utils.compute_probs_from_batch(model, main_kwargs, return_value=args.answer_scoring, num_answers_list=num_answers_list)
                    eval_stats['forward_time_sum'] += (time.time() - forward_begin)
                # get preds, loss, and acc
                preds = torch.argmax(answer_probs, dim=1)
                label_idx = batch['label_idx']
                if args.n_gpu > 0:
                    label_idx = label_idx.cuda()
                    answer_probs = answer_probs.cuda() # syncs probs with labels in multi-gpu case
                loss, label_probs = LM_utils.compute_mc_loss(answer_probs, label_idx)
                if hasattr(model, 'loss'): # i.e., if model is one of our probes. Useful for getting unsupervised loss
                    probe_loss = model.loss(label_idx, answer_probs)
                else:
                    probe_loss = None
                all_probs.append(answer_probs.detach().cpu())
                preds = preds.cpu().numpy()
                label_idx = label_idx.cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(label_idx.tolist())
                all_pd_index.extend(batch['pd_index'])
                n_correct, binary_correct = metrics.em_accuracy_sum(preds, label_idx, return_where_correct=True)
                label_confidence.extend(label_probs.tolist())
                all_binary_correct.extend(binary_correct.tolist())
                # update epoch stats
                epoch_stats['loss_sum'] += loss.item()
                if probe_loss is not None:
                    epoch_stats['probe_loss_sum'] += probe_loss.item()
                del loss, probe_loss
        # update epoch stats
        if not gathering_question_end_states:
            epoch_stats['acc_sum'] += n_correct
        epoch_stats['n_data_points'] += len(batch['items'])
        # update eval stats
        eval_stats['loss'] = epoch_stats['loss_sum'] / (batch_num+1)
        eval_stats['probe_loss'] = epoch_stats['probe_loss_sum'] / (batch_num+1)
        eval_stats['acc'] = epoch_stats['acc_sum'] / epoch_stats['n_data_points']
        eval_stats['n_batches'] += 1
        # accumulate hidden states
        if return_hidden_states:
            if args.probing_token_state == 'answer_end_token':
                num_answers_list = batch['num_answers_list']
                max_num_answers = data_utils.get_max_num_answers(dataloader.dataset.dataframe)
            else:
                num_answers_list = max_num_answers = None
            new_hidden_states = LM_utils.get_last_token_hidden_states(hidden_states_dict, 
                                                                      num_answers_list=num_answers_list, 
                                                                      max_num_answers=max_num_answers)
            running_hidden_states = np.concatenate([running_hidden_states, new_hidden_states], axis=0)
        # print examples
        if verbose:
            # if args.num_print > 0: 
            #     print_idx = list(range(min(args.num_print, len(batch['items']))))
            if (batch_num == 0 and args.num_print > 0): 
                print_idx = list(range(min(args.num_print, len(batch['items']))))
            else:
                print_idx = []
            if len(print_idx) > 0:
                print("\n" + "-"*20 + f"\nPrinting examples:")
                print(f" Exact Input 0     : {tokenizer.decode(batch['input_ids'][0])}")
                for i in print_idx:
                    prompt = batch['prompts'][i]
                    label = batch['label_strs'][i]
                    answer_choices = ['A', 'B', 'C', 'D'] if args.use_letter_labels else batch['answers_list'][i]
                    print(f" point {i}")
                    print(f" Prompt      : \n{prompt}")
                    if MC_or_classification:
                        probs = [np.round(x.item(), 4) for x in answer_probs[i].cpu()]
                        print(f" Preds       : {[x for x in zip(answer_choices, probs)]}")
                        pred = answer_choices[preds[i]]
                        correct = binary_correct[i]
                    elif args.use_cot:
                        print(f" Full pred   : {preds[i]}")
                        pred = trimmed_preds[i]
                        correct = binary_correct[i]
                    elif args.generative_eval:
                        pred = preds[i]
                        correct = binary_correct[i]
                    elif gathering_question_end_states:
                        pred = "No pred; just gathering hidden states"
                        correct = "N/A" 
                    print(f" Pred        : {pred}")
                    print(f" Label       : {label}")
                    print(f" Correct     : {correct}")
                    if args.dataset == 'gsm8k_main':
                        print(f"steps: {batch['items'][i][1].num_steps}  | {batch['items'][i][1].reasoning}")
                    if args.dataset == 'mmlu_STEM-5' or args.dataset == 'third_grade_to_college':
                        print(f"subject: {batch['items'][i][1].subject}  | {batch['items'][i][1].human_hardness}")
                        if correct:
                            write_to_file = prompt + "\n" +  str([x for x in zip(answer_choices, probs)]) + f"\n {correct}"
                            with open('tmp_example.txt', 'w') as f:
                                f.write(write_to_file)
                    if i != print_idx[-1]:
                        print()
                print("-"*20 + '\n')
        del batch

    # calibrate preds and overwrite values in eval_stats. On future calls of evaluate_model, the forward pass will automatically do this all_probs-model.probs_centroid step
    if calibrate_probe:
        all_probs = torch.concatenate(all_probs)
        model.set_calibration_params(probs=all_probs, verbose=True)
        all_probs = all_probs - model.probs_centroid.cpu()
        all_preds = torch.argmax(all_probs, dim=-1).numpy()
        n_correct, binary_correct = metrics.em_accuracy_sum(all_preds, all_labels, return_where_correct=True)
        eval_stats['acc'] = n_correct / epoch_stats['n_data_points']

    # make item level stats df
    item_level_stats = pd.DataFrame({
        'label_confidence': label_confidence if not args.generative_eval else None,
        'accuracy': 1*np.array(all_binary_correct),
    }, index=all_pd_index)
    eval_stats['item_level_stats'] = item_level_stats

    # add hidden_states, label confidence
    if return_hidden_states:
        eval_stats['hidden_states'] = running_hidden_states.astype(np.float32)
    # add model proportion as 'random' performance
    if total_batches > 0:
        label_props = [(label, np.mean(np.array(all_labels) == label)) for label in set(all_labels)]
        label_props = sorted(label_props, key=lambda x: -x[1])
        eval_stats['modal_label'] = f"{label_props[0][0]}: {label_props[0][1]:.2f}" if len(set(all_labels)) > 1 else "NA"
        if len(set(all_preds)) < 10:
            eval_stats['pred_distr'] = {y: round(np.mean(np.array(all_preds)==y), 2) for y in set(all_preds)}
            eval_stats['label_distr'] = {y: round(np.mean(np.array(all_labels)==y), 2) for y in set(all_labels)}
        else:
            eval_stats['pred_distr'] = {}
            eval_stats['label_distr'] = {}
        if verbose:
            print(" Pred distr: ", eval_stats['pred_distr'])
            print(" Label distr: ", eval_stats['label_distr'])
    return eval_stats