import argparse
from copy import deepcopy
import numpy as np
import torch
import pandas as pd
import time
import os
import sys
import jsonlines
import pickle
import random
import gc
from itertools import chain

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import LlamaTokenizer, LlamaForCausalLM

from utils import utils, metrics, LM_utils, data_utils, modeling_utils, plotting_utils
from utils.utils import str2bool
from utils.training_logger import TrainingLogger
from utils.modeling_utils import evaluate_model
from utils.prompt import Prompt
from models.probes import Probe
import globals

def write_hidden_states(args, datasets, model, tokenizer, prompt, log, overwrite_existing=False, verbose=False):
    '''
    writes hidden representations to file for each dataset, hardness/probing split, and prompts
    - hidden_representations will be in a dict saved with np, one per dataset
    - named based on data_type: expect hardness or probing
    - save representations for each prompt_id, {hardness/probing_split: states} nested dicts
    '''
    save_dir = os.path.join(args.data_dir, 'representations')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for dataname, data_dict in datasets.items(): 
        prompt_ids = prompt.dataname_to_prompt_ids[dataname]
        for prompt_id in prompt_ids:
            prompt_idx_kwargs = prompt.get_prompt_kwargs_from_id(prompt_id, dataname=dataname)
            prompt_template = prompt.get_str_prompt_templates(**prompt_idx_kwargs)
            token_state_name = args.probing_token_state.replace('_token', '')
            save_name = f"{dataname}_{args.short_model_name}_prompt-{prompt_id}_{token_state_name}_representations_sd{args.seed}.pkl"
            save_path = os.path.join(save_dir, save_name)
            if os.path.exists(save_path) and not overwrite_existing:
                continue
            representations_dict = {}
            for data_type, data in data_dict.items():
                if data is None:
                    continue
                assert data_type in ['hardness_data', 'probing_data'], f"Got split_name {data_type}, need it to be hardness_data/probing_data"    
                if verbose:
                    print(f"Writing representations for {dataname:15s} | {data_type:10s} | prompt {str(prompt_id):5s}")
                    print(f" Prompt template: {prompt_template}")
                    print(f" GPU mem: {utils.get_gpu_utilization()}")
                prompt_ex = None
                prepped_data = data_utils.prepare_dataframe_for_dataloader(data, prompt_ex, dataname, tokenizer, prompt, prompt_id, max_seq_len=args.max_seq_len)
                assert not args.generative_eval, "This would mess up the formatting for the dataloader below. Want formatting to depend only on force_generative_batch"
                fixed_hardness_method = "learned" # this doesn't really matter
                dataloader = data_utils.get_dataloader(args, fixed_hardness_method, prepped_data, dataname, data_source, tokenizer, args.eval_batch_size, 
                                                       args.padding_side, args.max_seq_len, 
                                                       letter_labels=args.use_letter_labels, 
                                                       force_generative_batch=(args.probing_token_state=='question_end_token'),
                                                       shuffle=False)
                eval_verbose = len(data) > 200
                # eval_verbose = True
                eval_output = modeling_utils.evaluate_model(args, log, model, dataloader, tokenizer, 
                                                                      return_hidden_states=True, 
                                                                      verbose=eval_verbose)
                representations_dict[data_type] = eval_output['hidden_states']
            # save representations
            with open(save_path, 'wb') as file:
                pickle.dump(representations_dict, file, protocol=4) # protocol 4 for file sizes above 4gb
            if verbose:
                print(f"  Saved hidden states at: {save_path}")
                print(f"  (file is : {os.stat(save_path).st_size / (1024**2):.2f} MB)")

def load_hidden_states(args, datasets, prompt):
    '''
    loads hidden representations written by write_hidden_states
    - return representations in {dataname: prompt_idx: reps} nested dicts
    '''
    save_dir = os.path.join(args.data_dir, 'representations')
    hidden_states_dict = {}
    for dataname, data_dict in datasets.items():
        prompt_ids = prompt.dataname_to_prompt_ids[dataname]
        hidden_states_dict[dataname] = {}
        for prompt_id in prompt_ids:
            token_state_name = args.probing_token_state.replace('_token', '')
            save_name = f"{dataname}_{args.short_model_name}_prompt-{prompt_id}_{token_state_name}_representations_sd{args.seed}.pkl"
            save_path = os.path.join(save_dir, save_name)
            try:
                with open(save_path, 'rb') as file:
                    hidden_states = pickle.load(file)
                hidden_states_dict[dataname][prompt_id] = hidden_states # dict of "encoder/decoder_hidden_states" key to ndarray of shape: num_items x num_answers x num_layers x hidden_dim 
            except:
                print(f"Could not find hidden representations at {save_path}")
    return hidden_states_dict

def write_MDL_scores(args, datasets, data_source, precomputed_hidden_states, probe, prompt,
                          supervision_n, boot_times=10, verbose=False):
    '''
    This function writes MDL-based hardness scores to all of the probing datasets we have loaded, by fitting models to the hardness datasets
    - saves sample efficiency results in outputs/
    - writes hardness_scores directly to probing csvs in args.data_dir
    args:
        supervision_n: list of sample sizes to use as train data. we train on 5, 10, 20, etc... points to get progressively stronger models for estimating hardness
        boot_times: since we fit models to as few as 5-10 points, to get an accuracy for these models, average those model outputs over boot_times resampled sets of 5-10 points
    returns:
        nothing
    '''
    hardness_sample_efficiency_results = []
    MDL_col_name = f"MDL_{args.hardness_method}_{args.short_model_name}"
    # make label_confidences data structure
    dataname_prompt_id_to_label_confidences = {} # will be nested dict of dataname: prompt_id: ndarray with shape num_items x boot_times x len(supervision_n)
    dataname_to_item_accs = {}
    for dataname in datanames:
        dataname_prompt_id_to_label_confidences[dataname] = {}
        data_size = len(datasets[dataname]['probing_data'])
        dataname_to_item_accs[dataname] = np.zeros((data_size, args.num_prompts, boot_times, len(supervision_n)))
        for prompt_id in prompt.dataname_to_prompt_ids[dataname]:
            empty_array = np.zeros((data_size, boot_times, len(supervision_n)))
            dataname_prompt_id_to_label_confidences[dataname][prompt_id] = empty_array
    # begin fitting models
    for boot_idx in range(boot_times):
        # first thing is to resample train/test split
        hardness_datasets = data_utils.split_datasets(args,
                                            datasets,
                                            seed=boot_idx,
                                            data_type='hardness',
                                            precomputed_hidden_states=precomputed_hidden_states)
        if verbose and boot_idx == 0:
            print("\nHardness dataset statistics: ")
            for dataname in hardness_datasets.keys():
                data_dict = hardness_datasets[dataname]['hardness_data']
                print(f"  {dataname:15s}: train: {len(data_dict['train']):5d} | dev: {len(data_dict['dev']):5d}")
            print()
        # then get nested training subsets for model fitting -- these are drawn from the train data, while test sets are always the same
        n_to_hardness_subsets = data_utils.get_sample_efficiency_subsets(supervision_n, hardness_datasets, data_type='hardness', seed=boot_idx)
        # loop over training data size
        for which_n, n_train in enumerate(supervision_n):
            n_subsets = n_to_hardness_subsets[n_train]
            # for each eval task...
            for dataname, data_dict in n_subsets.items():
                prompt_ids = prompt.dataname_to_prompt_ids[dataname]
                # SKIP IF ALREADY WRITTEN AND NOT OVERWRITING
                already_written = MDL_col_name in datasets[dataname]['probing_data'].columns
                if already_written and not args.overwrite_existing_measurements:
                    if boot_idx == 0 and which_n == 0:
                        print(f"\nSkipping MDL estimation because already written for {dataname} ({args.hardness_experiment_name})")
                    continue
                for prompt_id in prompt_ids:
                    # the reason this is done all the way inside this loop is so we've already done subsampling for sample size, and we can still loop over eval prompts
                    prepped_data_dict, prompt_ex = data_utils.prepare_dataset_for_dataloader(args,
                                                                        args.hardness_method,
                                                                        data_dict,
                                                                        dataname,
                                                                        prompt, 
                                                                        eval_prompt_id=prompt_id, 
                                                                        tokenizer=tokenizer, 
                                                                        data_type='hardness',
                                                                        k=n_train if args.hardness_method == 'decoding' else None, 
                                                                        multitask_training=args.hardness_multitask, 
                                                                        multiprompt_training=args.hardness_multiprompt,
                                                                        use_cot=False,
                                                                        all_split_datasets=n_subsets)
                    text_data_train = prepped_data_dict['hardness_data']['train']
                    text_data_dev = prepped_data_dict['hardness_data']['dev']
                    # get representations particular to that prompt
                    if args.hardness_method == 'learned':
                        representations_train = prepped_data_dict['hardness_states']['train']
                        representations_dev = prepped_data_dict['hardness_states']['dev'] 
                    else:
                        representations_train = None
                        representations_dev = None
                    # standardize batch sizes and optimization configs
                    if args.hardness_method == 'learned':
                        num_answers = data_utils.get_max_num_answers(pd.concat([text_data_train, text_data_dev]))
                        if args.hardness_batch_size_all_data:
                            train_batch_size = len(text_data_train) * num_answers
                            eval_batch_size = len(text_data_dev) * num_answers
                        else:
                            train_batch_size, eval_batch_size = args.train_batch_size, args.eval_batch_size
                    elif args.hardness_method == 'finetuned':
                        num_answers = data_utils.get_max_num_answers(pd.concat([text_data_train, text_data_dev]))
                        eval_batch_size = args.eval_batch_size
                        effective_num_answers = num_answers if not args.finetuning_objective == 'seq2seq' else 1
                        standardized_opt_config = utils.standardize_optimization_config(n_train,
                                                                                        effective_num_answers, 
                                                                                        args.train_batch_size,
                                                                                        args.grad_accumulation_factor if args.grad_accumulation_factor != 1 else None,
                                                                                        args.finetuned_epochs,
                                                                                        args.minimum_gradient_updates)
                        train_batch_size = standardized_opt_config['train_batch_size']
                        grad_accumulation_factor = standardized_opt_config['grad_accumulation_factor']
                        finetuning_epochs = standardized_opt_config['probing_epochs']
                    else:
                        train_batch_size, eval_batch_size = args.train_batch_size, args.eval_batch_size
                    train_dataloader = data_utils.get_dataloader(args, args.hardness_method, text_data_train, dataname, data_source, tokenizer, train_batch_size, args.padding_side, args.max_seq_len,
                        shuffle=True, for_training=True, precomputed_hidden_states=representations_train)
                    eval_dataloader = data_utils.get_dataloader(args, args.hardness_method, text_data_dev, dataname, data_source, tokenizer, eval_batch_size, args.padding_side, args.max_seq_len,
                        shuffle=False, precomputed_hidden_states=representations_dev)
                    final_train_size = n_train if args.hardness_method == 'decoding' else len(text_data_train)
                    
                    # fit probe if learning the probe
                    if args.hardness_method in ['learned', 'finetuned']:
                        data_utils.set_for_training(train_dataloader, True)
                        save_name = f"{dataname}_hardness_{args.experiment_name}_prompt{prompt_id}_boot{boot_idx}_n{final_train_size}.pt"
                        save_name = save_name.replace('.pt', '') if args.hardness_method == 'finetuned' else save_name
                        probe_save_path = os.path.join(args.model_dir, save_name)
                        fitting_probe = (args.overwrite_existing_models or not os.path.exists(probe_save_path))
                        if fitting_probe:
                            if args.hardness_method == 'learned':
                                num_fits = 1 if args.probe_loss == 'supervised' else 10
                                probe.repeated_fit(args, log, train_dataloader, args.probing_optimizer, epochs=args.learned_probe_epochs, 
                                                    l2_reg=args.l2_reg, num_fits=num_fits, max_grad_norm=args.max_grad_norm, verbose=True)
                                probe.save_probe(probe_save_path)
                            elif args.hardness_method == 'finetuned':
                                probe.finetune(args, log, train_dataloader, tokenizer, finetuning_epochs, grad_accumulation_factor,
                                               dev_dataloader=eval_dataloader, 
                                               model_selection=args.model_selection, 
                                               eval_every_n_epochs=args.dev_eval_every_epochs, 
                                               verbose=True)
                                if args.optimize_weights == 'LORA':
                                    probe.save_probe(probe_save_path)
                        else:
                            probe.load_probe(probe_save_path)
                            if args.hardness_method == 'learned' and args.normalize_representations:
                                probe.set_normalization_params(train_dataloader)

                    # record hardness model performance on train and eval data
                    data_utils.set_for_training(train_dataloader, False)
                    train_stats = modeling_utils.evaluate_model(args, log, probe, train_dataloader, tokenizer, 
                                                                verbose=False)
                    eval_stats = modeling_utils.evaluate_model(args, log, probe, eval_dataloader, tokenizer,
                        verbose=False)
                    hardness_data_result = {
                        'dataname': dataname,
                        'multitask_train': args.hardness_multitask,
                        'multiprompt_train': args.hardness_multiprompt,
                        'n_train': len(text_data_train),
                        'n_dev': len(text_data_dev),
                        'prompt_id': prompt_id,
                        'boot_idx': boot_idx,
                        'train_acc': train_stats['acc'],
                        'eval_acc': eval_stats['acc'],
                    }
                    hardness_sample_efficiency_results.append(hardness_data_result)
                    
                    # accumulate the label confidence for each PROBING point
                    probing_text_data = datasets[dataname]['probing_data']
                    prompt_ex = None
                    probing_text_data = data_utils.prepare_dataframe_for_dataloader(probing_text_data, prompt_ex, dataname, tokenizer, prompt, prompt_id, max_seq_len=args.max_seq_len)
                    if args.hardness_method == 'learned':
                        probing_hidden_states = precomputed_hidden_states[dataname][prompt_id]['probing_data']
                    else:
                        probing_hidden_states = None
                    probe_dataloader = data_utils.get_dataloader(args, args.hardness_method, probing_text_data, dataname, data_source, tokenizer, eval_batch_size, args.padding_side, args.max_seq_len,
                        shuffle=False, precomputed_hidden_states=probing_hidden_states)
                    probing_stats = modeling_utils.evaluate_model(args, log, probe, probe_dataloader, tokenizer,
                                                                 verbose=args.hardness_method in ['decoding', 'finetuned'])
                    label_confidence = probing_stats['item_level_stats']['label_confidence']
                    if verbose:
                        print(f" Data: {dataname:15s} | boot: {boot_idx:2d} | n: {str(len(text_data_train)):4s} | prompt {str(prompt_id):5s}"
                                f" | train acc: {train_stats['acc']:.2f} | dev acc: {eval_stats['acc']:.2f} | mem: {utils.get_mem()}")
                    # store label confidences for this dataset, prompt id, boot idx, and n_train
                    dataname_prompt_id_to_label_confidences[dataname][prompt_id][:, boot_idx,which_n] = np.array(label_confidence)
                    # reload the LORA weights if doing model finetuning and doing another loop
                    will_run_loop_again = len(datanames) > 1 or boot_times > 1 or len(prompt_ids) > 1 or len(supervision_n) > 1
                    if args.probing_method == 'finetuned' and will_run_loop_again:
                        assert args.optimize_weights == 'LORA'
                        probe.model = utils.PEFT_wrap_model(args, probe.model.base_model.model)
                        del train_dataloader, eval_dataloader, probe_dataloader, prepped_data_dict, train_stats, eval_stats, probing_stats, label_confidence
    
    # gather all hardness sample efficiency results
    if len(hardness_sample_efficiency_results) > 0:
        hardness_sample_efficiency_results = pd.DataFrame.from_records(hardness_sample_efficiency_results)
        hardness_sample_efficiency_results = hardness_sample_efficiency_results.sort_values(
            by=['dataname', 'n_train', 'prompt_id', 'boot_idx'], ascending=[True, True, True, True]
        )
    # now iterate through probing datasets and add hardness scores and plot dataset-specific results
    for dataname, data_dict in datasets.items():
        already_written = MDL_col_name in datasets[dataname]['probing_data'].columns
        if already_written and not args.overwrite_existing_measurements:
            continue
        if verbose:
            print(f"\nHardness results for {dataname}")
        # load probing csv
        prompt_ids = prompt.dataname_to_prompt_ids[dataname]
        probing_data = datasets[dataname]['probing_data']
        # gather label confidences per prompt
        label_confidences_list = []
        for prompt_id in prompt_ids:
            label_confidences = dataname_prompt_id_to_label_confidences[dataname][prompt_id] # nans where not enough hardness training data
            label_confidences_list.append(label_confidences)
        # add scores to probing data, averaged across prompts and boot_straps
        label_confidences = np.stack(label_confidences_list, axis=0) # shape: num_prompts x data_id x n_boots x len(supervision_n)
        average_label_confidence = np.nanmean(label_confidences, axis=(0,2,3))
        hardness_scores = 1 - average_label_confidence
        probing_data[MDL_col_name] = hardness_scores
        # save data 
        save_name = f"{dataname}_probing-data_sd{args.seed}.json"
        save_path = os.path.join(args.data_dir, save_name)
        probing_data.to_json(save_path, orient='records')
        # make plots
        grab_these_rows = hardness_sample_efficiency_results.dataname == dataname
        dataname_results = hardness_sample_efficiency_results.loc[grab_these_rows,:]
        if args.hardness_method in ['learned', 'finetuned']:
            save_prefix = f"{dataname}_hardness_efficiency_{args.hardness_experiment_name}"
            plotting_utils.plot_sample_efficiency(dataname_results, save_prefix, x_var_list=['n_train'], no_multiprompt_plot=True)
            save_prefix = f"{dataname}_hardness_distribution_{args.hardness_experiment_name}"
            plotting_utils.plot_hardness_distribution(hardness_scores, save_prefix)
        # save results-avg
        save_name = f"{dataname}_hardness_results_{args.hardness_experiment_name}.csv"
        hardness_results_save_path = os.path.join(args.output_dir, save_name)
        avg_results = dataname_results.groupby(['dataname', 'n_train'])['eval_acc'].mean().reset_index()
        avg_results.to_csv(hardness_results_save_path, index=False)
        # print some examples -- for last prompt, and then take a look at variance across prompts too
        if verbose:
            curves_avg_across_data = np.nanmean(label_confidences, axis=(0,1))
            curves_avg_across_boot = np.nanmean(label_confidences, axis=(0,2))
            prompt_var = np.nanmean(label_confidences, axis=(2,3)) # keep prompt dimension
            if args.num_print > 0:
                print("Estimated hardness examples: ")
            for i in range(args.num_print):
                print(f" {dataname} | probing point {i} | {probing_data.iloc[i]}")
                print(f"ALC curve avged across across data")
                utils.print_rounded_array(curves_avg_across_data[i])
                print(f"ALC curve avged across bootstrap samples")
                utils.print_rounded_array(curves_avg_across_boot[i])
                print(f"ALC scores across prompts")
                utils.print_rounded_array(prompt_var[:,i])
                print(f"ALC single score")
                print(round(average_label_confidence[i],3))
        print(avg_results)

def write_text_prob_scores(args, datasets, model, tokenizer):
    '''
    Add p(Q), p(A), and p(reasoning) scores to a dataset as a potential explanatory variable for model generalization
    '''
    q_col_name = f"question_prob_{args.short_model_name}"
    a_col_name = f"answer_prob_{args.short_model_name}"
    r_col_name = f"reasoning_prob_{args.short_model_name}"
    local_batch_size = int(args.eval_batch_size/2)
    reasoning_batch_size = int(np.ceil(args.eval_batch_size/4))
    for dataname in datasets.keys():
        probing_data = datasets[dataname]['probing_data']
        if not q_col_name in probing_data.columns or args.overwrite_existing_data:
            print("Computing likelihoods for questions...")
            questions = probing_data.question
            question_log_probs = LM_utils.score_seq_probs_from_strings(model, tokenizer, questions, breaking_batch_size=local_batch_size)
            probing_data[q_col_name] = -np.array(question_log_probs)
        if not a_col_name in probing_data.columns or args.overwrite_existing_data:
            print("Computing likelihoods for answers...")
            answers = probing_data.answer_text
            answer_log_probs = LM_utils.score_seq_probs_from_strings(model, tokenizer, answers, breaking_batch_size=local_batch_size)
            probing_data[a_col_name] = -np.array(answer_log_probs)
        if 'strategy-qa' in dataname or 'gsm8k' in dataname:
            if not r_col_name in probing_data.columns or args.overwrite_existing_data:
                print("Computing likelihoods for reasoning chains...")
                reasoning_chains = probing_data.reasoning
                reasoning_log_probs = LM_utils.score_seq_probs_from_strings(model, tokenizer, reasoning_chains, breaking_batch_size=reasoning_batch_size)
                probing_data[r_col_name] = -np.array(reasoning_log_probs)
        # save data
        save_name = f"{dataname}_probing-data_sd{args.seed}.json"
        save_path = os.path.join(args.data_dir, save_name)
        probing_data.to_json(save_path, orient='records')
    return datasets

def postprocess_hardness_scores(args, datasets, verbose=False):
    '''
    Averages already-written hardness scores across different models, per datapoint.
    Note, this function will try to post-process every possible model-based hardness metric, whether or not we've tried to write it yet
    '''
    # locals
    hardness_metrics = ['model_based_finetuned', 'model_based_learned', 'model_based_decoding', 
                        'question_prob', 'answer_prob', 'reasoning_prob',
                    'model_based_finetuned_avg', 'model_based_learned_avg', 'model_based_decoding_avg', 
                    'question_prob_avg', 'answer_prob_avg', 'reasoning_prob_avg']
    hardness_models = globals.hardness_models
    human_variables = ['human_hardness', 
                           'human_grade', 'human_difficulty', 'human_bloom', # 'human_depth_of_knowledge', 
                           'num_steps', 'question_num_words', 'answer_num_words', 'reasoning_num_words', 'answer_num_chars']
    all_variables = human_variables + hardness_metrics
    all_columns = utils.get_hardness_col_names(args.short_model_name, normed=False)
    could_not_find = []
    # collect each metric, for the requested models, for each dataset
    for dataname in datasets.keys():
        probing_data = datasets[dataname]['probing_data']
        for metric in hardness_metrics:
            averaging = 'avg' in metric
            gather_models = [args.model] if not averaging else hardness_models
            # Calculate average raw hardness and average normed hardness across models
            all_hardness_scores = []
            all_normed_hardness_scores = []
            for gather_model in gather_models:
                hardness_col_name = utils.get_hardness_col_name(metric, gather_model, model_avg=False)
                try:
                    hardness_scores = probing_data[hardness_col_name]
                    all_hardness_scores.append(hardness_scores)
                    # norm and add 0-1 scaled hardness scores, so easier to compare scale across models
                    min_score = hardness_scores.min()
                    hardness_scores = hardness_scores - min_score
                    normed_hardness_scores = hardness_scores / hardness_scores.max()
                    all_normed_hardness_scores.append(normed_hardness_scores)
                except:
                    could_not_find.append(f"{dataname} | {metric} | {short_model_name}")
            # save metric. average across "models", which is gather_models when 'avg' is in the metric name
            if len(all_hardness_scores) > 0:
                hardness_scores = np.stack(all_hardness_scores)
                avg_scores = np.mean(hardness_scores, axis=0)
                normed_hardness_scores = np.stack(all_normed_hardness_scores)
                avg_normed_scores = np.mean(normed_hardness_scores, axis=0)
                # col_name -- including 'avg' tag when metric has 'avg' in it
                save_col_name = utils.get_hardness_col_name(metric, gather_model, model_avg=averaging)
                probing_data[save_col_name] = avg_scores
                normed_col_name = f"{save_col_name}_NORMED"
                probing_data[normed_col_name] = avg_normed_scores
                if verbose:
                    print("Saving hardness score to: ", save_col_name)
                    print("Saving hardness score to: ", normed_col_name)
                    if averaging:
                        addendum = f"(tried to average across {len(hardness_models)} models)" if len(hardness_scores) != len(hardness_models) else ""
                        print(f" Avged across across {len(hardness_scores)} models {addendum}")
        if verbose:
            print("Could not produce these metrics (input columns were missing):")
            for not_found in could_not_find:
                print(f"  {not_found}")
        # add hardness_level category
        for var_name in all_variables:
            col_name = utils.get_hardness_col_name(var_name, args.model, model_avg='avg' in var_name)
            if col_name not in probing_data.columns:
                continue
            use_quantiles = 'model' in var_name or 'prob' in var_name or 'num_words' in var_name or 'num_chars' in var_name
            if use_quantiles:
                easy_max, hard_min = np.quantile(probing_data[col_name], [.3, .7])
            else:
                easy_max, hard_min = globals.data_x_hardness_var_to_cutoffs[dataname][var_name]
            def classify_hardness_level(x):
                if x <= easy_max:
                    return 'easy'
                elif x >= hard_min:
                    return 'hard'
                else:
                    return 'medium'
            probing_data[col_name + '_level'] = probing_data[col_name].apply(classify_hardness_level)
            probing_data[col_name + '_level'] = pd.Categorical(probing_data[col_name + "_level"].values,
                                                               categories=["easy", "medium", "hard"], ordered=True)
        available_columns = list(filter(lambda col: col in all_columns, probing_data.columns))
        hardness_data = probing_data[available_columns]
        # print distributoin info
        if verbose:
            # marginal distributinos
            for col in available_columns:
                hardness_scores = hardness_data[col]
                print(f"\n-- {col} --")
                if 'human' in col or 'num_steps' in col:
                    col_name = col
                    counts_df = probing_data.groupby(col_name)['input_text'].count().reset_index()
                else:
                    easy_max, hard_min = np.quantile(hardness_scores, [.3, .7])
                    col_name = col + '_level'
                    counts_df = probing_data.groupby(col_name)['input_text'].count().reset_index()
                    counts_df = counts_df.iloc[[0,2,1]]
                    print(f"{dataname} hardness cut-offs: easy max score is {easy_max:.3f} | hard min score is {hard_min:.3f}")
                total_count = counts_df['input_text'].sum()
                counts_df['prop'] = (counts_df['input_text'] / total_count).apply(lambda x: round(x, 2))
                print(counts_df)
        # correlation matrix
        col_order = sorted(list(hardness_data.columns))
        hardness_data = hardness_data[col_order]
        corr_matrix = hardness_data.corr(method='spearman').round(3)
        # corr_matrix = corr_matrix.dropna() # need to drop only a single empty row and column
        if verbose:
            print(f"{dataname} | Correlation matrix for hardness scores across models (rank correlation):")
            print(corr_matrix.round(2))
        save_path = os.path.join('result_sheets', f"{args.dataset}_corr_matrix_{args.short_model_name}.csv")
        corr_matrix.to_csv(save_path)
        # plot corr matrix
        save_name = f"{args.dataset}_corr_plot_{args.short_model_name}"
        plotting_utils.plot_corr_matrix(corr_matrix, save_name, title = f"{args.dataset} correlation matrix")
            
        # plot hardness distributions, first a facet wrap plot then individual plots
        plotting_utils.plot_hardness_distributions_facet(hardness_data[available_columns], 
                    plot_name=f"{dataname}_hardness_distribution_facet_{args.short_model_name}",
        )
        for col in available_columns:
            plotting_utils.plot_hardness_distribution(hardness_data[col], name=f"{dataname}_hardness_distribution_{col}")

        # ad-hoc answer_num_chars
        probing_data['answer_num_chars'] = probing_data.answer_text.apply(lambda x : len(x)).copy()
            
        # save data
        save_name = f"{dataname}_probing-data_sd{args.seed}.json"
        save_path = os.path.join(args.data_dir, save_name)
        probing_data.to_json(save_path, orient='records')
        # save a backup
        save_name = f"{dataname}_probing-data_sd{args.seed}_BACKUP.json"
        save_path = os.path.join(args.data_dir, save_name)
        probing_data.to_json(save_path, orient='records')
        
    return datasets

def run_modeling_experiment(args, datasets, data_source, probe, tokenizer, prompt, supervision_n, log, verbose=False):
    '''
    main experimental loop. Evaluates a model on the provided datasets, using training sizes in supervision_n
    Performs args.probing_bootstraps evals
    '''
    print(f"Starting probing experiments with supervision_n: {supervision_n}...")
    boot_times = args.probing_bootstraps
    probing_results = []
    # make item_accuracies dict for bootstrapping and plotting. this maps from dataset and prompt to a n_data x n_boots times matrix
    id_no_boot_to_item_accs = {}
    for dataname in datasets.keys():
        prompt_ids = prompt.dataname_to_prompt_ids[dataname]
        id_no_boot_to_item_accs[dataname] = {}
        for prompt_id in prompt_ids:
            id_no_boot_to_item_accs[dataname][prompt_id] = {}
    # hard-coded setting for 3rd grade ARC to MMLU below
    if args.dataset == 'third_grade_to_college':
        dataname_to_hardness_config_dict = {'mmlu_STEM-5': ('human_hardness', 0, 1), 'ai2_arc': ('human_grade', 3, 8), 'ai2_arc_all': ('human_grade', 3, 8)}
    else:
        dataname_to_hardness_config_dict = None
    # begin fitting models
    for boot_idx in range(boot_times):
        # resample train/test splits
        # really convoluted data sampling right now. I rewrote a clean version for our final experiments, as executed by run_jobs.py. The old version is useful for doing individual easy/hard experiemnts
        if args.test_on == "all" and args.record_results_by_hardness and args.stratify_hardness:
            probing_datasets = data_utils.stratified_test_sampling(
                args,
                datasets,
                seed=boot_idx,
                precomputed_hidden_states=precomputed_hidden_states,
                hardness_var_name=args.hardness_var_name,
                train_on=args.train_on,
                n_train=max(supervision_n),
                n_test=args.n_test,
                min_test_size=50 if args.dataset != 'third_grade_to_college' else 0, # very hard-coded for third_grade_to_college test. Don't need to save any data for testing on easy data here
                human_easy_cutoff=args.human_easy_max,
                human_hard_cutoff=args.human_hard_min,
                dataname_to_hardness_config_dict=dataname_to_hardness_config_dict,
            )
        # old code that is functional but convoluted
        else:
            probing_datasets = data_utils.split_datasets(args,
                                                datasets,
                                                seed=boot_idx,
                                                data_type='probing',
                                                precomputed_hidden_states=precomputed_hidden_states,
                                                stratify_hardness=args.stratify_hardness,
                                                hardness_var_name=args.hardness_var_name,
                                                train_on=args.train_on,
                                                test_on=args.test_on,
                                                human_easy_cutoff=args.human_easy_max,
                                                human_hard_cutoff=args.human_hard_min,
                                                human_hardness_exact=False,
                                                standardize_sample_sizes=args.standardize_sample_sizes,
                                                max_n_train=max(supervision_n),
                                                min_test_size=50 if args.dataset != 'third_grade_to_college' else 0, # very hard-coded for third_grade_to_college test. Don't need to save any data for testing on easy data here
                                                dataname_to_hardness_config_dict=dataname_to_hardness_config_dict,
                                                verbose=(boot_idx==0))
        if verbose and boot_idx == 0:
            print("\nProbing dataset statistics: ")
            for dataname in probing_datasets.keys():
                data_dict = probing_datasets[dataname]['probing_data']
                n_train, n_dev, n_test = len(data_dict['train']), len(data_dict['dev']), len(data_dict['test'])
                print(f"  {dataname:15s}: train: {n_train:5d} | dev: {n_dev:5d} | test: {n_test:5d} | total: {n_train+n_dev+n_test:6d}")
                if args.n_dev >= 0 and args.n_test >= 0:
                    print(f"   Subsetting to {args.n_dev} dev and {args.n_test} test...")
            print()
        # get nested training subsets for model fitting -- these are drawn from the train data, while test sets are always the same
        n_to_hardness_subsets = data_utils.get_sample_efficiency_subsets(supervision_n, probing_datasets, data_type='probing', seed=boot_idx)
        # loop over training data size
        for which_n, n_train in enumerate(supervision_n):
            n_subsets = n_to_hardness_subsets[n_train]
            # loop over eval datasets
            for dataname, data_dict in n_subsets.items():
                # SKIP IF ALREADY WRITTEN AND NOT OVERWRITING
                save_name = f"{dataname}_probing_results_{args.experiment_name}.csv"
                results_save_path = os.path.join(args.output_dir, save_name)
                if not args.overwrite_existing_results and os.path.exists(results_save_path):
                    if boot_idx == 0 and which_n == 0:
                        print("Skipping experiment saved at: ", results_save_path)
                    continue
                prompt_ids = prompt.dataname_to_prompt_ids[dataname]
                # SKIP eval on this dataset if a specific eval dataset is indicated, and this one doesn't match
                if args.force_train_dataname != 'NA':
                    if not dataname in args.force_train_dataname:
                        continue
                # hard-coded setting for 3rd grade ARC to MMLU below:
                _test_dataname = args.force_test_dataname if args.force_test_dataname != 'NA' else dataname
                if args.dataset == 'third_grade_to_college':
                    _hardness_var_name = dataname_to_hardness_config_dict[_test_dataname][0]
                else:
                    _hardness_var_name = args.hardness_var_name
                # loop over prompts
                for prompt_id in prompt_ids:
                    # prepare data dict for dataloaders. this does prompt formatting and handles multitask/multiprompt training
                    # the reason this is done all the way inside this loop is so we've already done subsampling for sample size, and we can still loop over eval prompts
                    prepped_data_dict, prompt_ex = data_utils.prepare_dataset_for_dataloader(args, 
                                                                            args.probing_method,
                                                                            data_dict,
                                                                            dataname,
                                                                            prompt, 
                                                                            eval_prompt_id=prompt_id, 
                                                                            tokenizer=tokenizer,
                                                                            data_type='probing',
                                                                            k=n_train if args.probing_method == 'decoding' else None, 
                                                                            multitask_training=args.probing_multitask, 
                                                                            multiprompt_training=args.probing_multiprompt,
                                                                            force_test_dataname=args.force_test_dataname,
                                                                            use_cot=args.use_cot,
                                                                            all_split_datasets=n_subsets)
                    # get pd df for each task
                    text_data_train = prepped_data_dict['probing_data']['train']
                    text_data_dev = prepped_data_dict['probing_data']['dev']
                    text_data_test = prepped_data_dict['probing_data']['test']
                    # manipulate data sizes before eval
                    if args.probing_method == 'decoding':
                        text_data_train = text_data_train.iloc[:0]
                    if args.n_dev >= 0:
                        text_data_dev = text_data_dev.iloc[:args.n_dev]
                    if args.n_test >= 0:
                        text_data_test = text_data_test.iloc[:args.n_test]
                    # get representations particular to that prompt
                    if args.probing_method == 'learned':
                        representations_train = prepped_data_dict['probing_states']['train']
                        representations_dev = prepped_data_dict['probing_states']['dev']
                        representations_test = prepped_data_dict['probing_states']['test']
                    else:
                        representations_train = None
                        representations_dev = None
                        representations_test = None
                    num_answers = data_utils.get_max_num_answers(pd.concat([text_data_train, text_data_dev, text_data_test]))
                    if args.probing_method == 'learned':
                        if args.probing_batch_size_all_data:
                            train_batch_size = len(text_data_train) * num_answers
                            eval_batch_size = len(text_data_test) * num_answers
                        else:
                            train_batch_size, eval_batch_size = args.train_batch_size, args.eval_batch_size
                    elif args.probing_method == 'finetuned':
                        eval_batch_size = args.eval_batch_size
                        effective_num_answers = num_answers if not args.finetuning_objective == 'seq2seq' else 1
                        standardized_opt_config = utils.standardize_optimization_config(n_train,
                                                                                        effective_num_answers, 
                                                                                        args.train_batch_size,
                                                                                        args.grad_accumulation_factor if args.grad_accumulation_factor != 1 else None,
                                                                                        args.finetuned_epochs,
                                                                                        args.minimum_gradient_updates)
                        train_batch_size = standardized_opt_config['train_batch_size']
                        grad_accumulation_factor = standardized_opt_config['grad_accumulation_factor']
                        finetuning_epochs = standardized_opt_config['probing_epochs']
                    else:
                        train_batch_size, eval_batch_size = args.train_batch_size, args.eval_batch_size
                    train_dataloader = data_utils.get_dataloader(args, args.probing_method, text_data_train, dataname, data_source, tokenizer, train_batch_size, 
                        args.padding_side, args.max_seq_len, letter_labels=args.use_letter_labels, use_cot=args.use_cot, shuffle=True, for_training=True, precomputed_hidden_states=representations_train)
                    dev_dataloader = data_utils.get_dataloader(args, args.probing_method, text_data_dev, _test_dataname, data_source, tokenizer, eval_batch_size, 
                        args.padding_side, args.max_seq_len, letter_labels=args.use_letter_labels, use_cot=args.use_cot, shuffle=False, precomputed_hidden_states=representations_dev)
                    test_dataloader = data_utils.get_dataloader(args, args.probing_method, text_data_test, _test_dataname, data_source, tokenizer, eval_batch_size, 
                        args.padding_side, args.max_seq_len, letter_labels=args.use_letter_labels, use_cot=args.use_cot, shuffle=False, precomputed_hidden_states=representations_test)
                    final_train_size = n_train if args.probing_method == 'decoding' else len(text_data_train)
                    
                    # fit probe if learning the probe
                    if args.probing_method in ['learned', 'finetuned']:
                        data_utils.set_for_training(train_dataloader, True)
                        save_name = f"{dataname}_probing_{args.experiment_name}_prompt{prompt_id}_boot{boot_idx}_n{final_train_size}.pt"
                        save_name = save_name.replace('.pt', '') if args.probing_method == 'finetuned' else save_name
                        probe_save_path = os.path.join(args.model_dir, save_name)
                        fitting_probe = (args.overwrite_existing_models or not os.path.exists(probe_save_path))
                        if fitting_probe:
                            if args.probing_method == 'learned':
                                num_fits = 1 if args.probe_loss == 'supervised' else 10
                                probe.repeated_fit(args, log, train_dataloader, args.probing_optimizer, epochs=args.learned_probe_epochs, 
                                                    l2_reg=args.l2_reg, num_fits=num_fits, max_grad_norm=args.max_grad_norm, verbose=True)
                                probe.save_probe(probe_save_path)
                            elif args.probing_method == 'finetuned':
                                probe.finetune(args, log, train_dataloader, tokenizer, finetuning_epochs, grad_accumulation_factor,
                                               dev_dataloader=dev_dataloader, 
                                               model_selection=args.model_selection, 
                                               eval_every_n_epochs=args.dev_eval_every_epochs, 
                                               verbose=True)
                                if args.optimize_weights == 'LORA':
                                    probe.save_probe(probe_save_path)
                        else:
                            probe.load_probe(probe_save_path)
                            if args.probing_method == 'learned' and args.normalize_representations:
                                probe.set_normalization_params(train_dataloader)
                    
                    # evaluate model
                    eval_train = len(text_data_train) <= 100 or not (args.generative_eval and args.max_gen_len > 10)
                    if eval_train:
                        data_utils.set_for_training(train_dataloader, False)
                        train_stats = modeling_utils.evaluate_model(args, log, probe, train_dataloader, tokenizer,
                                                                        verbose=args.use_cot and args.probing_method == 'finetuned')
                    else:
                        train_stats = {'acc': -1, 'modal_label': 'NA'}
                    dev_stats = modeling_utils.evaluate_model(args, log, probe, dev_dataloader, tokenizer, 
                        verbose=verbose,
                        calibrate_probe=args.calibrate_probe)
                    test_stats = modeling_utils.evaluate_model(args, log, probe, test_dataloader, tokenizer,
                        verbose=verbose)
                    # compute test_acc_train_distr, which is test acc for data of the same hardness distr as training data
                    text_data_test = pd.merge(text_data_test,
                                            test_stats['item_level_stats']['accuracy'],
                                            left_index=True,
                                            right_index=True)
                    hardness_col_level = utils.get_hardness_col_name(_hardness_var_name, args.short_model_name, model_avg='avg' in _hardness_var_name)  + '_level'
                    if args.train_on == 'all':
                        test_acc_train_distr = test_stats['acc']
                    elif hardness_col_level not in text_data_test.columns:
                        test_acc_train_distr = None
                    elif args.train_on == 'easy_and_hard':
                        easy_acc = text_data_test[text_data_test[hardness_col_level] == 'easy']['accuracy'].mean()
                        hard_acc = text_data_test[text_data_test[hardness_col_level] == 'hard']['accuracy'].mean()
                        test_acc_train_distr = np.array([easy_acc, hard_acc]).mean()
                    else:
                        test_acc_train_distr = text_data_test[text_data_test[hardness_col_level] == args.train_on]['accuracy'].mean()
                        
                    # record eval results
                    probing_result = {
                        'dataname': dataname, 
                        'test_dataname': _test_dataname,
                        'train_on': args.train_on,
                        'test_on': args.test_on,
                        'n_train': final_train_size,
                        'n_dev': len(text_data_dev),
                        'n_test': len(text_data_test),
                        'prompt_id': prompt_id,
                        'boot_idx': boot_idx,
                        'train_acc': train_stats['acc'],
                        'dev_acc': dev_stats['acc'],
                        'dev_probe_loss': dev_stats['probe_loss'],
                        'test_acc': test_stats['acc'],
                        'test_acc_train_distr': test_acc_train_distr,
                        'multitask_train': args.probing_multitask,
                        'multiprompt_train': args.probing_multiprompt,
                    }
                    hardness_col_names = utils.get_hardness_col_names(args.short_model_name, normed=False)
                    actual_train_data = prompt_ex if args.probing_method == 'decoding' else text_data_train
                    train_hardness_properties = utils.get_mean_std_metrics_from_df(actual_train_data, hardness_col_names,
                                                                             postfix="_TRAIN")
                    test_hardness_properties = utils.get_mean_std_metrics_from_df(text_data_test, hardness_col_names,
                                                                             postfix="_TEST")
                    probing_result.update(train_hardness_properties)
                    probing_result.update(test_hardness_properties)
                    if 'mmlu' in args.dataset and actual_train_data is not None:
                        topic_proportions = {
                            'math_prop_TRAIN': (actual_train_data['subject'] == 'mathematics').mean(),
                            'physics_prop_TRAIN': (actual_train_data['subject'] == 'physics').mean(),
                            'chem_prop_TRAIN': (actual_train_data['subject'] == 'chemistry').mean(),
                            'bio_prop_TRAIN':(actual_train_data['subject'] == 'biology').mean(),
                            'cs_prop_TRAIN':  (actual_train_data['subject'] == 'computer_science').mean(),
                            'math_prop_TEST': (text_data_test['subject'] == 'mathematics').mean(),
                            'physics_prop_TEST': (text_data_test['subject'] == 'physics').mean(),
                            'chem_prop_TEST': (text_data_test['subject'] == 'chemistry').mean(),
                            'bio_prop_TEST': (text_data_test['subject'] == 'biology').mean(),
                            'cs_prop_TEST': (text_data_test['subject'] == 'computer_science').mean(),
                        }
                        probing_result.update(topic_proportions)
                    probing_results.append(probing_result)

                    # BREAK DOWN TEST RESULTS BY HARDNESS SUBGROUP. this means adding 'extra' probing_results rows 
                    if args.record_results_by_hardness:
                        # save results averaged over boots
                        hardness_col_name = utils.get_hardness_col_name(_hardness_var_name, args.short_model_name, model_avg='avg' in _hardness_var_name)
                        hardness_level_name = hardness_col_name + '_level'
                        hardness_col_names = utils.get_hardness_col_names(args.short_model_name, normed=False)
                        metric_vars = ['accuracy'] + hardness_col_names
                        # get the test_acc_train_distr to add to individual results 
                        if args.train_on != 'all':
                            test_acc_train_distr = text_data_test[text_data_test[hardness_level_name] == args.train_on]['accuracy'].mean()
                        else:
                            test_acc_train_distr = text_data_test['accuracy'].mean()
                        for hardness_level in set(text_data_test[hardness_level_name]):
                            test_hardness_subset = text_data_test[text_data_test[hardness_level_name] == hardness_level]
                            hardness_subset_properties = utils.get_mean_std_metrics_from_df(test_hardness_subset, hardness_col_names,
                                                                             postfix="_TEST")
                            hardness_subset_properties['test_on'] = hardness_level
                            hardness_subset_properties['test_acc'] = test_hardness_subset['accuracy'].mean()
                            hardness_subset_properties['test_acc_train_distr'] = test_acc_train_distr
                            # add train statistic
                            hardness_subset_properties.update(train_hardness_properties)
                            # add all the normal experiment metadata
                            hardness_subset_properties.update({
                                'dataname': dataname,
                                'test_dataname': _test_dataname,
                                'multitask_train': args.probing_multitask,
                                'multiprompt_train': args.probing_multiprompt,
                                'n_train': final_train_size,
                                'n_dev': len(text_data_dev),
                                'n_test': len(test_hardness_subset),
                                'prompt_id': prompt_id,
                                'boot_idx': boot_idx,
                                'train_on': args.train_on,
                                'test_on': hardness_level,
                                'train_acc': train_stats['acc'],
                                'dev_acc': dev_stats['acc'],
                                'dev_probe_loss': dev_stats['probe_loss'],
                            })
                            # add question topics if mmlu
                            if 'mmlu' in args.dataset and actual_train_data is not None:
                                topic_proportions = {
                                    'math_prop_TRAIN': (actual_train_data['subject'] == 'mathematics').mean(),
                                    'physics_prop_TRAIN': (actual_train_data['subject'] == 'physics').mean(),
                                    'chem_prop_TRAIN': (actual_train_data['subject'] == 'chemistry').mean(),
                                    'bio_prop_TRAIN':(actual_train_data['subject'] == 'biology').mean(),
                                    'cs_prop_TRAIN':  (actual_train_data['subject'] == 'computer_science').mean(),
                                    'math_prop_TEST': (test_hardness_subset['subject'] == 'mathematics').mean(),
                                    'physics_prop_TEST': (test_hardness_subset['subject'] == 'physics').mean(),
                                    'chem_prop_TEST': (test_hardness_subset['subject'] == 'chemistry').mean(),
                                    'bio_prop_TEST': (test_hardness_subset['subject'] == 'biology').mean(),
                                    'cs_prop_TEST': (test_hardness_subset['subject'] == 'computer_science').mean(),
                                }
                                hardness_subset_properties.update(topic_proportions)
                            probing_results.append(hardness_subset_properties)

                    # make item level stats
                    test_stats_df = datasets[_test_dataname]['probing_data'].copy() # get the full probing data df
                    item_level_stats = test_stats['item_level_stats'].loc[:,['accuracy']]
                    test_stats_df = pd.merge(test_stats_df, item_level_stats, left_index=True, right_index=True, how='outer')
                    if verbose:
                        if boot_idx == 0 and which_n == 0 and prompt_id == prompt_ids[0]:
                            print(f"Train modal label: {train_stats['modal_label']} | Test modal label: {test_stats['modal_label']}")
                        gpu_mem = utils.get_gpu_utilization() if "cuda" in str(args.device) else None
                        print(f" Data: {dataname:10s} | boot: {boot_idx:2d} | n: {str(final_train_size):4s} | prompt {str(prompt_id):5s}"
                            f" | train acc: {train_stats['acc']:.2f} | dev acc: {dev_stats['acc']:.2f} | test acc: {test_stats['acc']:.2f}"
                            f" | mem: {gpu_mem}")
                    # accumulate item level stats for this model
                    if n_train in id_no_boot_to_item_accs[dataname][prompt_id]:
                        id_no_boot_to_item_accs[dataname][prompt_id][n_train].append(test_stats_df)
                    else:
                        id_no_boot_to_item_accs[dataname][prompt_id][n_train] = [test_stats_df]
                    # reload the base model if doing model finetuning
                    will_run_loop_again = len(datanames) > 1 or boot_times > 1 or len(prompt_ids) > 1 or len(supervision_n) > 1
                    if args.probing_method == 'finetuned' and will_run_loop_again:
                        # don't need to reload the whole thing if doing LORA, just re-wrap the base model
                        if args.optimize_weights == 'LORA':
                            probe.model = utils.PEFT_wrap_model(args, probe.model.base_model.model)
                        else:
                            print("\nReloading model...(NOTE: there is a memory leak here that increases the peak memory usage by about 30%. using args.optimize_weights==LORA strongly recommended when running over multiple training sizes or datasets)")
                            for x in gc.get_referrers(probe.model):
                                del x
                            del probe.model
                            probe.model = utils.load_model(args)
                        del train_dataloader, dev_dataloader, test_dataloader, prepped_data_dict
                        
    # collect results
    probing_results = pd.DataFrame.from_records(probing_results)
    probing_results = probing_results.sort_values(
        by=['dataname', 'test_dataname', 'n_train', 'prompt_id', 'boot_idx'], ascending=[True, True, True, True, True]
    )
    # now iterate through datasets and save dataset-specific results
    for dataname in datasets.keys():
        grab_these_rows = probing_results.dataname == dataname
        dataname_results = probing_results.loc[grab_these_rows,:]
        if len(dataname_results) == 0:
            continue
        # define hardness_col_name and level again
        if _hardness_var_name != 'NA':
            hardness_col_name = utils.get_hardness_col_name(_hardness_var_name, args.model, model_avg='avg' in _hardness_var_name)
            hardness_col_level = hardness_col_name  + '_level'
        else:
            hardness_col_name = None
        # save all results, for each bootstrap and prompt
        save_name = f"{dataname}_probing_results_all-boots_{args.experiment_name}.csv"
        results_save_path = os.path.join(args.output_dir, save_name)
        dataname_results.to_csv(results_save_path, index=False)
        # plot sample_efficiency for each prompt
        save_prefix = f"{dataname}_probing_efficiency_{args.experiment_name}"
        if len(supervision_n) > 1: 
            plotting_utils.plot_sample_efficiency(dataname_results, save_prefix, outcome='test_acc', no_prompt_avg_plot=True)
        # get best prompt id
        max_n_train = dataname_results['n_train'].max()
        where_max_n_train = dataname_results.n_train == max_n_train
        boot_avg_results = dataname_results.loc[where_max_n_train].groupby(['prompt_id'])['dev_probe_loss'].mean().reset_index()
        best_prompt_idx = boot_avg_results['dev_probe_loss'].idxmin()
        best_prompt = boot_avg_results.loc[best_prompt_idx].prompt_id
        # plot acc vs hardness for max_n_train (for best prompt)
        if _hardness_var_name != 'NA' and args.test_on == 'all':
            test_stats_df = utils.gather_item_level_stats_df(id_no_boot_to_item_accs, 
                                                                n_train, 
                                                                dataname, 
                                                                best_prompt, 
                                                                hardness_col_name)
            save_name = f"{dataname}_probing_acc_vs_hardness_{args.experiment_name}"
            plotting_utils.plot_acc_vs_hardness(test_stats_df, save_name, hardness_col_name)
        # save results averaged over boots, with one row per prompt
        hardness_col_names = utils.get_all_possible_hardness_col_names(args.short_model_name)
        grouping_vars = ['dataname', 'test_dataname', 'train_on', 'test_on', 'n_train', 'n_dev', 'prompt_id'] # average over n_test rather than group by it, because different boots might have different amounts of test data per hardness subset
        metric_vars = ['n_test', 'train_acc', 'dev_acc', 'test_acc', 'test_acc_train_distr'] + hardness_col_names
        if 'mmlu' in args.dataset:
            metric_vars += globals.mmlu_subject_stat_cols
        boot_avg_results = utils.average_df_over_metrics(dataname_results, grouping_vars, metric_vars)
        assert len(boot_avg_results) == len(dataname_results) / args.probing_bootstraps, "Having trouble averaging overbootstrapped experiments. Value in grouping_vars must differ between bootstrapped splits"
        save_name = f"{dataname}_probing_results_{args.experiment_name}.csv"
        results_save_path = os.path.join(args.output_dir, save_name)
        boot_avg_results.to_csv(results_save_path, index=False)
        # save a melted/long version of the result if it's a single row
        if len(boot_avg_results) == 1:
            save_name = f"{dataname}_probing_results_LONG_{args.experiment_name}.csv"
            long_results_save_path = os.path.join(args.output_dir, save_name)
            boot_avg_results.T.reset_index().to_csv(long_results_save_path, index=False) 
        
        # get best prompt results and bootstrap CIs
        best_prompt_results = boot_avg_results.loc[boot_avg_results['prompt_id'] == best_prompt]
        # bootstrap the item level accuracy ndarray, saving separately for each n_train and adding CIs to df
        acc_bootstrap_df = []
        all_n_train_item_accs = []
        for n_train in supervision_n:
            test_stats_df = utils.gather_item_level_stats_df(id_no_boot_to_item_accs, 
                                                    n_train, 
                                                    dataname, 
                                                    best_prompt, 
                                                    hardness_col_name)
            hardness_levels = [args.test_on] if not args.record_results_by_hardness else ['all', 'easy', 'medium', 'hard']
            if _hardness_var_name == 'NA':
                hardness_levels = ['all']
            for hardness_level in hardness_levels:
                # perform bootstraps for test subsets (or entire test set)
                if hardness_level != 'all':
                    test_subset = test_stats_df[test_stats_df[hardness_col_level] == hardness_level]
                else:
                    test_subset = test_stats_df
                acc_cols = filter(lambda x: 'acc' in x, test_stats_df.columns)
                acc_ndarray = test_subset.loc[:,acc_cols].to_numpy()
                acc_bootstrap_results = metrics.grid_bootstrap(acc_ndarray, np.nanmean, boot_times=10000)
                acc_bootstrap_results['test_on'] = hardness_level
                acc_bootstrap_results['n_train'] = n_train
                acc_bootstrap_results = pd.DataFrame.from_records([acc_bootstrap_results])
                acc_bootstrap_df.append(acc_bootstrap_results)
                # accumulate the csv with accs and the hardness var name
                if hardness_level == 'all' or len(hardness_levels) == 1:
                    test_subset = test_stats_df.copy()
                    test_subset = test_subset.rename(columns={hardness_col_name: 'hardness_value',
                                                              hardness_col_level: 'hardness_level_value'}).copy()
                    test_subset['model'] = args.model
                    test_subset['probing_method'] = args.probing_method
                    test_subset['n_train'] = n_train
                    test_subset['train_on'] = args.train_on
                    test_subset['test_on'] = args.test_on
                    test_subset['hardness_var_name'] = _hardness_var_name
                    all_n_train_item_accs.append(test_subset)
        # save the csv with accs and the hardness var name
        all_n_train_item_accs = pd.concat(all_n_train_item_accs)
        save_name = f'item_accs_{dataname}_{args.experiment_name}.csv'
        save_path = os.path.join(args.data_dir, save_name)
        all_n_train_item_accs.to_csv(save_path, index=True)
        # merge bootstrap results to the best_prompt_results
        acc_bootstrap_df = pd.concat(acc_bootstrap_df)
        acc_bootstrap_df = acc_bootstrap_df.loc[:,['test_on', 'n_train', 'error_bar', 'str_format', 'sample_size']]
        group_vars = ['test_on', 'n_train']
        # best_prompt_results = pd.merge(best_prompt_results, acc_bootstrap_df, left_on='n_train', right_on='n_train', how='outer')
        best_prompt_results = pd.merge(best_prompt_results, acc_bootstrap_df, left_on=group_vars, right_on=group_vars, how='outer')
        best_prompt_results = best_prompt_results.rename(columns={'str_format': 'test_acc_str'})
        # move CI and effective test sample size columns up in the column ordering
        move_col_idx = list(range(len(best_prompt_results.columns)))[:-2]
        move_col_idx.insert(7, -2)
        move_col_idx.insert(4, -1)
        best_prompt_results = best_prompt_results.iloc[:,move_col_idx]
        # save best prompt results
        best_prompt_results.to_csv(results_save_path.replace('results', 'results-best-prompt'), index=False)
        # plot best prompt learning curve
        if len(supervision_n) > 1:
            save_prefix = f"{dataname}_probing_efficiency_best-prompt_{args.experiment_name}"
            plotting_utils.plot_sample_efficiency(best_prompt_results, save_prefix, outcome='test_acc')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc. & debugging
    parser.add_argument('--gpu', type = int, default = 0, help = 'gpu id to use. set to -1 for no GPU')
    parser.add_argument("--seed", default=0, type=int, 
                        help='intended to determine hardness/probing splits from original datasets ONLY. bootstrapping logic controls training + testing randomness')
    parser.add_argument("--debug", default=False, type=str2bool)
    parser.add_argument("--num_print", '-np', default = 0, type=int, help = 'number of points to print')
    parser.add_argument("--quantization", default = '8bit', choices=['NA', '8bit', '4bit', '16bit'], type=str, help = 'quantize model for inference')
    parser.add_argument("--gradient_checkpointing", default = False, type=str2bool, help = '')
    # model
    parser.add_argument("--model", default='gpt2-medium', type=str, help='name of pretrained model')
    parser.add_argument("--model_type", default='decoder', choices=['decoder', 'encoder_decoder'], help='')
    # paths/directories for data, cache, etc.
    parser.add_argument("--server", default='', type=str, help='')
    parser.add_argument("--data_dir", default='data', type=str, help='')
    parser.add_argument("--model_dir", type=str, help='')
    parser.add_argument("--cache_dir", type=str, help='')
    parser.add_argument("--output_dir", default='outputs', type=str, help='')
    parser.add_argument("--log_dir", default='training_logs', type=str, help='')
    # data selection args
    parser.add_argument("--dataset", default='amazon_polarity', help='')
    parser.add_argument("--force_train_dataname", default = 'NA', type=str, help = 'used for dataset transfer experiemnts. used to only train on this dataname in the run_modeling_experiment loop, when looping over multiple datasets')
    parser.add_argument("--force_test_dataname", default = 'NA', type=str, help = 'used for dataset transfer experiments. When loading multiple datasets, can specify which single dataset to test on')
    # generic prompting, data transformation, and scoring args
    parser.add_argument("--prompt_source", default='custom', choices=['promptsource', 'custom'], help='promptsource only usable for burns datasets')
    parser.add_argument("--specific_prompt", default=None, type=str, help='str id for a specific prompt to use. requires num_prompts==1. see prompts.py')
    parser.add_argument("--num_prompts", default=1, type=int, help='number of (random) prompts to use. set to -1 if we use all prompts available (only reasonable for promptsource)')
    parser.add_argument("--use_letter_labels", default=False, type=str2bool, help='use A/B/C/D labels p(y|x) scoring and as labels to in-context examples (MUST USE answer choices in the prompt -- see prompt.answer_choices_functions)')
    parser.add_argument("--noise_labels_p", default=0, type=float, help='if p>0, noise the labels p percent of the time')
    parser.add_argument("--k_shot", '-k', default=5, type=int, help='')
    parser.add_argument("--think_step_by_step", default=False, type=str2bool, help='')
    parser.add_argument("--padding_side", default='left', choices=['left', 'right'], help='padding side for batching inputs')
    parser.add_argument("--answer_scoring", default='log_probs', choices=['probs', 'log_probs', 'log_probs_token_normed', 'log_probs_char_normed'], 
                        help='the score for each multiple-choice option. Only used for autoregressive scoring')
    # generic probing args
    parser.add_argument("--finetuning_objective", '-fo', default='seq2seq', choices=['seq2seq', 'MC'], help='when doing probing_method=finetuned, either finetune model with seq2seq loss or a proper multiple choice loss')
    parser.add_argument("--optimize_weights", default='LORA', choices=['embeddings', 'all', 'LORA', 'ABCD_embeddings'], help='model weights to finetune. only relevant if probing_method=finetuned')
    parser.add_argument("--generative_eval", default=False, type=str2bool, help='evaluate decoding probe in generative manner')
    parser.add_argument("--max_gen_len", '-mgl', default=1, type=int, help='max generation steps for when generative_eval is true')
    parser.add_argument("--use_cot", default=False, type=str2bool, help='uses reasoning steps for CoT for datasets where reasoning steps are availabe')
    parser.add_argument("--probe_loss", default='supervised', choices=['supervised', 'LM_loss', 'CCS', 'CCS_ours', 'unsupervised', 'mixed-supervision', 'random'], 
                        help='only relevant if probing_method=learned. random just avoids training probe, so the probe is a random projection')
    parser.add_argument("--inference_prompt_strategy", default='single-prompt', choices=['single-prompt', 'ensemble'], 
                        help='either use single prompt for inference, or ensemble prompt preds and take majority vote, or some more complicated inference algorithm, e.g. maiuetic prompting')
    parser.add_argument("--calibrate_probe", default = False, type=str2bool, help = 'whenever calling modeling_utils.evaluate_model, calibrate preds to be close to uniform')
    parser.add_argument("--model_selection", '-ms', default='NA', choices=['train_acc', 'dev_acc', 'NA'], help='pick the best training epoch based on this statistic when doing model finetuning')
    parser.add_argument("--minimum_gradient_updates", "-mgu", default = 10, type=int, help='')
    parser.add_argument("--learned_probe_epochs", "-pe", default = 100, type=int, help='')
    parser.add_argument("--finetuned_epochs", "-fe", default = -1, type=int, 
                            help='Automatically set in utils.standardize_optimization_configs if left at -1')
    # hardness model + training parameters
    parser.add_argument("--hardness_method", '-hm', default='learned', choices=['decoding', 'learned', 'finetuned'], 
                        help='method for computing model-based hardness scores. decoding computes autoregressive probability. learned probe is a parametric probe on hidden states. finetuned is QLoRA')
    parser.add_argument("--hardness_probe_model", default='linear', choices=['linear', 'MLP', 'transformer'], help='only relevant if probing_method=learned')
    parser.add_argument("--hardness_multitask", default = False, type=str2bool, help = 'hardness model is trained multi-task on all passed training datasets')
    parser.add_argument("--hardness_multiprompt", default = False, type=str2bool, help = 'hardness model is trained multi-prompt, evaluated on a single prompt')
    parser.add_argument("--hardness_batch_size_all_data", default = True, type=str2bool, help = 'fit learned hardness model to the entire train dataset, ignoring train_batch_size')
    parser.add_argument("--hardness_optimizer", default='adamw', choices=['sgd', 'adamw', 'LBFGS'])
    parser.add_argument("--hardness_bootstraps", "-hb", default = 1, type=int, help='')
    parser.add_argument("--hardness_lr_decay", default='10-percent', choices=['linear', 'constant', '10-percent'])
    # probe model and probe evaluation arguments
    parser.add_argument("--probing_method", '-pm', default='learned', choices=['decoding', 'learned', 'finetuned'], 
                        help='modeling method for solving the task. decoding computes autoregressive probability. learned probe is a parametric probe on hidden states. finetuned is QLoRA by default')
    parser.add_argument("--probe_model", default='linear', choices=['linear', 'MLP', 'transformer'], help='clasifier head architecture. only relevant if probing_method=learned')
    parser.add_argument("--probing_multitask", default = False, type=str2bool, help = 'probing model is trained multi-task on all passed training datasets')
    parser.add_argument("--probing_multiprompt", default = False, type=str2bool, help = 'probing model is trained multi-prompt, evaluated on a single prompt')
    parser.add_argument("--probing_batch_size_all_data", default = True, type=str2bool, help = 'fit learned supervised probe to the entire train dataset, ignoring train_batch_size')
    parser.add_argument("--probing_lr", default=1e-4, type=float, help='applies only to finetuned models')
    parser.add_argument("--probing_layers", default = 'middle_and_last', type=str, choices=['middle', 'last', 'middle_and_last'])
    parser.add_argument("--probing_lr_decay", default='10-percent', choices=['linear', 'constant', '10-percent'])
    parser.add_argument("--probing_optimizer", default='sgd', choices=['sgd', 'adamw', 'rmsprop', 'adam', 'LBFGS'],
                                            help='ONLY APPLIES TO LINEAR PROBE, not model finetuning. ')
    parser.add_argument("--probing_token_state", '-pts', default='answer_end_token', choices=['question_end_token', 'answer_end_token'],
                                            help='Do probing in a classification f(x) fashion or scoring f(x, a) fashion')
    parser.add_argument("--stratify_hardness", default = True, type=str2bool, help = 'stratifies probing data by hardness variable')
    parser.add_argument("--probing_bootstraps", "-pb", default=1, type=int, help='')
    parser.add_argument("--dev_eval_every_epochs", "-deee", default=-1, type=int, help='')
    # evaluation arguments
    parser.add_argument("--train_on", default = 'all', type=str, help = 'split of data to fit probe to')
    parser.add_argument("--test_on", default = 'all', type=str, help = 'split of data to fit probe to')
    parser.add_argument("--standardize_sample_sizes", '-sss', default = False, type=str2bool, help = '')
    parser.add_argument("--use_extra_easy_data", default = False, type=str2bool, help = '')
    parser.add_argument("--no_dev", default = True, type=str2bool, 
                        help = 'when doing standardize_sample_sizes, this moves dev data to training data')
    parser.add_argument("--all_data_to_test", default = False, type=str2bool, 
                        help = 'when NOT doing standardize_sample_sizes: allocate all data to be test data, except for the maximum requested examples to be used during training')
    parser.add_argument("--probing_learning_curve", '-lc', default = False, type=str2bool, help = '')
    parser.add_argument("--human_easy_max", default = .5, type=float, help = 'used for determining threshold for easy data for ARC/MMLU datasets')
    parser.add_argument("--human_hard_min", default = .5, type=float, help = 'used for determining threshold for hard data for ARC/MMLU datasets')
    parser.add_argument("--hardness_var_name", default='NA',
                        choices=['NA', 
                                 'human_grade', 'human_difficulty', 'human_bloom', 'human_depth_of_knowledge', # arc
                                 'human_hardness', # mmlu
                                 'num_steps', # strategy-qa and gsm8k
                                 'question_num_words', 'answer_num_words', 'reasoning_num_words', 'answer_num_chars', # all
                                 'model_based_finetuned', 'model_based_learned', 'model_based_decoding', 
                                 'question_prob', 'answer_prob', 'reasoning_prob',
                                 'model_based_finetuned_avg', 'model_based_learned_avg', 'model_based_decoding_avg', 
                                 'question_prob_avg', 'answer_prob_avg', 'reasoning_prob_avg',
                                 ], 
                        help='kind of variable to use as hardness measurement. x_avg vars are averaged across several LLMs')
    parser.add_argument("--hardness_model_names", default='self',
                        choices=['self', 'hardness_models'], 
                        help='self: use only architecure of --model arg for model-based hardness variable. globals: average across models in globals.py')
    parser.add_argument("--n_train", default=320, type=int, help='specify number of desired training points for a learned probe')    
    parser.add_argument("--n_dev", '-nd', default=-1, type=int, help='adjust the number of test points -- useful for shorter ICL runs')    
    parser.add_argument("--n_test", '-nt', default=-1, type=int, help='adjust the number of test points -- useful for shorter ICL runs')
    # generic training hyperparams + conditions -- used in various settings
    parser.add_argument("--train_batch_size", '-tbs', default=8, type=int, help='')
    parser.add_argument("--eval_batch_size", '-ebs', default=8, type=int, help='')
    parser.add_argument("--max_seq_len", default=2048, type=int, help='sequences cannot be longer than this -- automatically truncated if they are, see dataset.collate_fn')
    parser.add_argument("--max_grad_norm", default=1., type=float, help='')
    parser.add_argument("--normalize_representations", default=True, type=str2bool, 
                        help='z-normalize hidden states for supervised probe. If the label space is fixed, do this per label. If it is multiple-choice, share information across answers')
    parser.add_argument("--grad_accumulation_factor", '-gaf', default=1, type=int, help='effective batch size = batch_size * grad_accumulation_factor')
    parser.add_argument("--weight_decay", default=1e-5, type=float, help='')
    parser.add_argument("--l2_reg", default=1, type=float, help='L2 reg to apply only with probing_method==learned')
    parser.add_argument("--dropout", default=0, type=float, help='')
    # control flow + experiment conditions
    parser.add_argument("--load_model", '-llm', default = False, type=str2bool, help = 'load LLM')
    parser.add_argument("--write_hidden_states", default = False, type=str2bool, help = 'write model representations to file for tasks and prompts')
    parser.add_argument("--estimate_hardness", default = False, type=str2bool, help = 'write model representations to file for tasks and prompts')
    parser.add_argument("--do_eval", default = False, type=str2bool, help = 'performs a full evaluation of a model')
    parser.add_argument("--record_results_by_hardness", default = False, type=str2bool, 
                        help = 'when test_on=all, this converts a single experiment result row into multiple rows, one for each level of hardness in the data')
    parser.add_argument("--overwrite_existing_results", '-oer', default = True, type=str2bool, help = 'will overwrite existing results files from past experiments')
    parser.add_argument("--overwrite_existing_models", default = True, type=str2bool, help = 'will overwrite existing models from past experiments')
    parser.add_argument("--overwrite_existing_measurements", '-oem', default = True, type=str2bool, help = 'will overwrite hardness variables and postprocessed datasets')
    parser.add_argument("--overwrite_existing_data", '-oed', default = False, type=str2bool, 
                        help = 'overwrite hidden states and x_probs in for questions/answers/probs')
        
    # parse + env variables + check args
    args = parser.parse_args()
    experiment_start = time.time()
    if args.finetuning_objective == 'seq2seq' and 'finetuned' in [args.hardness_method, args.probing_method]:
        assert args.probe_loss == "LM_loss"
    if 'learned' in args.hardness_method and args.estimate_hardness:
        assert args.probe_loss != "LM_loss"
    if 'learned' in args.probing_method and args.do_eval:
        assert args.probe_loss != "LM_loss"
    if args.record_results_by_hardness:
        assert args.test_on == 'all'
        assert args.stratify_hardness
    if args.debug:
        args.n_train = args.n_dev = args.n_test = 10
    if args.no_dev:
        args.n_dev = 0
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)

    # set probing configs
    args.short_model_name = short_model_name = utils.shorten_model_name(args.model)
    if args.probing_layers == 'last':
        probing_layers = [1]
    if args.probing_layers == 'middle':
        probing_layers = [0]
    if args.probing_layers == 'middle_and_last':
        probing_layers = [0,1]
    hardness_probing_config = {
            'probe_model': args.hardness_probe_model,
            'lr_decay': args.hardness_lr_decay,
            'hidden_size': globals.model_to_hidden_size[short_model_name],
            'features_enc_dec': args.model_type.split('_'), 
            'features_layers': probing_layers,
        }
    probing_config = {
        'probe_model': args.probe_model,
        'lr_decay': args.probing_lr_decay,
        'hidden_size': globals.model_to_hidden_size[short_model_name],
        'features_enc_dec': args.model_type.split('_'),
        'features_layers': probing_layers, 
    } 
    # init experiment name, TrainingLogger, stats_dict, and saving/loading paths
    experiment_name = utils.get_experiment_name(args)
    args.experiment_name = experiment_name
    args.hardness_experiment_name = utils.get_hardness_experiment_name(args)
    # args.hardness_col_name = utils.get_hardness_col_name(args.hardness_var_name, args.model, model_avg='avg' in args.hardness_var_name)
    if args.do_eval:
        print(f"Starting experiment: {experiment_name}")
    if args.estimate_hardness:
        print(f"  Hardness exp name: {args.hardness_experiment_name}\n")
    log_file = os.path.join(args.log_dir, f"log_{args.dataset}_{experiment_name}.csv")
    log = TrainingLogger(args, log_file, experiment_name = experiment_name, overwrite_existing=args.overwrite_existing_results)

    # GPU + SEED setup
    n_gpu = torch.cuda.device_count()
    args.n_gpu = n_gpu
    if n_gpu == 1 and args.gpu != -1:
        device = torch.cuda.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    elif n_gpu > 1 and args.gpu != -1:
        device = torch.cuda.device("cuda")
        torch.cuda.set_device(device)
    else:
        print("RUNNING EXPERIMENT ON CPU")
        device = torch.device("cpu")
    torch.manual_seed(args.seed)
    args.device = device
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # download/write data
    print("Loading data...")
    load_start = time.time()
    if args.dataset in globals.known_hardness_data:
        data_utils.write_datasets_with_known_hardness(args,
                        args.dataset,
                        data_dir=args.data_dir,
                        seed=args.seed,
                        make_hardness_split=(args.dataset in globals.still_make_hardness_data), # still going to separate some points for model-based hardness, to compare with human measures
                        overwrite_existing=False, # THIS WILL OVERWRITE SAVED MODEL-BASED HARDNESS SCORES IN PROBING DATA. BE CAREFUL
                        verbose=True)
    elif args.dataset in globals.eligible_datasets:
        probing_sample_size = 1000
        data_utils.write_datasets(args.dataset,
                        sample_size=probing_sample_size,
                        data_dir=args.data_dir,
                        seed=args.seed,
                        min_hardness_points=200,
                        max_hardness_points=1000,
                        overwrite_existing=False, # THIS WILL OVERWRITE SAVED HARDNESS SCORES IN PROBING DATA. BE CAREFUL
                        verbose=True)
    data_source = data_utils.standardize_data_source(args.dataset)
    args.data_source = data_source
    # get list of names of individual datasets
    if hasattr(globals, args.dataset):
        datanames = getattr(globals, args.dataset)
    else:
        datanames = [args.dataset]
    # load hardness and probing datasets
    datasets = data_utils.load_datasets(args,
                  datanames, 
                  data_dir=args.data_dir,
                  seed=args.seed)
    print("Loaded datasets: ", list(datasets.keys()))
    
    # clean datanames and model name to remove slash + prefix for proper file paths later
    datanames = [x.split('/')[-1] for x in datanames]
    args.short_model_name = short_model_name
    
    # load prompt object
    prompt = Prompt(args, datanames, data_source=data_source, prompt_source=args.prompt_source, 
                    use_cot=args.use_cot, use_letter_labels=args.use_letter_labels, 
                    seed=args.seed, num_prompts=args.num_prompts)
    
    # load tokenizer and add any missing eos/pad token ids
    if 'Llama-2' in args.model:
        size = utils.get_model_size(args.model)
        llama2_path = f"/net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/{size.upper()}"
        tokenizer = LlamaTokenizer.from_pretrained(llama2_path, legacy=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, use_fast=True, legacy=False, trust_remote_code=True)
    tokenizer.padding_side = args.padding_side
    # add a pad token id if needed, but do not use a pad_token equal to the bos_token, since this will lead to bad behavior
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.unk_token})
        elif tokenizer.eos_token_id is not None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        elif 'qwen' in args.model.lower():
            tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
        else:
            raise ValueError("Need to set the tokenizer pad token id, but missing both an unk_token_id and eos_token_id to use")
    # if the bos token was already equal to the unk or eos tokens...just get rid of it. our collate_fn then will not use any bos token
    if tokenizer.bos_token == tokenizer.pad_token:
        tokenizer.bos_token, tokenizer.bos_token_id = None, None
    # force persimmon bos to None
    if 'persimmon' in args.model:
        tokenizer.bos_token, tokenizer.bos_token_id = None, None
    if args.max_seq_len > 0:
        tokenizer.max_length = args.max_seq_len

    # load model
    if args.load_model:
        print("Loading model...")
        save_load_path = utils.get_model_save_load_path(args)
        model = utils.load_model(args, save_load_path, first_load=True)
        print(f"Num model parameters: {sum([np.prod(p.size()) for p in model.parameters()])/1e6:.2f}m")
    else:
        model = None

    # peft-wrap the model
    if ('finetuned' in args.hardness_method and args.estimate_hardness) or ('finetuned' in args.probing_method and args.do_eval) and args.optimize_weights in ['LORA', 'IA3']:
        print("Peft-wrapping model")
        model = utils.PEFT_wrap_model(args, model)
        model.print_trainable_parameters()

    # write/load representations for all loaded datasets
    if args.write_hidden_states:
        print("Writing hidden states...")
        start_write = time.time()
        write_hidden_states(args, datasets, model, tokenizer, prompt, log, overwrite_existing=args.overwrite_existing_data, verbose=True)
        print(f"Writing took {(time.time()-start_write)/60:.2f} minutes")
    if 'learned' in [args.probing_method, args.hardness_method]:
        start_load = time.time()        
        precomputed_hidden_states = load_hidden_states(args, datasets, prompt) # takes datasets as an arg in order to iterate over datanames
        print(f"Loading hidden states took {(time.time()-start_load):.2f} seconds")
    else:
        precomputed_hidden_states = None

    # first experiment: hardness estimation and sample efficiency results
    if args.estimate_hardness:
        # load hardness probe
        if args.hardness_method in ['learned', 'finetuned']:
            dataset_max_num_answers = [data_utils.get_max_num_answers(dataset['probing_data']) for dataset in datasets.values()]
            probe_num_classes = max(dataset_max_num_answers) if args.probing_token_state == 'question_end_token' else None
        else:
            probe_num_classes = None
        hardness_probe = Probe(args, 
                    probing_method=args.hardness_method,
                    probe_loss=args.probe_loss,
                    tokenizer=tokenizer,
                    normalize_representations=args.normalize_representations,
                    calibrate_probe=False,
                    model=model,
                    num_classes=probe_num_classes,
                    probing_config=hardness_probing_config,
        )
        if args.hardness_method == 'decoding':
            supervision_n = [0]
        else:
            supervision_n = [5, 20, 80, 340, 900]
        print(f"Estimating hardness scores for conditions: datasets: {len(datasets)} | bootstraps: {args.hardness_bootstraps} | prompts: {args.num_prompts} | supervision_n: {len(supervision_n)}...")
        start = time.time()
        write_MDL_scores(args, datasets, data_source, precomputed_hidden_states, hardness_probe, prompt,
                              supervision_n, boot_times=args.hardness_bootstraps, verbose=True)
        print(f"Writing hardness scores took {(time.time()-start):.2f} seconds")
        if model is not None:
            print("Writing text prob scores...")
            start = time.time()
            write_text_prob_scores(args, datasets, model, tokenizer)
            print(f"Writing text prob scores took {(time.time()-start):.2f} seconds")

    print("Post-processing hardness scores...")
    datasets = postprocess_hardness_scores(args, datasets, verbose=False)
    
    # evaluate a model on the datasets
    if args.do_eval:
        # reload datasets based on post-processing if needed
        if any(var_name in args.hardness_var_name for var_name in ['model_based', 'question_prob', 'answer_prob', 'reasoning_prob']):
            datasets = data_utils.load_datasets(args, datanames, args.data_dir, args.seed)
        # get n_train values for a learning curve
        if args.probing_method in ['learned']:
            supervision_n = [args.n_train] if not args.probing_learning_curve else [10, 20, 40, 80, 160, 320]
        if args.probing_method in ['finetuned']:
            supervision_n = [args.n_train] if not args.probing_learning_curve else [40, 80, 160, 320]
        elif args.probing_method == 'decoding':
            if args.probing_learning_curve:
                if 'ai2_arc' in args.dataset:
                    supervision_n = [0, 5, 10, 20]
                elif 'mmlu' in args.dataset:
                    supervision_n = [0, 5, 10, 20]
                elif 'strategy-qa' in args.dataset:
                    supervision_n = [4, 8]
                elif 'gsm8k' in args.dataset:
                    supervision_n = [4, 8]
            else:
                supervision_n = [args.k_shot]
        args.max_n_train = max(supervision_n)

        # load probe for model eval
        dataset_max_num_answers = [data_utils.get_max_num_answers(dataset['probing_data']) for dataset in datasets.values()]
        probe_num_classes = max(dataset_max_num_answers) if args.probing_token_state == 'question_end_token' else None
        probe = Probe(args, 
            probing_method=args.probing_method,
            probe_loss=args.probe_loss,
            tokenizer=tokenizer,
            normalize_representations=args.normalize_representations,
            calibrate_probe=args.calibrate_probe,
            num_classes=probe_num_classes,
            model=model,
            probing_config=probing_config,
        )
        run_modeling_experiment(args, datasets, data_source, probe, tokenizer, prompt, supervision_n, log, 
                                verbose=True)

    print("Experiment runtime: ", utils.format_time(time.time()-experiment_start))