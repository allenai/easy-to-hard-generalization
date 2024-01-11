import argparse
import os
import utils
from utils import str2bool
import pandas as pd
import globals
import time
import torch
import PIL

from utils import data_utils, utils, plotting_utils

# define global and configs

# columns in datasets
all_hardness_col_names = utils.get_all_possible_hardness_col_names(model_name='self')
all_datasets = ['ai2_arc', 'mmlu_STEM-5', 'strategy-qa', 'gsm8k_main']
results_columns = ['model', 'dataname', 'test_dataname', 'probing_method', 'probe_loss', 'use_cot', 'noise_labels_p', 'n_train', 'n_dev', 'n_test', 'sample_size', 'prompt_id', 'boot_idx',
                   'hardness_var_name', 'train_on', 'test_on', 'train_acc', 'test_acc_str', 'test_acc', 'error_bar', 'test_acc_train_distr', 'force_test_dataname', 'quantization'] + \
                   all_hardness_col_names + globals.mmlu_subject_stat_cols

# metric names per dataset
model_based_metrics = ['model_based_finetuned', 'model_based_learned', 'model_based_decoding', 
                        'model_based_finetuned_avg', 'model_based_learned_avg', 'model_based_decoding_avg']
model_avg_metrics = ['model_based_finetuned_avg', 'model_based_learned_avg', 'model_based_decoding_avg']
universal_metrics = ['question_prob', 'answer_prob', 'question_prob_avg', 'answer_prob_avg', 'question_num_words', 'answer_num_words']
reasoning_metrics =  ['reasoning_prob', 'reasoning_prob_avg', 'reasoning_num_words']
num_words_metrics = ['question_num_words', 'answer_num_words', 'reasoning_num_words']
dataname_to_hardness_vars = {
    'ai2_arc': model_based_metrics + ['human_grade', 'human_difficulty', 'human_bloom'] + universal_metrics,
    'mmlu_STEM-5': ['model_based_decoding', 'model_based_decoding_avg'] + ['human_hardness'] + universal_metrics,
    'strategy-qa': ['model_based_decoding', 'model_based_decoding_avg'] + ['num_steps'] + ['question_prob', 'question_prob_avg', 'question_num_words'] + reasoning_metrics,
    'gsm8k_main': ['model_based_decoding', 'model_based_decoding_avg'] + ['num_steps'] + ['question_prob', 'question_prob_avg', 'question_num_words', 'answer_num_chars'] + reasoning_metrics,
}
dataname_to_human_hardness = {
    'ai2_arc': ['human_grade', 'human_difficulty', 'human_bloom'] + ['question_num_words', 'answer_num_words'],
    'mmlu_STEM-5': ['human_hardness'] + ['question_num_words', 'answer_num_chars'],
    'strategy-qa': ['num_steps'] + ['question_num_words', 'reasoning_num_words'],
    'gsm8k_main': ['num_steps'] + ['question_num_words', 'reasoning_num_words', 'answer_num_chars'],
}
dataname_to_stratify_vars = {
    'ai2_arc': ['human_grade', 'human_difficulty', 'human_bloom'] + ['question_num_words', 'answer_num_chars'] + model_avg_metrics,
    'mmlu_STEM-5': ['human_hardness'] + ['question_num_words', 'answer_num_chars'] + ['model_based_decoding_avg'],
    'strategy-qa': ['num_steps'] + ['question_num_words', 'reasoning_num_words'] + ['model_based_decoding_avg'],
    'gsm8k_main': ['num_steps'] + ['question_num_words', 'reasoning_num_words', 'answer_num_chars'] + ['model_based_decoding_avg', 'model_based_finetuned_avg'],
}
dataname_to_unique_variables = {
    'ai2_arc': ['human_grade', 'human_difficulty', 'human_bloom'],
    'mmlu_STEM-5': ['human_hardness'],
    'strategy-qa': ['num_steps'],
    'gsm8k_main': ['num_steps'],
}
data_x_hardness_var_to_cutoffs = {
    'ai2_arc': {
        'human_bloom': " --human_easy_max 2 --human_hard_min 4",
        'human_difficulty': " --human_easy_max 1 --human_hard_min 3",
        'human_grade': " --human_easy_max 5 --human_hard_min 8",
        'human_depth_of_knowledge': " --human_easy_max 1 --human_hard_min 3",
    },
    'mmlu_STEM-5': {
        'human_hardness': " --human_easy_max 0 --human_hard_min 1",
    },
    'strategy-qa': {
        'num_steps': " --human_easy_max 2 --human_hard_min 4",
    },
    'gsm8k_main': {
        'num_steps': " --human_easy_max 4 --human_hard_min 7",
    },
}

# for random model-specific overrides
model_config = {
    'mosaicml/mpt-7b': " --padding_side right",
}
# assumes 4x48gb for experiments looping
model_size_to_ZS_bs = {
    '1_gpu': {
        '7b': 12,
        '8b': 12,
        '13b': 8,
        '56b': -1,
        '70b': -1,
        '72b': -1,
    },
    '4_gpu': {
        '7b': 64,
        '8b': 64,
        '13b': 36,
        '56b': 4,
        '70b': 4,
        '72b': 4,
    }
}
model_size_to_finetune_or_gen_bs = {
    '1_gpu': {
        '7b': 8,
        '8b': 8,
        '13b': 4,
        '56b': -1,
        '70b': -1,
        '72b': -1,
    },
    '4_gpu': {
        '7b': 36,
        '8b': 36,
        '13b': 12,
        '56b': 4,
        '70b': 4,
        '72b': 4,
    }
}
model_size_to_min_grad_updates = {
    '7b': 10, 
    '8b': 10,
    '13b': 10,
    '56b': 10,
    '70b': 10,
    '72b': 10,
}
model_to_quantization_decoding = {
    "huggyllama/llama-7b": "8bit",
    "tiiuae/falcon-7b": "NA",
    "mistralai/Mistral-7B-v0.1": "8bit",
    "mosaicml/mpt-7b": "NA",
    "facebook/opt-13b": "8bit",
    "adept/persimmon-8b-base": "8bit",
    'Llama-2-7b': "8bit", 
    'Llama-2-13b': "8bit",
    'Llama-2-70b': "8bit",
    'Llama-2-7b-chat': "NA",
    'Llama-2-13b-chat': "8bit",
    'Llama-2-70b-chat': "8bit",
    'Qwen/Qwen-72B': "16bit", 
    'mistralai/Mixtral-8x7B-v0.1': "16bit",
}
model_to_quantization_finetuned = {
    "huggyllama/llama-7b": "8bit",
    "tiiuae/falcon-7b": "NA",
    "mistralai/Mistral-7B-v0.1": "8bit",
    "mosaicml/mpt-7b": "NA",
    "facebook/opt-13b": "8bit",
    "adept/persimmon-8b-base": "NA",
    'Llama-2-7b': "8bit",
    'Llama-2-13b': "8bit",
    'Llama-2-70b': "8bit",
    'Llama-2-7b-chat': "8bit",
    'Llama-2-13b-chat': "8bit",
    'Llama-2-70b-chat': "8bit",
    'Qwen/Qwen-72B': "16bit", 
    'mistralai/Mixtral-8x7B-v0.1': "16bit",
}
probing_split_configs = [
    ' --train_on all --test_on all',
    ' --train_on easy --test_on easy',
    ' --train_on medium --test_on easy',
    ' --train_on hard --test_on easy',
    ' --train_on easy_and_hard --test_on easy',
    ' --train_on easy --test_on medium',
    ' --train_on medium --test_on medium',
    ' --train_on hard --test_on medium',
    ' --train_on easy_and_hard --test_on medium',
    ' --train_on easy --test_on hard',
    ' --train_on medium --test_on hard',
    ' --train_on hard --test_on hard',
    ' --train_on easy_and_hard --test_on hard',
]
train_conditions = [
    ' --train_on all',
    ' --train_on easy',
    ' --train_on medium',
    ' --train_on hard',
    # ' --train_on easy_and_hard',
    ' --train_on zero_shot'
]

# regression data configs
# method_configs = {
#     'learned_CoT=False': ' --n_test -1 --hardness_method learned --probing_method learned --use_cot false --probe_loss supervised -llm false',
#     'finetuned_CoT=True': ' --n_test 500 --hardness_method finetuned --probing_method finetuned --probing_lr 1e-4 --use_cot true --finetuning_objective seq2seq --probe_loss LM_loss -llm true',
#     'finetuned_CoT=False': ' --n_test 500 --hardness_method finetuned --probing_method finetuned --probing_lr 1e-4 --use_cot false --finetuning_objective MC --probe_loss supervised -llm true',
#     'decoding_CoT=True': ' --n_test 500 --hardness_method decoding --probing_method decoding --use_cot true --probe_loss supervised -llm true',
#     'decoding_CoT=False': ' --n_test 500 --hardness_method decoding --probing_method decoding --use_cot false --probe_loss supervised -llm true',
# }
method_configs = {
    'learned_CoT=False': ' --n_test -1 --hardness_method learned --probing_method learned --use_cot false --probe_loss supervised -llm false',
    'finetuned_CoT=True': ' --n_test -1 --hardness_method finetuned --probing_method finetuned --probing_lr 1e-4 --use_cot true --finetuning_objective seq2seq --probe_loss LM_loss -llm true',
    'finetuned_CoT=False': ' --n_test -1 --hardness_method finetuned --probing_method finetuned --probing_lr 1e-4 --use_cot false --finetuning_objective MC --probe_loss supervised -llm true',
    # 'finetuned_CoT=False': ' --n_test -1 --hardness_method finetuned --probing_method finetuned --probing_lr 1e-4 --use_cot false --finetuning_objective seq2seq --probe_loss LM_loss -llm true', # use to enable tbs < 4 for arc/mmlu, to save memory. But then need to increase mgu!
    'decoding_CoT=True': ' --n_test -1 --hardness_method decoding --probing_method decoding --use_cot true --probe_loss supervised -llm true',
    'decoding_CoT=False': ' --n_test -1 --hardness_method decoding --probing_method decoding --use_cot false --probe_loss supervised -llm true',
}
data_x_method_configs = {
    'ai2_arc_CoT=True': None,
    'ai2_arc_CoT=False': ' --specific_prompt 0040',
    'strategy-qa_CoT=True': ' --specific_prompt 0305 -mgl 100 --generative_eval true',
    'strategy-qa_CoT=False': ' --specific_prompt 0306',
    'mmlu_STEM-5_CoT=True': None,
    'mmlu_STEM-5_CoT=False': ' --specific_prompt 0040',
    'gsm8k_main_CoT=True': ' --n_test 500 --specific_prompt 0305 -mgl 300 --generative_eval true',
    'gsm8k_main_CoT=False': ' --specific_prompt 0306 -mgl 5 --finetuning_objective seq2seq --probe_loss LM_loss --generative_eval true',
}

def get_config_from_args(args, attrs):
    config = {}
    for attr in attrs:
        if hasattr(args, attr):
            config[attr] = getattr(args, attr)
        else:
            config[attr] = 'NA'
    return config

def filter_results_columns(results_df):
    results_df = results_df[results_columns].copy()
    # sort_columns = ['model', 'dataname', 'probing_method', 'noise_labels_p', 'use_cot', 'prompt_id', 
                #    'hardness_var_name', 'test_on', 'train_on'] # sort by test before train
    sort_columns = ['model', 'dataname', 'test_dataname', 'probing_method', 'noise_labels_p', 'use_cot', 'prompt_id', 
                   'hardness_var_name', 'train_on', 'test_on'] # sort by train before test
    results_df['train_on'] = pd.Categorical(results_df["train_on"].values,
                                    categories=["all", "easy", "medium", "hard", "easy_and_hard"], ordered=True)
    results_df['test_on'] = pd.Categorical(results_df["test_on"].values,
                                    categories=["all", "easy", "medium", "hard"], ordered=True)
    results_df = results_df.sort_values(by=sort_columns, ascending=[True]*len(sort_columns))
    return results_df

def job_function(job_command):
    os.system(job_command)

def get_base_command(args):
    command = f"python temp/main.py --gpu {args.gpu} "
    return command

def accumulate_results_to_df(results_path, running_results_df, config_dict, exp_args, filter_to_n_train=None):
    # this adds read results from the experiment to the running_results_df which has columns=results_columns
    try:
        experiment_results = pd.read_csv(results_path)
        if filter_to_n_train:
            experiment_results = experiment_results[experiment_results['n_train'] == filter_to_n_train]
        # ad-hoc copy to server-wide nfs
        ad_hoc_save_path = os.path.join("/net/nfs.cirrascale/aristo/peterh/outputs", results_path.split('/')[-1])
        if not os.path.exists(ad_hoc_save_path):
            print("COPYING RESULT TO NFS")
            experiment_results.to_csv(ad_hoc_save_path, index=False)
        for k,v in config_dict.items():
            experiment_results[k] = v
        # rename hardness vars in experiment results to use 'self' rather than 'model' for comparability models
        used_model = exp_args.model.split('/')[-1]
        new_cols = [col_name.replace(used_model, 'self') if used_model in col_name else col_name for col_name in experiment_results.columns]
        experiment_results.columns = new_cols
        combined_results = pd.concat([running_results_df, experiment_results])
        return combined_results
    except:
        print(f"\nWARNING: Couldn't find result sheet: {results_path}\n")
        return running_results_df

def write_hidden_states(args):
    base_command = get_base_command(args)
    exp_specific_args = f" --write_hidden_states true --hardness_bootstraps 1"
    # for dataname in all_datasets:
    for dataname in [args.dataset]:
        # for model in globals.hardness_models + llama_models:
        # for model in globals.one_gpu_models:
        models = ['Llama-2-70b']
        # llama_models = ['Llama-2-7b', 'Llama-2-13b', 'Llama-2-70b', 'Llama-2-7b-chat', 'Llama-2-13b-chat', 'Llama-2-70b-chat']
        for model in models:
            method_config = method_configs['learned_CoT=False']
            data_x_method_config = data_x_method_configs[dataname + '_CoT=False']
            bs = model_size_to_ZS_bs[f'{n_gpu}_gpu'][utils.get_model_size(model)]
            quantization = model_to_quantization_decoding[model]
            if quantization != '8bit':
                bs = int(bs / 2)
            model_override = model_config[model] if model in model_config else ""
            model_and_bs = f" --load_model true --model {model} {model_override} --quantization {quantization} --eval_batch_size {bs}"
            dataname_override = f" --dataset {dataname}"
            experiment_command = base_command + exp_specific_args + args.add_job_args + method_config + data_x_method_config + model_and_bs + dataname_override
            exp_name, exp_args = utils.get_hardness_experiment_name_from_command(experiment_command)
            assert args.run_jobs
            print(f"\n\nStarting job | {args.experiment} | {exp_name}")
            print(experiment_command)
            job_function(experiment_command)
    rep_dir = os.path.join(args.data_dir, 'representations')
    print(f"Written hidden representations to {rep_dir}...")
    os.system(f"du -sh {rep_dir}/*")
    return

def estimate_hardness(args):
    base_command = get_base_command(args)
    exp_specific_args = f" --estimate_hardness true"
    dataname_to_methods = {
        'ai2_arc': ['decoding', 'learned', 'finetuned'],
        'mmlu_STEM-5': ['decoding'],
        'strategy-qa': ['decoding'],
        'gsm8k_main': ['decoding', 'finetuned'],
    }
    # for dataname in all_datasets:
    for dataname in [args.dataset]:
        for model in globals.hardness_models:
            use_methods = dataname_to_methods[dataname]
            model_size = utils.get_model_size(model)
            for method in use_methods:
                method_config = method_configs[f'{method}_CoT=False']
                data_x_method_config = data_x_method_configs[dataname + '_CoT=False']
                model_size_config = model_size_to_ZS_bs if method == 'decoding' else model_size_to_finetune_or_gen_bs
                bs = model_size_config[f'{n_gpu}_gpu'][model_size]
                quantization = model_to_quantization_finetuned[model] if method == 'finetuned' else model_to_quantization_decoding[model]
                if quantization != '8bit':
                    bs = max(4,int(bs / 2))
                model_override = model_config[model] if model in model_config else ""
                model_and_bs = f" --model {model} {model_override} --quantization {quantization} --train_batch_size {bs} --eval_batch_size {bs}"
                dataname_override = f" --dataset {dataname}"
                num_boots = 1 if method == 'decoding' or (method == 'finetuned' and '70b' in model_size) else args.num_bootstraps
                hardness_bootstraps = f" --hardness_bootstraps {num_boots}"
                grad_updates = f" --minimum_gradient_updates {model_size_to_min_grad_updates[model_size]}"
                experiment_command = base_command + exp_specific_args + args.add_job_args + method_config + data_x_method_config + \
                                     model_and_bs + dataname_override + hardness_bootstraps + grad_updates
                exp_name, exp_args = utils.get_hardness_experiment_name_from_command(experiment_command)
                assert args.run_jobs
                print(f"\n\nStarting job | {args.experiment} | {exp_name}")
                print(experiment_command)
                job_function(experiment_command)
    return

def all_to_all_table(args):
    exp_specific_args = f" --do_eval true --record_results_by_hardness false --standardize_sample_sizes false --all_data_to_test true --stratify_hardness false --probing_bootstraps {args.num_bootstraps}"
    base_command = get_base_command(args)
    experiment_results = pd.DataFrame(columns=results_columns)
    experiment_all_boot_results = pd.DataFrame(columns=results_columns)
    all_item_level_accs_df = None
    config_attrs = ['model', 'probing_method', 'noise_labels_p', 'probe_loss', 'use_cot', 'hardness_var_name']
    non_CoT_methods = ['decoding_CoT=False', 'learned_CoT=False', 'finetuned_CoT=False']
    CoT_methods = ['decoding_CoT=True', 'finetuned_CoT=True']
    dataname_to_methods = {
        'ai2_arc': non_CoT_methods,
        'mmlu_STEM-5': non_CoT_methods,
        'strategy-qa': non_CoT_methods + CoT_methods,
        'gsm8k_main': CoT_methods, 
    }
    use_models = ['Llama-2-70b']
    for model in use_models:
        use_methods = dataname_to_methods[args.dataset]
        for method in use_methods:
            method_config = method_configs[f'{method}']
            data_x_method_config = args.dataset + '_' + method.split('_')[1]
            data_x_method_config = data_x_method_configs[data_x_method_config]
            model_size_config = model_size_to_ZS_bs if method == 'decoding_CoT=False' else model_size_to_finetune_or_gen_bs
            bs = model_size_config[f'{n_gpu}_gpu'][utils.get_model_size(model)]
            quantization = model_to_quantization_finetuned[model] if method == 'finetuned_CoT=False' else model_to_quantization_decoding[model]
            if quantization != '8bit':
                bs = max(4,int(bs / 2))
            model_override = model_config[model] if model in model_config else ""
            model_and_bs = f" --model {model} {model_override} --quantization {quantization} --train_batch_size {bs} --eval_batch_size {bs}"
            model_override = model_config[model] if model in model_config else ""
            min_grad_updates = model_size_to_min_grad_updates[utils.get_model_size(model)]
            if 'CoT=True' in method:
                min_grad_updates *= 15
            grad_updates = f" --minimum_gradient_updates {min_grad_updates}"
            experiment_command = base_command + exp_specific_args + args.add_job_args + method_config + data_x_method_config + \
                                model_and_bs + model_override + grad_updates
            exp_name, exp_args = utils.get_experiment_name_from_command(experiment_command)
            if args.run_jobs:
                results_path = os.path.join(args.output_dir, f"{args.dataset}_probing_results-best-prompt_{exp_name}.csv")
                skip_result = (not args.overwrite_existing_results and os.path.exists(results_path))
                if skip_result:
                    print("Skipping experiment saved at: ", results_path)
                else:
                    print(f"\n\nStarting job | {args.experiment} | {exp_name}")
                    print(experiment_command)
                    job_function(experiment_command)
            else:
                print(f"Collecting results | {args.experiment} | {exp_name}")
            results_path = os.path.join(args.output_dir, f"{args.dataset}_probing_results-best-prompt_{exp_name}.csv")
            config_dict = get_config_from_args(exp_args, config_attrs)
            experiment_results = accumulate_results_to_df(results_path, experiment_results, config_dict, exp_args)
            results_path = os.path.join(args.output_dir, f"{args.dataset}_probing_results_all-boots_{exp_name}.csv")
            experiment_all_boot_results = accumulate_results_to_df(results_path, experiment_all_boot_results, config_dict, exp_args)
            # SAVE RESULTS
            save_name = f"{args.dataset}_{args.experiment}_models-{len(use_models)}_LC-{1*args.probing_learning_curve}_prompts-{args.num_prompts}_boots-{args.num_bootstraps}"
            save_path = os.path.join('result_sheets', save_name + '.csv')
            experiment_results = filter_results_columns(experiment_results)
            experiment_results.to_csv(save_path, index=False)
            # gather item level acc dfs
            save_name = f'item_accs_{args.dataset}_{exp_name}.csv'
            acc_path = os.path.join(args.data_dir, save_name)
            if os.path.exists(acc_path):
                item_level_acc_df = pd.read_csv(acc_path)
                all_item_level_accs_df = pd.concat([all_item_level_accs_df, item_level_acc_df]) if all_item_level_accs_df is not None else  item_level_acc_df
            else:
                print(f" WARNING: could not find item level accs at: {acc_path}")
        # PER MODEL -- save item level acc dfs
        if all_item_level_accs_df is not None:
            save_name = f'item_level_accs_{args.dataset}_{args.experiment}_models-{len(use_models)}_LC-{1*args.probing_learning_curve}_prompts-{args.num_prompts}_boots-{args.num_bootstraps}.csv'
            item_level_accs_path = os.path.join('result_sheets', save_name)
            all_item_level_accs_df.to_csv(item_level_accs_path, index=False)
    return

def get_population_table(args):
    exp_specific_args = f" --do_eval true --record_results_by_hardness true --all_data_to_test true --stratify_hardness true --probing_bootstraps {args.num_bootstraps}"
    base_command = get_base_command(args)
    experiment_results = pd.DataFrame(columns=results_columns)
    experiment_all_boot_results = pd.DataFrame(columns=results_columns)
    all_item_level_accs_df = None
    config_attrs = ['model', 'probing_method', 'noise_labels_p', 'probe_loss', 'use_cot', 'hardness_var_name']
    non_CoT_methods = ['decoding_CoT=False', 'learned_CoT=False', 'finetuned_CoT=False']
    CoT_methods = ['decoding_CoT=True', 'finetuned_CoT=True']
    dataname_to_methods = {
        'ai2_arc': non_CoT_methods,
        'mmlu_STEM-5': non_CoT_methods,
        'strategy-qa': non_CoT_methods + CoT_methods,
        'gsm8k_main': CoT_methods, 
    }
    local_train_conditions = [
        ' --train_on easy',
        ' --train_on medium',
        ' --train_on hard',
        ' --train_on zero_shot',
    ]
    use_models = globals.base_llama_models
    # use_models = ['Llama-2-70b']
    for model in use_models:
        use_methods = dataname_to_methods[args.dataset]
        for method in use_methods:
            # stratify_var_names = dataname_to_stratify_vars[args.dataset]
            stratify_var_names = dataname_to_human_hardness[args.dataset]
            # stratify_var_names = dataname_to_unique_variables[args.dataset]
            for hardness_var_name in stratify_var_names:
                for train_condition in local_train_conditions:
                    method_config = method_configs[f'{method}']
                    data_x_method_config = args.dataset + '_' + method.split('_')[1]
                    data_x_method_config = data_x_method_configs[data_x_method_config]
                    model_size_config = model_size_to_ZS_bs if method == 'decoding_CoT=False' else model_size_to_finetune_or_gen_bs
                    bs = model_size_config[f'{n_gpu}_gpu'][utils.get_model_size(model)]
                    quantization = model_to_quantization_finetuned[model] if method == 'finetuned_CoT=False' else model_to_quantization_decoding[model]
                    if quantization != '8bit':
                        bs = max(4,int(bs / 2))
                    model_override = model_config[model] if model in model_config else ""
                    model_and_bs = f" --model {model} {model_override} --quantization {quantization} --train_batch_size {bs} --eval_batch_size {bs}"
                    model_override = model_config[model] if model in model_config else ""
                    data_x_hardness_var_cutoffs = data_x_hardness_var_to_cutoffs[args.dataset]
                    hardness_cutoffs = data_x_hardness_var_cutoffs[hardness_var_name] if hardness_var_name in data_x_hardness_var_cutoffs else ''
                    hardness_config = f" --hardness_var_name {hardness_var_name} {hardness_cutoffs}"
                    min_grad_updates = model_size_to_min_grad_updates[utils.get_model_size(model)]
                    if 'CoT=True' in method:
                        min_grad_updates *= 15
                    # custom train_on zero_shot setting. only do zero_shot once for decoding
                    if 'zero_shot' in train_condition:
                        data_x_method_config += ' --k_shot 0'
                        train_condition = ' --train_on all --probe_loss unsupervised --probing_bootstraps 1 --n_test -1'
                        if 'decoding' not in method:
                            continue
                        if 'CoT=True' in method and args.dataset == 'gsm8k_main':
                            model_override += " --specific_prompt 5305 --num_print 1"
                            train_condition += ' --probing_bootstraps 5 --n_test 500'
                    grad_updates = f" --minimum_gradient_updates {min_grad_updates}"
                    experiment_command = base_command + exp_specific_args + args.add_job_args + method_config + data_x_method_config + \
                                        model_and_bs + model_override + train_condition + hardness_config + grad_updates
                    exp_name, exp_args = utils.get_experiment_name_from_command(experiment_command)
                    # skip medium train for mmlu
                    if '--train_on medium' in experiment_command and 'mmlu' in args.dataset:
                        continue
                    if args.run_jobs:
                        results_path = os.path.join(args.output_dir, f"{args.dataset}_probing_results-best-prompt_{exp_name}.csv")
                        skip_result = (not args.overwrite_existing_results and os.path.exists(results_path))
                        if skip_result:
                            print("Skipping experiment saved at: ", results_path)
                        else:
                            print(f"\n\nStarting job | {args.experiment} | {exp_name}")
                            print(experiment_command)
                            job_function(experiment_command)
                    else:
                        print(f"Collecting results | {args.experiment} | {exp_name}")
                    results_path = os.path.join(args.output_dir, f"{args.dataset}_probing_results-best-prompt_{exp_name}.csv")
                    config_dict = get_config_from_args(exp_args, config_attrs)
                    experiment_results = accumulate_results_to_df(results_path, experiment_results, config_dict, exp_args)
                    results_path = os.path.join(args.output_dir, f"{args.dataset}_probing_results_all-boots_{exp_name}.csv")
                    experiment_all_boot_results = accumulate_results_to_df(results_path, experiment_all_boot_results, config_dict, exp_args)
                    # SAVE RESULTS
                    save_name = f"{args.dataset}_{args.experiment}_models-{len(use_models)}_LC-{1*args.probing_learning_curve}_prompts-{args.num_prompts}_boots-{args.num_bootstraps}"
                    save_path = os.path.join('result_sheets', save_name + '.csv')
                    experiment_results = filter_results_columns(experiment_results)
                    experiment_results.to_csv(save_path, index=False)
                    # SAVE all-boot RESULTS
                    save_name = f"{args.dataset}_{args.experiment}_all-boots_models-{len(use_models)}_LC-{1*args.probing_learning_curve}_prompts-{args.num_prompts}_boots-{args.num_bootstraps}"
                    save_path = os.path.join('result_sheets', save_name + '.csv')
                    experiment_all_boot_results = filter_results_columns(experiment_all_boot_results)
                    experiment_all_boot_results.to_csv(save_path, index=False)
                    # gather item level acc dfs
                    save_name = f'item_accs_{args.dataset}_{exp_name}.csv'
                    acc_path = os.path.join(args.data_dir, save_name)
                    if os.path.exists(acc_path):
                        item_level_acc_df = pd.read_csv(acc_path)
                        all_item_level_accs_df = pd.concat([all_item_level_accs_df, item_level_acc_df]) if all_item_level_accs_df is not None else  item_level_acc_df
                    else:
                        print(f" WARNING: could not find item level accs at: {acc_path}")
            # PER MODEL+METHOD -- save item level acc dfs
            if all_item_level_accs_df is not None:
                save_name = f'item_level_accs_{args.dataset}_{args.experiment}_models-{len(use_models)}_LC-{1*args.probing_learning_curve}_prompts-{args.num_prompts}_boots-{args.num_bootstraps}.csv'
                item_level_accs_path = os.path.join('result_sheets', save_name)
                all_item_level_accs_df.to_csv(item_level_accs_path, index=False)
    return

def noisy_labels_table(args):
    exp_specific_args = f" --do_eval true --record_results_by_hardness true --standardize_sample_sizes false --all_data_to_test true --stratify_hardness true --probing_bootstraps {args.num_bootstraps}"
    base_command = get_base_command(args)
    experiment_results = pd.DataFrame(columns=results_columns)
    config_attrs = ['model', 'probing_method', 'noise_labels_p', 'probe_loss', 'use_cot', 'hardness_var_name']
    noise_levels = [0, .05, .1, .2, .3] # p=.3 noise -> 2*p=.6 noise for hard data, corresponds to 40% correctly labeled + 15% correct by luck for 4-way problems.
    local_train_conditions = [
        ' --train_on easy', 
        ' --train_on hard']
    use_models = ['Llama-2-70b']
    for model in use_models:
        for method in ['learned_CoT=False']:
            for hardness_var_name in ['human_hardness']:
                for train_condition in local_train_conditions:
                    # custom train_on zero_shot setting. only do zero_shot once for decoding
                    method_config = method_configs[f'{method}']
                    data_x_method_config = args.dataset + '_' + method.split('_')[1]
                    data_x_method_config = data_x_method_configs[data_x_method_config]
                    # only get decoding for ZS
                    if 'decoding' in method and 'zero_shot' not in train_condition:
                        continue
                    # set ZS custom config
                    if 'zero_shot' in train_condition:
                        data_x_method_config += ' --k_shot 0'
                        train_condition = ' --train_on all --probe_loss unsupervised --n_test -1 --probing_bootstraps 1'
                        if method != 'decoding_CoT=False':
                            continue
                    for noise_p in noise_levels:
                        model_size_config = model_size_to_ZS_bs if method == 'decoding_CoT=False' else model_size_to_finetune_or_gen_bs
                        bs = model_size_config[f'{n_gpu}_gpu'][utils.get_model_size(model)]
                        quantization = model_to_quantization_finetuned[model] if method == 'finetuned_CoT=False' else model_to_quantization_decoding[model]
                        if quantization != '8bit':
                            bs = max(4,int(bs / 2))
                        model_override = model_config[model] if model in model_config else ""
                        model_and_bs = f" --model {model} {model_override} --quantization {quantization} --train_batch_size {bs} --eval_batch_size {bs}"
                        model_override = model_config[model] if model in model_config else ""
                        data_x_hardness_var_cutoffs = data_x_hardness_var_to_cutoffs[args.dataset]
                        # make noise config for non-ZS setting
                        if 'train_on easy' in train_condition:
                            hardness_based_noise_p = noise_p 
                            noise_config = f" --noise_labels_p {hardness_based_noise_p}"
                        elif 'train_on hard' in train_condition:
                            hardness_based_noise_p = 2*noise_p 
                            noise_config = f" --noise_labels_p {hardness_based_noise_p}"
                        elif 'zero_shot' in train_condition or 'probe_loss unsupervised' in train_condition:
                            noise_config = ''
                        hardness_cutoffs = data_x_hardness_var_cutoffs[hardness_var_name] if hardness_var_name in data_x_hardness_var_cutoffs else ''
                        hardness_config = f" --hardness_var_name {hardness_var_name} {hardness_cutoffs}"
                        min_grad_updates = model_size_to_min_grad_updates[utils.get_model_size(model)]
                        grad_updates = f" --minimum_gradient_updates {min_grad_updates}"
                        experiment_command = base_command + exp_specific_args + args.add_job_args + method_config + data_x_method_config + \
                                            model_and_bs + model_override + train_condition + hardness_config + noise_config + grad_updates
                        exp_name, exp_args = utils.get_experiment_name_from_command(experiment_command)
                        if args.run_jobs:
                            results_path = os.path.join(args.output_dir, f"{args.dataset}_probing_results-best-prompt_{exp_name}.csv")
                            skip_result = (not args.overwrite_existing_results and os.path.exists(results_path))
                            if skip_result:
                                print("Skipping experiment saved at: ", results_path)
                            else:
                                print(f"\n\nStarting job | {args.experiment} | {exp_name}")
                                print(experiment_command)
                                job_function(experiment_command)
                        else:
                            print(f"Collecting results | {args.experiment} | {exp_name}")
                        results_path = os.path.join(args.output_dir, f"{args.dataset}_probing_results-best-prompt_{exp_name}.csv")
                        config_dict = get_config_from_args(exp_args, config_attrs)
                        experiment_results = accumulate_results_to_df(results_path, experiment_results, config_dict, exp_args)
                        # SAVE RESULTS
                        save_name = f"{args.dataset}_{args.experiment}_models-{len(use_models)}_prompts-{args.num_prompts}_boots-{args.num_bootstraps}"
                        save_path = os.path.join('result_sheets', save_name + '.csv')
                        experiment_results = filter_results_columns(experiment_results)
                        experiment_results.to_csv(save_path, index=False)
    return

def third_grade_to_college(args):
    exp_specific_args = f" --dataset third_grade_to_college --force_test_dataname mmlu_STEM-5 --do_eval true --record_results_by_hardness false --test_on hard --standardize_sample_sizes false --all_data_to_test true --stratify_hardness true --probing_bootstraps {args.num_bootstraps}"
    base_command = get_base_command(args)
    experiment_results = pd.DataFrame(columns=results_columns)
    config_attrs = ['model', 'probing_method', 'probe_loss']
    # use_models = globals.base_llama_models
    use_models = ['mistralai/Mixtral-8x7B-v0.1']
    local_train_conditions = [
        ' --train_on zero_shot', 
        ' --train_on easy', 
        ' --train_on hard'
        ]
    for model in use_models:
        for method in ['decoding_CoT=False']:
            for train_condition in local_train_conditions:
                method_config = method_configs[f'{method}']
                data_x_method_config = ' --specific_prompt 0040'
                model_size_config = model_size_to_ZS_bs if method == 'decoding_CoT=False' else model_size_to_finetune_or_gen_bs
                bs = model_size_config[f'{n_gpu}_gpu'][utils.get_model_size(model)]
                quantization = model_to_quantization_finetuned[model] if method == 'finetuned_CoT=False' else model_to_quantization_decoding[model]
                if quantization != '8bit':
                    bs = max(4,int(bs / 2))
                model_override = model_config[model] if model in model_config else ""
                eval_bs = max(4, bs)
                model_and_bs = f" --model {model} --quantization {quantization} --train_batch_size {bs} --eval_batch_size {eval_bs}"
                min_grad_updates = model_size_to_min_grad_updates[utils.get_model_size(model)]
                # custom train_on zero_shot setting. only do zero_shot once for decoding
                if 'zero_shot' in train_condition:
                    data_x_method_config += ' --k_shot 0'
                    train_condition = ' --train_on easy --probe_loss unsupervised --force_train_dataname mmlu_STEM-5 --n_test -1 --probing_bootstraps 1'
                    if method != 'decoding_CoT=False':
                        continue
                if 'learned' in method:
                    data_x_method_config += ' --probing_bootstraps 10'
                if 'finetuned' in method and '--probe_loss LM_loss' in data_x_method_config:
                    min_grad_updates *= 4
                grad_updates = f" --minimum_gradient_updates {min_grad_updates}"
                experiment_command = base_command + args.add_job_args + exp_specific_args + method_config + data_x_method_config + \
                                    model_and_bs + model_override + train_condition + grad_updates
                exp_name, exp_args = utils.get_experiment_name_from_command(experiment_command)
                if args.run_jobs:
                    results_path = os.path.join(args.output_dir, f"mmlu_STEM-5_probing_results-best-prompt_{exp_name}.csv")
                    skip_result = (not args.overwrite_existing_results and os.path.exists(results_path))
                    if skip_result:
                        print("Skipping experiment saved at: ", results_path)
                    else:
                        print(f"\n\nStarting job | {args.experiment} | {exp_name}")
                        print(experiment_command)
                        job_function(experiment_command)
                else:
                    print(f"Collecting results | {args.experiment} | {exp_name}")
                for dataname in ['ai2_arc', 'ai2_arc_all', 'mmlu_STEM-5']:
                    if 'k_shot 0' in data_x_method_config and 'ai2_arc' in dataname:
                        continue
                    results_path = os.path.join(args.output_dir, f"{dataname}_probing_results-best-prompt_{exp_name}.csv")
                    config_dict = get_config_from_args(exp_args, config_attrs)
                    experiment_results = accumulate_results_to_df(results_path, experiment_results, config_dict, exp_args)
                # SAVE RESULTS
                save_name = f"{args.experiment}_models-{len(use_models)}_LC-{1*args.probing_learning_curve}_prompts-{args.num_prompts}_boots-{args.num_bootstraps}"
                save_path = os.path.join('result_sheets', save_name + '.csv')
                experiment_results = filter_results_columns(experiment_results)
                experiment_results.to_csv(save_path, index=False)
    return

def quantization_check(args):
    base_command = get_base_command(args)
    experiment_results = pd.DataFrame(columns=results_columns)
    config_attrs = ['model', 'probing_method', 'quantization', 'probe_loss', 'use_cot']
    exp_specific_args = f" --do_eval true --standardize_sample_sizes false --stratify_hardness false --probing_bootstraps {args.num_bootstraps} "
    use_models = ['huggyllama/llama-7b', 'Llama-2-13b']
    method = 'finetuned_CoT=False'
    # method = 'learned_CoT=False'
    for model in use_models:
        # for method in use_methods:
        for quantization in ['8bit', 'NA']:
            # for hardness_var_name in hardness_var_names:
                method_config = method_configs[f'{method}']
                data_x_method_config = args.dataset + '_' + method.split('_')[1]
                data_x_method_config = data_x_method_configs[data_x_method_config]
                model_size_config = model_size_to_ZS_bs if method == 'decoding' else model_size_to_finetune_or_gen_bs
                bs = model_size_config[f'{n_gpu}_gpu'][utils.get_model_size(model)]
                if quantization != '8bit':
                    bs = max(4,int(bs / 2))
                model_override = model_config[model] if model in model_config else ""
                model_and_bs = f" --model {model} {model_override} --quantization {quantization} --eval_batch_size {bs}"
                experiment_command = base_command + exp_specific_args + args.add_job_args + method_config + data_x_method_config + \
                                    model_and_bs
                exp_name, exp_args = utils.get_experiment_name_from_command(experiment_command)
                if args.run_jobs:
                    print(f"\n\nStarting job | {args.experiment} | {exp_name}")
                    print(experiment_command)
                    job_function(experiment_command)
                else:
                    print(f"Collecting results | {args.experiment} | {exp_name}")
                # collect results
                results_path = os.path.join(args.output_dir, f"{args.dataset}_probing_results-best-prompt_{exp_name}.csv")
                config_dict = get_config_from_args(exp_args, config_attrs)
                experiment_results = accumulate_results_to_df(results_path, experiment_results, config_dict, exp_args, only_last_row=not args.probing_learning_curve)
                # SAVE RESULTS
                save_name = f"{args.dataset}_{args.experiment}_models-{len(use_models)}_LC-{1*args.probing_learning_curve}_prompts-{args.num_prompts}_boots-{args.num_bootstraps}"
                save_path = os.path.join('result_sheets', save_name + '.csv')
                experiment_results = filter_results_columns(experiment_results)
                experiment_results.to_csv(save_path, index=False)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", '-e', type=str) 
    parser.add_argument("--seed", default=0, type=int, 
                        help='intended to determine hardness/probing splits from original datasets ONLY. bootstrapping logic controls training + testing randomness')
    parser.add_argument("--gpu", default=0, type=int) 
    parser.add_argument("--train_batch_size", '-tbs', type=int, default=8, help='')
    parser.add_argument("--eval_batch_size", '-ebs', type=int, default=8, help='')
    parser.add_argument("--grad_accumulation_factor", '-gaf', default=-1, type=int, help='')
    parser.add_argument("--debug", default=False, type=str2bool)
    # prompts and bootstraps
    parser.add_argument("--num_prompts", '-np', type=int, default=1, help='')
    parser.add_argument("--num_bootstraps", '-nb', type=int, default=5, help='')
    parser.add_argument("--specific_prompt", default=None, type=str, help='')
    # model, data, and other experiment naming args
    parser.add_argument("--model", default='Llama-2-13b', type=str, help='')
    parser.add_argument("--model_type", default='decoder', choices=['decoder', 'encoder_decoder'], help='')
    parser.add_argument("--quantization", default = '8bit', choices=['NA', '8bit', '4bit', '16bit'], type=str, help = 'quantize model for inference')
    parser.add_argument("--dataset", default="NA", type=str, help='')
    parser.add_argument("--probing_layers", default = 'middle_and_last', type=str,
                        choices=['middle_and_last', 'middle', 'last'], help='comma separated list of layer indices, or just "last" to indicate last layer')
    parser.add_argument("--hardness_probe_model", default='linear', choices=['linear', 'MLP', 'transformer'], help='only relevant if probing_method=learned')
    parser.add_argument("--probe_model", default='linear', choices=['linear', 'MLP', 'transformer'], help='only relevant if probing_method=learned')
    parser.add_argument("--k_shot", '-k', default=0, type=int, help='')
    parser.add_argument("--n_train", default=160, type=int, help='')
    parser.add_argument("--n_test", default=-1, type=int, help='')
    parser.add_argument("--probing_multitask", default = False, type=str2bool, help = 'probing model is trained multi-task on all passed training datasets')
    parser.add_argument("--probing_multiprompt", default = False, type=str2bool, help = 'probing model is trained multi-prompt, evaluated on a single prompt')
    parser.add_argument("--probing_task_transfer", default = False, type=str2bool, help = 'for two passed datasets, train on one and test on the other')
    parser.add_argument("--hardness_multitask", default = False, type=str2bool, help = 'hardness model is trained multi-task on all passed training datasets')
    parser.add_argument("--hardness_multiprompt", default = False, type=str2bool, help = 'hardness model is trained multi-prompt, evaluated on a single prompt')
    parser.add_argument("--stratify_hardness", default = False, type=str2bool, help = 'stratifies probing data by hardness variable')
    parser.add_argument("--max_seq_len", default=2048, type=int, help='')
    parser.add_argument("--hardness_var_name", default='NA', help='name of hardness column for data with human hardness metadata')
    parser.add_argument("--optimize_weights", default='LORA', help='name of hardness column for data with human hardness metadata')
    # save dirs
    parser.add_argument("--data_dir", default='/net/nfs.cirrascale/aristo/peterh/data', type=str, help='')
    parser.add_argument("--model_dir", default='/net/nfs.cirrascale/aristo/peterh/models', type=str, help='')
    parser.add_argument("--cache_dir", default='/net/nfs.cirrascale/aristo/peterh/models', type=str, help='')
    parser.add_argument("--output_dir", default='/net/nfs.cirrascale/aristo/peterh/outputs', type=str, help='')
    # parser.add_argument("--output_dir", default='~/latent-knowledge/ours/outputs', type=str, help='')
    # control flow
    parser.add_argument("--run_jobs", '-rj', default=True, type=str2bool)
    parser.add_argument("--probing_learning_curve", '-lc', default = False, type=str2bool, help = '')
    parser.add_argument("--overwrite_existing_results", '-oer', default = False, type=str2bool, help = 'will overwrite existing results files from past experiments')
    parser.add_argument("--overwrite_existing_models", default = True, type=str2bool, help = 'will overwrite existing models from past experiments')
    parser.add_argument("--overwrite_existing_measurements", '-oem', default = True, type=str2bool, help = 'will overwrite hardness variables and postprocessed datasets')
    parser.add_argument("--overwrite_existing_data", '-oed', default = False, type=str2bool, 
                        help = 'overwrite hidden states and x_probs in for questions/answers/probs')
    
    # parse
    args = parser.parse_args()
    start_time = time.time()
    if not os.path.exists('result_sheets'):
        os.mkdir('result_sheets')
    n_gpu = max(torch.cuda.device_count(), 1)

    # recreate codebase in temp/, so we can run experiments while editing code
    os.system("rsync -am --include='*.py' --include='*/' --exclude='*' ./ temp/")

    # add datanames to args
    data_source = data_utils.standardize_data_source(args.dataset)
    if hasattr(globals, args.dataset):
        datanames = getattr(globals, args.dataset)
    else:
        datanames = [args.dataset]
    args.datanames = datanames
    args.short_model_name = utils.shorten_model_name(args.model)
    # set promptsource
    prompt_source = 'custom' if data_source != 'burns' else 'promptsource'
    # add args -- these may be overridden later, but are required here for utils.get_experiment_name
    args.add_job_args = \
                   f" --debug {args.debug}"\
                   f" --seed {args.seed}"\
                   f" --data_dir {args.data_dir}"\
                   f" --model_dir {args.model_dir}"\
                   f" --cache_dir {args.cache_dir}"\
                   f" --output_dir {args.output_dir}"\
                   f" --dataset {args.dataset}"\
                   f" --optimize_weights {args.optimize_weights}"\
                   f" --use_extra_easy_data false"\
                   f" --probing_learning_curve {args.probing_learning_curve}"\
                   f" --k_shot {args.k_shot}"\
                   f" --n_train {args.n_train}"\
                   f" --n_test {args.n_test}"\
                   f" --model {args.model}"\
                   f" --model_type {args.model_type}"\
                   f" --hardness_probe_model {args.hardness_probe_model}"\
                   f" --probe_model {args.probe_model}"\
                   f" --probing_multitask {args.probing_multitask}"\
                   f" --probing_multiprompt {args.probing_multiprompt}"\
                   f" --hardness_multitask {args.hardness_multitask}"\
                   f" --hardness_multiprompt {args.hardness_multiprompt}"\
                   f" --probing_layers {args.probing_layers}"\
                   f" --num_prompts {args.num_prompts}"\
                   f" --prompt_source {prompt_source}"\
                   f" --max_seq_len {args.max_seq_len}"\
                   f" --noise_labels_p 0"\
                   f" --force_test_dataname NA"\
                   f" --standardize_sample_sizes false"\
                   f" --hardness_var_name {args.hardness_var_name}"\
                   f" --overwrite_existing_results {args.overwrite_existing_results}"\
                   f" --overwrite_existing_models {args.overwrite_existing_models}"\
                   f" --overwrite_existing_measurements {args.overwrite_existing_measurements}"\
                   f" --overwrite_existing_data {args.overwrite_existing_data}"
    if args.train_batch_size is not None:       args.add_job_args += f" --train_batch_size {args.train_batch_size}"
    if args.eval_batch_size is not None:        args.add_job_args += f" --eval_batch_size {args.eval_batch_size}"
    if args.grad_accumulation_factor > -1:      args.add_job_args += f" -gaf {args.grad_accumulation_factor}"
    
    # specify job to run
    if args.experiment == 'write_hidden_states':            write_hidden_states(args)
    if args.experiment == 'estimate_hardness':              estimate_hardness(args)
    if args.experiment == 'all_to_all_table':               all_to_all_table(args)
    if args.experiment == 'get_population_table':           get_population_table(args)
    if args.experiment == 'quantization_check':             quantization_check(args)
    if args.experiment == 'noisy_labels_table':             noisy_labels_table(args)
    if args.experiment == 'third_grade_to_college':         third_grade_to_college(args)

    print(f"\nTotal experiment runtime: {utils.format_time(time.time()-start_time)}")