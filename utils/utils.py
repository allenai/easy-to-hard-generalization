import argparse
import numpy as np
import pandas as pd
import torch
import os
import re
import pynvml
import globals
from itertools import chain

from peft import get_peft_model, LoraConfig, TaskType, IA3Config, AutoPeftModelForCausalLM
from transformers import AutoConfig, LlamaForCausalLM, AutoModelForSeq2SeqLM, AutoModelForCausalLM

def gather_item_level_stats_df(id_no_boot_to_item_accs, n_train, dataname, prompt_id, hardness_col_name=None):
    # get all the item level test accs across bootstrapped test samples
    test_stats_df = id_no_boot_to_item_accs[dataname][prompt_id][n_train][0].loc[:, ['accuracy']].copy()
    for i in range(1, len(id_no_boot_to_item_accs[dataname][prompt_id][n_train])):
        join_df = id_no_boot_to_item_accs[dataname][prompt_id][n_train][i].loc[:, ['accuracy']].rename(columns={'accuracy': f'accuracy{i}'})
        test_stats_df = pd.merge(test_stats_df, join_df, left_index=True, right_index=True, how='outer')
    # add hardness column variable for item level stats
    if hardness_col_name is not None:
        hardness_col_level = hardness_col_name + '_level'
        test_stats_df[hardness_col_name] = id_no_boot_to_item_accs[dataname][prompt_id][n_train][0][hardness_col_name]
        test_stats_df[hardness_col_level] = id_no_boot_to_item_accs[dataname][prompt_id][n_train][0][hardness_col_level]
    return test_stats_df

def get_hardness_col_names(model_name, normed=False):
    '''
    Gets the final list of model-based hardness cols that are written to probing_data, based on the argument supplied as args.hardness_var_name
    '''
    human_variables = ['human_hardness', 
                           'human_grade', 'human_difficulty', 'human_bloom', # 'human_depth_of_knowledge', 
                           'num_steps', 'question_num_words', 'answer_num_words', 'reasoning_num_words', 'answer_num_chars']
    hardness_metrics = [
        f'MDL_finetuned_{model_name}', 
        f'MDL_learned_{model_name}', 
        f'MDL_decoding_{model_name}', 
        f'question_prob_{model_name}', 
        f'answer_prob_{model_name}', 
        f'reasoning_prob_{model_name}',
        f'MDL_finetuned_model-avg', 
        f'MDL_learned_model-avg', 
        f'MDL_decoding_model-avg', 
        f'question_prob_model-avg',
        f'answer_prob_model-avg',
        f'reasoning_prob_model-avg'
    ]
    if normed:
        hardness_metrics = [x + "_NORMED" for x in hardness_metrics]
    return human_variables + hardness_metrics

def get_all_possible_hardness_col_names(model_name):
    hardness_col_names = get_hardness_col_names(model_name, normed=False)
    hardness_col_names = list(chain(*[[hardness_col_name + "_mean", hardness_col_name + "_std"] for hardness_col_name in hardness_col_names]))
    hardness_col_names = list(chain(*[[hardness_col_name + "_TRAIN", hardness_col_name + "_TEST"] for hardness_col_name in hardness_col_names]))
    return hardness_col_names

def get_mean_std_metrics_from_df(text_data, hardness_var_names, postfix=""):
    # used for adding avg and std of each hardness variable to the train and test dataframes, for saving with probing results
    if text_data is None:
        hardness_properties = {
            hardness_var_name + "_mean" + postfix: None
            for hardness_var_name in hardness_var_names
         }
        hardness_properties.update({
            hardness_var_name + "_std" + postfix: None
            for hardness_var_name in hardness_var_names
        })
    else:
        hardness_properties = {
            hardness_var_name + "_mean" + postfix: text_data[hardness_var_name].mean()
            for hardness_var_name in hardness_var_names if hardness_var_name in text_data.columns
        }
        hardness_properties.update({
            hardness_var_name + "_std" + postfix: text_data[hardness_var_name].std()
            for hardness_var_name in hardness_var_names if hardness_var_name in text_data.columns
        })
    return hardness_properties

def average_df_over_metrics(df, grouping_vars, metrics_vars):
    # averages the metrics_vars columns in a df, while keeping grouping_vars
    collapsed_dfs = []
    for metric in metrics_vars:
        if metric in df.columns:
            avg_df = df.groupby(grouping_vars)[metric].mean().reset_index()
            collapsed_dfs.append(avg_df)
    joined_df = collapsed_dfs[0]
    for collapsed_df in collapsed_dfs[1:]:
        joined_df = joined_df.merge(collapsed_df)
    return joined_df

def get_model_size(model):
    match = re.search(r'(\d+[bB])', model)
    if match:
        size = re.search(r'(\d+[bB])', model).group(1)
    match = re.search(r'(\d+)x(\d+)[bB]', model)
    if match:
        num1, num2 = map(int, match.groups())
        size = f"{num1 * num2}b"
        return size.lower()
    return size.lower()

def get_hardness_col_name(metric, model_name, model_avg=False):
    # maps from args.hadness_var_name to the column in the dataset, based on model_name
    if 'human' in metric or 'num_steps' in metric or 'words' in metric or 'chars' in metric:
        return metric
    else:
        metric_short = metric.replace("_avg", "")
        if 'model_based' in metric_short:
            metric_short = metric_short.replace("model_based", "MDL")
        if model_avg:
            short_model_name = 'model-avg'
        else:
            short_model_name = model_name.split('/')[-1]
        hardness_col_name = f"{metric_short}_{short_model_name}"
        return hardness_col_name

def PEFT_wrap_model(args, model):
    manually_add = ['mistral', 'falcon', 'persimmon', 'mpt', 'qwen']
    if args.optimize_weights in ['LORA', 'IA3']:
        task_type = TaskType.SEQ_2_SEQ_LM if args.model_type == 'encoder_decoder' else TaskType.CAUSAL_LM
        if args.optimize_weights == 'LORA':
            peft_config = LoraConfig(
                task_type=task_type, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1
            )
        if args.optimize_weights == 'IA3':
            peft_config = IA3Config(task_type=task_type, inference_mode=False)
        if any(x in args.model.lower() for x in manually_add):
            if 'mistral' in args.model.lower() or 'mixtral' in args.model.lower():
                peft_config.target_modules = ["q_proj", "v_proj"]
            elif 'persimmon' in args.model.lower():
                peft_config.target_modules = ["query_key_value", "dense"]
            elif 'mpt' in args.model.lower():
                peft_config.target_modules = ["Wqkv"]
            elif 'falcon' in args.model.lower():
                peft_config.target_modules = ["query_key_value"]
            elif 'qwen' in args.model.lower():
                peft_config.target_modules = ["c_attn"]
        assert any([x in args.model.lower() for x in ['llama', 'gpt-j', 'mistral', 'persimmon', 'mpt', 'falcon', 'qwen']]), f"\nNeed to add QLoRA params to peft_config manually -- add exact q_proj and v_proj layer paths to peft_config.target_modules = [paths] from the model: \n{model} \n(SEE MESSAGE ABOVE)"
        model = get_peft_model(model, peft_config)
    return model

def load_model(args, save_load_path=None, first_load=False):
    model_type_dict = {'encoder-decoder': AutoModelForSeq2SeqLM, 'decoder': AutoModelForCausalLM}
    model_type = model_type_dict[args.model_type]
    short_model_name = shorten_model_name(args.model)
    size = get_model_size(args.model)
    load_8bit = args.quantization == '8bit'
    load_4bit = args.quantization == '4bit'
    # load from a trained model
    load_from_trained_model = not first_load and save_load_path is not None and os.path.exists(save_load_path)
    if load_from_trained_model:
        print(f"Loading from path: {save_load_path}")
        final_folder = size.upper()
        if 'chat' in args.model:
            final_folder += '-chat'
        llama2_path = f""
        maybe_get_config_here = llama2_path if 'Llama-2' in args.model else None
        model_config = get_custom_config(args, maybe_get_config_here)
        if args.quantization == 'NA':
            model = model_type.from_config(model_config)
            state_dict = torch.load(save_load_path)
            model.load_state_dict(state_dict)
        elif args.quantization in ['4bit', '8bit']:
            if args.optimize_weights=='LORA':
                task_type = TaskType.SEQ_2_SEQ_LM if args.model_type == 'encoder_decoder' else TaskType.CAUSAL_LM
                peft_config = LoraConfig(task_type=task_type, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
                model = AutoPeftModelForCausalLM.from_pretrained(pretrained_model_name_or_path=save_load_path, config=peft_config, cache_dir=None, device_map='auto', load_in_4bit=load_4bit, load_in_8bit=load_8bit, low_cpu_mem_usage=True)
            else:
                model = model_type.from_pretrained(save_load_path, config=model_config, cache_dir=None, device_map='auto', load_in_4bit=load_4bit, load_in_8bit=load_8bit, low_cpu_mem_usage=True)
        elif args.quantization == '16bit':
            model = model_type.from_pretrained(save_load_path, config=model_config, cache_dir=None, device_map='auto', torch_dtype=torch.float16)
    # load local Llama-2 weights
    elif 'Llama-2' in args.model:
        final_folder = size.upper()
        if 'chat' in args.model:
            final_folder += '-chat'
        llama2_path = f""
        model_config = get_custom_config(args, llama2_path)
        model_type = LlamaForCausalLM
        if args.quantization in ['NA', '4bit', '8bit']:
            model = model_type.from_pretrained(llama2_path, config=model_config, cache_dir=None, device_map='auto', 
                                                    load_in_4bit=load_4bit, load_in_8bit=load_8bit, low_cpu_mem_usage=True)
        elif args.quantization == '16bit':
            model = model_type.from_pretrained(save_load_path, config=model_config, cache_dir=None, device_map='auto', torch_dtype=torch.float16)
    # load new or pretrained model
    else:
        # load config (has some custom adjustments to set dropout to 0)
        model_config = get_custom_config(args)
        if args.quantization in ['NA', '4bit', '8bit']:
            model = model_type.from_pretrained(args.model, config=model_config, cache_dir=args.cache_dir, device_map='auto', 
                                        load_in_4bit=load_4bit, load_in_8bit=load_8bit, low_cpu_mem_usage=True, trust_remote_code=True) 
        elif args.quantization == '16bit':
            model = model_type.from_pretrained(args.model, config=model_config, cache_dir=args.cache_dir, device_map='auto', 
                                                torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
    if short_model_name not in globals.model_to_hidden_size:
            print(f"You should add model hidden size of {model_config.hidden_size} to globals.model_to_hidden_size for {args.model}")
    model = model.eval()
    if not args.quantization:
        model = model.to(args.device)
    return model

def standardize_optimization_config(n_train, num_answers, max_batch_size, 
                                    grad_accumulation_factor=None, probing_epochs=None,
                                    minimum_grad_updates=10):
    '''
    We want to standardize the amount and manner of optimization applied during finetuning, for different n_train sizes.
    The goal is to get an effective batch size of up to 50, and generally apply 5 epochs of finetuning.
    The effective batch size is capped by the number of items in the training data
    '''
    num_points = n_train*num_answers
    if num_points < max_batch_size:
        print("Dataset size * num_answers smaller than requested batch size...capping batch size by train data size and setting grad accumulation factor to 1.")
        dataloader_batch_size = num_points
        effective_batch_size = num_points
        gaf = 1
    else:
        items_per_batch = max_batch_size // num_answers
        effective_batch_size = min(n_train, 50)
        dataloader_batch_size = max_batch_size
        gaf = effective_batch_size // items_per_batch
    gaf = gaf if not grad_accumulation_factor else grad_accumulation_factor
    # initial set of probing_epochs
    probing_epochs = 3 if probing_epochs is None or probing_epochs <= 0 else probing_epochs
    # now get the number of updates per batch, and ensure it is at least minimum_grad_updates
    updates_per_epoch = int(np.ceil(n_train / effective_batch_size))
    total_updates = updates_per_epoch * probing_epochs
    if total_updates < minimum_grad_updates:
        probing_epochs = int(np.ceil(minimum_grad_updates / updates_per_epoch))
    optimization_config = {
        'probing_epochs' : probing_epochs,
        'train_batch_size': dataloader_batch_size,
        'grad_accumulation_factor': gaf
    }
    return optimization_config

def get_hardness_experiment_name(args, model_override=None, method_override=None):
    # make hardness experiment name. Also used in naming hardness variable columns
    model_name = shorten_model_name(args.model) if not model_override else shorten_model_name(model_override)
    use_method = method_override if method_override else args.hardness_method
    if use_method == 'learned':
        probe_enc_dec = "-".join([x[:3] for x in args.model_type.split('_')])
        probe_layers = 'midlast' if args.probing_layers == 'middle_and_last' else args.probing_layers
        probing_insert = f"_{args.hardness_probe_model}" 
        probing_insert += f"_{probe_enc_dec}"
        probing_insert += f"_lyrs-{probe_layers}"
        probing_insert += f"_mt-{str(args.probing_multitask)[0]}_mp-{str(args.probing_multiprompt)[0]}"
        _probing_style = 'learned' if args.probe_loss != 'random' else 'random'
    elif use_method == 'decoding':
        _probing_style = 'decoding'
        probing_insert = ""
    elif use_method == 'finetuned':
        _probing_style = 'finetuned'
        probing_insert = f"-{args.optimize_weights}"
    experiment_name = f"{model_name}_{_probing_style}" + \
        f"{probing_insert}" + \
        f"_prompts-{args.num_prompts}" + \
        f"_boots-{args.hardness_bootstraps}" + \
        f"_sd{args.seed}"
    if args.debug:
        experiment_name += "_DEBUG"
    return experiment_name

def get_experiment_name(args):
    # make experiment name
    model_name = shorten_model_name(args.model)
    if args.probing_method == 'learned':
        probe_enc_dec = "-".join([x[:3] for x in args.model_type.split('_')])
        probe_layers = 'midlast' if args.probing_layers == 'middle_and_last' else args.probing_layers
        probing_insert = f"_{args.probe_model}" 
        probing_insert += f"_{probe_enc_dec}"
        probing_insert += f"_lyrs-{probe_layers}"
        probing_insert += f"_mt-{str(args.probing_multitask)[0]}_mp-{str(args.probing_multiprompt)[0]}"
        _probing_style = 'learned' if args.probe_loss != 'random' else 'random'
        if args.probe_loss == 'supervised':
            loss_insert = ""
        elif args.probe_loss == 'CCS':
            loss_insert = "_CCS"
        elif args.probe_loss == 'CCS_ours':
            loss_insert = "_CCS-ours"
        elif args.probe_loss == 'unsupervised':
            loss_insert = "_unsup"
        elif args.probe_loss == 'random':
            loss_insert = "" # probing_style gets edited above
        elif args.probe_loss == 'mixed-supervision':
            loss_insert = '_mixed-sup'
        elif args.probe_loss == "LM_loss":
            loss_insert = ""
    elif args.probing_method == 'decoding':
        probing_insert = ""
        if args.k_shot == 0:
            probing_insert += "_ZS"
        _probing_style = 'decoding'
        loss_insert = ""
    elif args.probing_method == 'finetuned':
        _probing_style = 'finetuned'
        probing_insert = f"-{args.optimize_weights}"
        loss_insert = f"-{args.finetuning_objective}"
    if args.use_cot:
        probing_insert += '-CoT'
    if args.probing_learning_curve:
        probing_insert += "-LCs"
    if args.noise_labels_p > 0:
        probing_insert += f"_noise-{args.noise_labels_p}"
    if args.force_test_dataname != 'NA':
        probing_insert += f"_test-{args.force_test_dataname}"
    if args.stratify_hardness:
        if args.record_results_by_hardness:
            strat_insert = f"_train-{args.train_on}_test-all-splits"
        else:
            strat_insert = f"_train-{args.train_on}_test-{args.test_on}"
        if args.hardness_var_name != 'model-based':
            strat_insert += f"_{args.hardness_var_name}"
        if args.standardize_sample_sizes:
            strat_insert += "_sss"
        if args.use_extra_easy_data:
            strat_insert += "_extra-easy"
    else:
        strat_insert = f"_full-distr"
    experiment_name = f"{model_name}_{_probing_style}" + \
        f"{probing_insert}" + \
        f"{loss_insert}" + \
        f"{strat_insert}" + \
        f"_prompts-{args.num_prompts}" + \
        f"_boots-{args.probing_bootstraps}" + \
        f"_sd{args.seed}"
    if args.debug:
        experiment_name += "_DEBUG"
    return experiment_name

def shorten_model_name(model_name):
    model_name = model_name.replace('facebook/', '')
    model_name = model_name.replace('meta-llama/', '')
    model_name = model_name.replace('tiiuae/', '')
    model_name = model_name.replace('EleutherAI/', '')
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    return model_name

def get_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return f"{info.used//1024**3} GB."

def get_mem():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return f"{info.used//1024**3} GB."

def check_nan_weights(model):
    for param in model.parameters():
        if torch.isnan(param.data).any():
            return True
    return False

def get_custom_config(args, weights_location=None):
    # define variables for custom model configs
    if weights_location is not None:
        from_pretrained_or_path = os.path.join(weights_location, 'config.json')
    else:
        from_pretrained_or_path = args.model
    config = AutoConfig.from_pretrained(from_pretrained_or_path, cache_dir=args.cache_dir, trust_remote_code=True)
    # edit config
    if args.dropout >= 0:
        allowed_models = ['facebook/opt', 'gpt2', 'gpt-j', 'falcon', 'llama', 'Llama', 'mpt', 'adept', 'persimmon', 'mistral', 'qwen']
        assert any([x in args.model.lower() for x in allowed_models]), f"If overriding dropout during training, need to use model in {allowed_models} or extend this in utils.get_custom_config. See config options: {config}"
        for k,v in config.__dict__.items():
            if 'pdrop' in k or 'dropout' in k:
                setattr(config, k, args.dropout)
            return config
        
def str2bool(v):
    # used for boolean argparse values
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def print_rounded_array(array):
    print([round(x,2) for x in array])

def chunk_array(array, size):
    # return chunks from the array of size=size, in left to right order
    # if array % size != 0, then last components of the array are also added, but will not be of size=size
    if len(array) <= size:
        return [array]
    start_idx = 0
    chunks = []
    for end_idx in range(1, len(array)+1):
        if end_idx % size == 0 or end_idx == len(array):
            chunks.append(array[start_idx:end_idx])
            start_idx = end_idx
    return chunks

def min_max_mean(array):
    return {'min': round(np.min(array),2), 'mean:': round(np.mean(array),2), 'max:': round(np.max(array),2)}

def move_kwargs_to_gpu(kwargs):
    for k,v in kwargs.items():
        if type(v) is torch.Tensor:
            kwargs[k] = v.cuda(non_blocking=True)

def get_model_save_load_path(args):
    experiment = get_experiment_name(args)
    model_path = os.path.join(args.model_dir, f"{experiment}.pt")
    return model_path

def format_time(x):
    time_diff = x / 60
    unit = 'minutes' if time_diff < 60 else 'hours'
    time_diff = time_diff if time_diff < 60 else time_diff / 60
    time_msg = f"{time_diff:.2f} {unit}"
    return time_msg

def str2arg(v):
    if v.lower() in ('yes', 'true', 't', 'y') + ('no', 'false', 'f', 'n'):
        return str2bool(v)
    else:
        try:
            if float(v) % 1 == 0:
                return int(float(v))
            else:
                return float(v)
        except:
            return v

def args_from_cli_command(command):
    class DummyArgs:
        pass
    dummy_args = DummyArgs()
    command_dict = {}
    items = command.split()
    for idx, item in enumerate(items):
        if idx == len(items)-1:
            break
        if item[:2] == '--':
            k = item[2:]
            v = items[idx+1]
            command_dict[k] = str2arg(v)
        elif item[0] == '-':
            k = item[1:]
            v = items[idx+1]
            command_dict[k] = str2arg(v)
    for k,v in command_dict.items():
        setattr(dummy_args, k, v)
    return dummy_args

def get_experiment_name_from_command(command):
    args = args_from_cli_command(command)
    experiment_name = get_experiment_name(args)
    args.short_model_name = args.model.split('/')[-1]
    return experiment_name, args

def get_hardness_experiment_name_from_command(command):
    args = args_from_cli_command(command)
    experiment_name = get_hardness_experiment_name(args)
    args.short_model_name = args.model.split('/')[-1]
    return experiment_name, args