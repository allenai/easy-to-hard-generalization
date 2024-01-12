import numpy as np
import pandas as pd
import sys
import os
import copy

import transformers
import datasets
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import chain
import json
from transformers import LlamaTokenizer, LlamaForCausalLM

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import globals
from utils import LM_utils
from utils import utils

class MCTextDataset(Dataset):
    '''
    multiple-choice text dataset
    - takes a pd dataframe of data that has been preprocessed by prepare_dataset_for_dataloader
    - expects data to follow a multiple choice format
    '''
    def __init__(
        self,
        args,
        probing_method,
        dataframe,
        dataname,
        data_source,
        tokenizer,
        batch_size,
        padding_side,
        max_seq_len,
        letter_labels=False,
        use_cot=False,
        for_training=False,
        force_generative_batch=False,
        precomputed_hidden_states=None, 
    ):
        self.args = args
        self.probing_method = probing_method
        self.dataframe = dataframe
        self.dataname = dataname
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.padding_side = padding_side
        self.max_seq_len = max_seq_len
        self.letter_labels = letter_labels
        self.for_training = for_training
        self.precomputed_hidden_states = precomputed_hidden_states # expected to be of shape: n x num_choices x hidden_dim
        self.use_cot = use_cot and for_training
        self.make_generative_batches = force_generative_batch or (args.generative_eval and not for_training)
        self.make_seq2seq_batches = args.finetuning_objective == 'seq2seq' and for_training and probing_method=='finetuned'
        self.make_mc_batches = not self.make_generative_batches and not self.make_seq2seq_batches
        assert self.make_generative_batches + self.make_mc_batches + self.make_seq2seq_batches == 1, \
            f"the MCTextDataset should have precisely one of make_generative_batches, make_mc_batches, and make_seq2seq_batches be true"
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return (idx, self.dataframe.iloc[idx])

    def get_effective_num_answers(self):
        return get_max_num_answers(self.dataframe) if not (self.probing_method == 'finetuned' and self.args.finetuning_objective == 'seq2seq') else 1

    def collate_fn(self, items):
        data_idx = []
        item_prompts = []
        answers_list = []
        label_idx_list = []
        num_answers_list = [] # some tasks have variable number of answers per item
        precomputed_hidden_states = []
        repeated_inputs = []
        reasoning_chains = []
        for idx, datapoint in items:
            data_idx.append(idx)
            # use dataframe index to get corresponding hidden_state if we have precomputed hidden_states
            if self.precomputed_hidden_states is not None:
                precomputed_hidden_states.append(self.precomputed_hidden_states[idx])
            # process points
            input_prompt = datapoint.model_input
            answer_choices = datapoint.answer_choices
            label_idx = datapoint.label_idx
            letter_answer_choices = datapoint.letter_labels
            # accumulate input, answer choices, and labels
            item_prompts.append(input_prompt)
            answers_list.append(answer_choices)
            label_idx_list.append(label_idx)
            num_answers_list.append(len(answer_choices))
            repeated_inputs.extend([input_prompt]*len(answer_choices))
            if hasattr(datapoint, 'reasoning_target'):
                reasoning_chains.append(datapoint.reasoning_target)
        # define label strs
        if self.letter_labels:
            flat_answers = [['A', 'B', 'C', 'D'][:len(answer_choices)] for answer_choices in answers_list]
            flat_answers = list(chain(*flat_answers))
            label_strs = [letter_answer_choices[label_idx] for label_idx in label_idx_list]
        else:
            flat_answers = list(chain(*answers_list))
            label_strs = [answers_list[item_num][label_idx] for item_num, label_idx in enumerate(label_idx_list)]
        assert len(repeated_inputs) == len(flat_answers), "Length of input prompts and answers do not match"
        # make batch for passing to compute_probs_from_batch in a train/eval loop
        if isinstance(self.tokenizer, transformers.T5Tokenizer):
            raise NotImplementedError()
        if self.make_generative_batches:
            batch = LM_utils.make_LM_batch(self.tokenizer, item_prompts, label_strs, [0]*len(label_strs), 
                                            padding_side=self.padding_side, max_len=self.max_seq_len,
                                            generative_batch=True, reasoning_chains=None)
        elif self.make_mc_batches:
            batch = LM_utils.make_LM_batch(self.tokenizer, repeated_inputs, flat_answers, label_idx_list, 
                                            padding_side=self.padding_side, max_len=self.max_seq_len,
                                            generative_batch=False)
        elif self.make_seq2seq_batches:
            reasoning_chains_in_labels = self.use_cot and self.for_training and self.probing_method == 'finetuned'
            batch = LM_utils.make_LM_batch(self.tokenizer, item_prompts, label_strs, [0]*len(label_strs), 
                                            padding_side=self.padding_side, max_len=self.max_seq_len,
                                            add_eos_token=True,
                                            generative_batch=False,
                                            reasoning_chains=reasoning_chains if reasoning_chains_in_labels else None)
        if len(precomputed_hidden_states) > 0:
            precomputed_hidden_states = np.stack(precomputed_hidden_states, axis=0)
            batch['precomputed_hidden_states'] = torch.tensor(precomputed_hidden_states)
        # add input strs and metadata to batch
        batch['items'] = items
        batch['prompts'] = item_prompts
        batch['answers_list'] = answers_list
        batch['answer_choices'] = flat_answers if self.make_mc_batches else label_strs
        batch['label_strs'] = label_strs
        batch['data_idx'] = data_idx
        batch['pd_index'] = [datapoint.name for idx, datapoint in items]
        batch['num_answers_list'] = num_answers_list if self.make_mc_batches else [1] * len(items)
        return batch
    
def get_dataloader(args, probing_method, dataframe, dataname, data_source, tokenizer, batch_size, padding_side, max_seq_len, 
                   letter_labels=False, force_generative_batch=False, use_cot=False,
                   shuffle=True, for_training=False, precomputed_hidden_states=None):
    '''
    assembles MCTextDataset dataloader from a provided datasource
    - optionally uses k-shot examples and precomputed hidden states
    - data will be formatted using prompt and prompt_id
    - force_generative_batch used for writing hidden states from the question_end_token, even though we want to train/dev/test as an MC dataset
    '''
    if len(dataframe) == 0:
        class DummyDataLoader(list):
            def __init__(self):
                self.dataset = [] # need this for later checks for len(dataloader.dataset)
            def __len__(self):
                return 0
        return DummyDataLoader() 
    else:
        dataset = MCTextDataset(args, probing_method, dataframe, dataname, data_source, tokenizer, batch_size, padding_side, max_seq_len, letter_labels,
                                for_training=for_training,
                                use_cot=use_cot,
                                force_generative_batch=force_generative_batch, 
                                precomputed_hidden_states=precomputed_hidden_states)
        max_num_answers = get_max_num_answers(dataframe, dataname)
        make_generative_batches = force_generative_batch or (args.generative_eval and not for_training)
        make_seq2seq_batches = args.finetuning_objective == 'seq2seq' and for_training
        reduced_batch_size = batch_size // (1 if (make_generative_batches or make_seq2seq_batches) else max_num_answers)
        dataloader = DataLoader(dataset, shuffle=shuffle, collate_fn=dataset.collate_fn, pin_memory=True, num_workers=0, batch_size=reduced_batch_size)
        return dataloader

def set_for_training(dataloader, for_training):
    if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'args'):
        dataset = dataloader.dataset
        args = dataset.args
        dataset.for_training = for_training
        dataset.make_generative_batches = (args.generative_eval and not for_training)
        dataset.make_seq2seq_batches = args.finetuning_objective == 'seq2seq' and for_training and dataset.probing_method=='finetuned'
        dataset.make_mc_batches = not dataset.make_generative_batches and not dataset.make_seq2seq_batches

def get_load_name(dataname, data_source=None):
    if data_source == 'ai2_arc':
        return ['ai2_arc', dataname]
    elif dataname in ["imdb", "amazon_polarity", "ag_news", "dbpedia_14", "piqa"]:
        return [dataname]
    elif dataname in ["copa", "rte", "boolq"]:
        return ["super_glue", dataname]
    elif dataname in ["qnli"]:
        return ["glue", dataname]
    elif dataname == "story_cloze":
        return ["story_cloze"]
    elif dataname in globals.mmlu_datasets:
        return ["tasksource/mmlu", dataname]
        # return ["cais/mmlu", dataname] # cais upload has duplicated auxiliary_train across tasks
    elif dataname == 'strategy-qa':
        # return ["wics", "strategy-qa"]
        return ["wics/strategy-qa"]
    elif '/' in dataname:
        return dataname.split('/')
    else:
        return [dataname]
    
def get_max_num_answers(dataframe, dataname=None):
    # slightly overcomplicated way of getting the max number of labels for a dataset, before or after column standardization
    if dataname is None: # leave none for when using after column standardization
        num_answers = max([len(datapoint.answer_choices) for _, datapoint in dataframe.iterrows()])
    elif 'ai2_arc' in dataname.lower() or 'ARC' in dataname:
        num_answers = max([len(dataframe.iloc[i].choices['text']) for i in range(len(dataframe))])
    elif dataname in globals.mmlu_datasets:
        num_answers = max([len(dataframe.iloc[i].choices) for i in range(len(dataframe))])
    elif 'mmlu' in dataname:
        num_answers = max([len(dataframe.iloc[i].choices) for i in range(len(dataframe))])
    elif 'strategy-qa' in dataname:
        num_answers = 2
    elif 'gsm8k' in dataname:
        num_answers = 1
    elif dataname in globals.arc_datasets:
        num_answers = max([len(dataframe.iloc[i].choices['text']) for i in range(len(dataframe))])
    return num_answers

def standardize_column_names(dataname, df):
    # standardize str label and label idx (int) for each dataset, based on dataset source
    # also adds standardized text_input col for mmlu
    # returns dataframe with new label_idx and answer_choices columns
    df = df.copy() # silences sliceing warnings
    if dataname in globals.mmlu_datasets or 'mmlu' in dataname:
        labels_col = 'answer'
        choices_col = 'choices'
        df['answer_choices'] = df[choices_col]
        df['input_text'] = df.question.values
        df['label_idx'] = df[labels_col].values
    elif 'strategy-qa' in dataname:
        df['reasoning'] = df.facts.apply(lambda x : ' '.join(x))
        df['answer_choices'] = [['yes', 'no'] for i in range(len(df))]
        df['input_text'] = df.question.values
        df['label_idx'] = df['answer'].apply(lambda x: [True, False].index(x))
    elif 'gsm8k' in dataname:
        df['reasoning'] = df.reasoning
        df['answer_choices'] = [[x] for x in df.answer.values]
        df['input_text'] = df.question.values
        df['label_idx'] = [0 for _ in range(len(df))]
    elif dataname == 'ai2_arc' or 'ARC' in dataname:
        df['input_text'] = df['question']
        df['answer_choices'] = df['choices'].apply(lambda x: x['text'])
        df['label_idx'] = df['answerKey'].apply(lambda x: ['A', 'B', 'C', 'D'].index(x) if not x in ['1', '2', '3', '4'] else ['1', '2', '3', '4'].index(x))
    # add answer_text
    df['answer_text'] = [list(df['answer_choices'].iloc[i])[answer] for i, answer in enumerate(df.label_idx.values)]
    # add A/B/C/D labels
    df['letter_labels'] = [['A', 'B', 'C', 'D'] for _ in range(len(df))]
    return df

def standardize_data_source(dataname_or_data_source):
    # pass in a dataname or named data_source or data subset from globals.py
    all_mmlu = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    if dataname_or_data_source in [None, "NA", ""]:
        return None
    elif hasattr(globals, dataname_or_data_source):
        data_source = dataname_or_data_source
    elif dataname_or_data_source in all_mmlu or "mmlu" in dataname_or_data_source:
        data_source = "mmlu"
    elif "arc" in dataname_or_data_source.lower():
        data_source = 'ai2_arc'
    elif "gsm8k" in dataname_or_data_source.lower():
        data_source = ''
    elif 'strategy-qa' in dataname_or_data_source.lower():
        data_source = 'wics'
    elif 'third_grade_to_college' in dataname_or_data_source:
        data_source = ''
    else:
        raise ValueError(f"Not expecting dataname_or_data_source: {dataname_or_data_source} in data_utils.standardize_data_source (check names against globals.py)")
    return data_source

def randomize_dataframe_labels(dataframe, seed, p=1, randomize_reasoning=False):
    # randomize the labels in a dataframe by randomly picking from the label space. applies to every datapoint with probability p
    # applies only to dataframes processed by standardize_column_names
    randomize_rng = np.random.default_rng(seed)
    label_col_name = 'label_idx'
    orig_col = dataframe[label_col_name].copy()
    for idx in range(len(dataframe)):
        # replace label with probaility p
        if randomize_rng.random() > 1-p:
            datapoint = dataframe.iloc[idx]
            # first option: pick a new choice from the answer set for that problem
            if len(datapoint.answer_choices) > 1:
                choices = datapoint.answer_choices
                new_label = randomize_rng.choice(np.arange(len(choices))) # always an int
                dataframe.loc[dataframe.index[idx], label_col_name] = new_label
            # second option: apply permutation to answers
            else:
                random_ordering = randomize_rng.permutation(len(dataframe))
                dataframe['answer_choices'] = dataframe.answer_choices.iloc[random_ordering].values
    if randomize_reasoning: # this applies a permutation to reasoning...very ad-hoc 'randomization' but retains p(reasoning) marginal of course
        random_ordering = randomize_rng.permutation(len(dataframe))
        dataframe['reasoning'] = dataframe.reasoning.iloc[random_ordering].values
    final_col = dataframe[label_col_name]
    print(f"Shuffled dataframe labels... % match before and after is: {np.mean(orig_col.values==final_col.values):.3f}")
    return dataframe
    
def write_datasets_with_known_hardness(args, request, data_dir, seed, make_hardness_split=False, overwrite_existing=False, verbose=False):
    '''
    Loads datasets with metadata indicating hardness, e.g. grade level for ARC
    args:
        make_hardness_split: still split data into 1k hardness points for model-based hardness estimation, with remaining points for training probes
    '''
    # locals
    data_rows = [] # records metadata
    data_source = standardize_data_source(request)
    short_request_name = request.split('/')[-1]
    hardness_save_name = f"{short_request_name}_hardness-data_sd{seed}.json"
    probing_save_name = f"{short_request_name}_probing-data_sd{seed}.json"
    save_path = os.path.join(data_dir, probing_save_name)
    already_written = os.path.exists(save_path)
    if already_written and not overwrite_existing:
        return
    # first, mmlu case. we merge high_school_x and college_x splits and give them 0/1 hardness scores
    if 'mmlu' in request:
        assert not make_hardness_split, "Not enough MMLU data for making a separate hardness split"
        all_subjects = ['computer_science', 'physics', 'mathematics', 'chemistry', 'biology']
        all_combined_datasets = []
        for subject in all_subjects:
            data_rng = np.random.default_rng(seed)
            # check written
            save_name = f"mmlu_{subject}_probing-data_sd{seed}.json"
            save_path = os.path.join(data_dir, save_name)
            if os.path.exists(save_path) and not overwrite_existing:
                continue
            # combine high_school and college_level
            combined_datasets = []
            for level in ['high_school', 'college']:
                dataname = f"{level}_{subject}"
                load_name = get_load_name(dataname, data_source)
                raw_dataset = datasets.load_dataset(*load_name, cache_dir=data_dir)
                if verbose:
                    print(dataname)
                    for key in raw_dataset.keys():
                        print(f" {key}: {raw_dataset[key].shape}")
                gather_splits = ["validation", "test"]
                all_data = pd.concat([raw_dataset[split].to_pandas() for split in gather_splits])
                all_data['human_hardness'] = 0 if level == 'high_school' else 1
                combined_datasets.append(all_data)
            combined_datasets = pd.concat(combined_datasets)
            combined_datasets['subject'] = subject
            all_combined_datasets.append(combined_datasets)
            # save data
            combined_datasets.to_json(save_path, orient='records')
            row = {
                "source": data_source,
                'name': subject,
                "sample_size": len(combined_datasets),
                "n_easy": sum(combined_datasets['human_hardness'] == 0),
                "n_hard": sum(combined_datasets['human_hardness'] == 1),
                'n_answers': get_max_num_answers(combined_datasets, dataname), # dataname will be the college_x, but label cols should be same between them
            }
            data_rows.append(row)
        all_data = pd.concat(all_combined_datasets)
        all_data = standardize_column_names('mmlu_STEM-5', all_data)
        if verbose:
            print(f"Number points total: {len(all_data)}")
    # for arc, we combine data from "easy" and "challenge" splits, then use our metadata for assigning difficulty levle
    elif 'arc' in request:
        data_rng = np.random.default_rng(seed)
        combined_datasets = []
        for dataname in ['ARC-Easy', 'ARC-Challenge']:
            # load data and print split sizes
            load_name = get_load_name(dataname, data_source)
            raw_dataset = datasets.load_dataset(*load_name, cache_dir=data_dir, ignore_verifications=True)
            if verbose:
                print(dataname)
                for key in raw_dataset.keys():
                    print(f" {key}: {raw_dataset[key].shape}")
            # gather all the labeled data
            gather_splits = ['train', 'validation', 'test']
            all_data = pd.concat([raw_dataset[split].to_pandas() for split in gather_splits])
            combined_datasets.append(all_data)
        # combine data
        combined_datasets = pd.concat(combined_datasets)
        # join with our metadata
        metadata_path = os.path.join('data', 'arc-challenge-easy-annotations.json')
        with open(metadata_path, 'r') as f:
            arc_metadata = json.load(f)
            records = []
            for key, value in arc_metadata.items():
                source = key[:5]
                id = key[6:]
                record = {'source': source, 'id': id}
                record.update(value)
                records.append(record)
            arc_metadata = pd.DataFrame(records)
            def extract_grade_level(x):
                grade_parts = x.split()
                if grade_parts[-1].isdigit():
                    return int(grade_parts[-1])
                return None
            def extract_difficulty(x):
                levels = ['Low', 'Medium', 'High']
                if x in levels:
                    return levels.index(x) + 1
                return None
            def extract_bloom(x):
                levels = ['Remembering', 'Understanding', 'Applying', 'Analyzing', 'Evaluating']
                if x in levels:
                    return levels.index(x) + 1
                return None
            def extract_depth_of_knowledge(x):
                levels = ['I', 'II', 'III']
                if x in levels:
                    return levels.index(x) + 1                    
                return None
            arc_metadata['human_grade'] = arc_metadata['grade'].apply(extract_grade_level)
            arc_metadata['human_difficulty'] = arc_metadata['difficulty'].apply(extract_difficulty)
            arc_metadata['human_bloom'] = arc_metadata['bloom'].apply(extract_bloom)
            arc_metadata['human_depth_of_knowledge'] = arc_metadata['depth_of_knowledge'].apply(extract_depth_of_knowledge)
            # merge difficulty variable to ids
            combined_datasets = combined_datasets.merge(arc_metadata[['id', 'human_grade', 'human_difficulty', 'human_bloom', 'human_depth_of_knowledge']], left_on='id', right_on='id')
        # drop missing hardness annotations
        if verbose:
            print(f"Number points total: {len(combined_datasets)}")
            print(f"Number points with hardness metadata: {len(combined_datasets.dropna())}")
            print(f"Number points with unique ids: {len(combined_datasets.id.unique())}")
        all_data = combined_datasets.dropna()
        # standardize columns
        all_data = standardize_column_names('ai2_arc', all_data)
        # print final hardness distr
        if verbose:
            if 'human' in args.hardness_var_name:
                print("Final hardness distr: ")
                print(all_data.groupby(args.hardness_var_name)['id'].count())
    elif 'strategy-qa' in request:
        if request == 'strategy-qa-dev':
            file_path = os.path.join(data_dir, 'strategy-qa-dev.json')
            samples = []
            with open(file_path, "r", encoding="utf-8-sig") as f:
                json_inputs = json.load(f)
                for i, json_input in enumerate(json_inputs):
                    samples.append({
                        "index": i,
                        "qid": json_input["qid"],
                        "question": json_input["question"],
                        "answer": json_input["answer"],
                        "gold_explanation": " ".join(json_input["facts"]),
                        "decomposition": "",
                        "facts": json_input['facts']
                    })
            all_data = pd.DataFrame.from_records(samples)
        else:
            raw_dataset = datasets.load_dataset("wics/strategy-qa", cache_dir=data_dir)
            # gather all the labeled data
            gather_splits = ['test'] # this is the train split of 2290 points from the paper
            all_data = pd.concat([raw_dataset[split].to_pandas() for split in gather_splits])
            all_data['gold_explanation'] = all_data.facts.apply(lambda x: " ".join(x))
        # make hardness variable
        def get_num_steps(steps):
            return len(steps)
        def get_num_words(gold_explanation):
            return len(gold_explanation.split())
        all_data['num_steps'] = all_data.decomposition.apply(get_num_steps)
        all_data['reasoning_num_words'] = all_data.gold_explanation.apply(get_num_words)
        # standardize cols
        all_data = standardize_column_names('strategy-qa', all_data)
        if verbose:
            print("Final hardness distr: ")
            print(all_data.groupby('num_steps')['qid'].count())
    elif 'gsm8k' in request.lower():
        assert request in ['gsm8k_main', 'gsm8k_socratic', 'gsm8k_main_test']
        split_name = request.split('_')[1]
        # load data and print split sizes
        raw_dataset = datasets.load_dataset('gsm8k', split_name, cache_dir=data_dir)
        if request == 'gsm8k_main_test':
            gather_splits = ['test']
        else:
            gather_splits = ['train', 'test']
        all_data = pd.concat([raw_dataset[split].to_pandas() for split in gather_splits]).reset_index()
        data_rng = np.random.default_rng(seed)
        print("Extracting reasoning steps and answer from gsm8k...")
        all_data['reasoning'] = None
        all_data['num_steps'] = None
        all_data['reasoning_num_words'] = None
        for i in range(len(all_data)):
            full_answer = all_data.iloc[i].answer
            reasoning, answer = full_answer.split('####')
            reasoning = reasoning.strip('\n')
            sentences = reasoning.split('\n')
            num_steps = min(len(sentences), 10) # cap at 10...there are like 10 datapoints with >= 10 steps
            num_words = len(reasoning.replace('\n', " ").split())
            all_data.loc[i, 'reasoning'] = reasoning
            all_data.loc[i, 'answer'] = answer.strip()
            all_data.loc[i, 'num_steps'] = num_steps
            all_data.loc[i, 'reasoning_num_words'] = num_words
        all_data = standardize_column_names('gsm8k', all_data)
        
        # print final hardness distr
        if verbose:
            print("Final hardness distr: ")
            print(all_data.groupby('num_steps')['input_text'].count())
    else:
        raise NotImplementedError(f"Didn't expect data request {request} for data_utils.write_datasets_with_known_hardness")

    # add more hardness variables...
    all_data = all_data.copy()
    all_data['question_num_words'] = all_data.input_text.apply(lambda x : len(x.split())).copy()
    all_data['answer_num_words'] = all_data.answer_text.apply(lambda x : len(x.split())).copy()
    all_data['answer_num_chars'] = all_data.answer_text.apply(lambda x : len(x)).copy()

    # save hardness/probing splits
    if not make_hardness_split:
        save_path = os.path.join(data_dir, probing_save_name)
        all_data = all_data.drop_duplicates(subset='id')
        all_data.to_json(save_path, orient='records')
    else:
        all_idx = np.arange(len(all_data))
        all_data = all_data.drop_duplicates(subset='id')
        hardness_idx = data_rng.choice(all_idx, size=1000, replace=False)
        hardness_split = all_data.iloc[hardness_idx,:]
        probing_idx = np.setdiff1d(all_idx, hardness_idx)
        probing_split = all_data.iloc[probing_idx,:]
        save_path = os.path.join(data_dir, hardness_save_name)
        hardness_split.to_json(save_path, orient='records')
        save_path = os.path.join(data_dir, probing_save_name)
        probing_split.to_json(save_path, orient='records')
    # add metadata to data_info df
    row = {
        "source": data_source,
        'name': request,
        'n_hardness_points': len(hardness_split) if make_hardness_split else None,
        "sample_size": len(all_data) if not make_hardness_split else len(probing_split),
        'n_answers': get_max_num_answers(all_data), # dataname will be last dataname from for loop, but label cols should be same between them
    }
    data_rows.append(row)
    data_info = pd.DataFrame(data_rows)
    # write metadata if there were gathered datasets
    if verbose:
        print(data_info)
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    data_info.to_csv(f'outputs/data_info_{data_source}.csv', index=False)
    return
    
def write_datasets(request, sample_size, data_dir, seed, make_hardness_split=True, min_hardness_points=None, max_hardness_points=None, overwrite_existing=False, verbose=False):
    '''
    Loads datasets from hugging face datasets library, splits into our data splits and writes to file
    - Saves points for probing experiments, from all labeled data
    - Saves up to max_hardness_points remaining points from labeled data for training a hardness model
    - Filters to datasets with at least sample_size labeled points for probing, and optinally min_hardness_points leftover for hardness estimation
    - This function is not used for loading data with metadata for hardness splits. See write_datasets_with_known_hardness
    '''
    data_source = standardize_data_source(request)
    all_mmlu = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    # use pre-specified list of datasets, or load something else
    if hasattr(globals, request):
        datanames = getattr(globals, request)
    else:
        datanames = [request]

    # will save metadata for tasks
    data_info = pd.DataFrame(columns=["source", "name", "sample_size", "n_hardness_points"])
    # skip some tasks that throw formatting errors
    skip_tasks = ["simple_arithmetic_json_multiple_choice", "simple_arithmetic_multiple_targets_json"]
    datanames = all_mmlu if datanames == "mmlu" else datanames
    for dataname in datanames:
        short_dataname = dataname.split('/')[-1]
        data_rng = np.random.default_rng(seed)
        # check written
        save_name = f"{short_dataname}_probing-data_sd{seed}.json"
        save_path = os.path.join(data_dir, save_name)
        if os.path.exists(save_path) and not overwrite_existing:
            continue

        if dataname in skip_tasks:
            continue

        load_name = get_load_name(dataname)
        raw_dataset = datasets.load_dataset(*load_name, cache_dir=data_dir)
        if verbose:
            print(dataname)
            for key in raw_dataset.keys():
                print(f" {key}: {raw_dataset[key].shape}")

        # gather all the labeled data
        if dataname in ["imdb", "amazon_polarity", "ag_news", "dbpedia_14"]:
            gather_splits = ["train", "test"]
        elif dataname in ["copa", "rte", "boolq", "piqa", "qnli"]:
            gather_splits = ["train", "validation"] 
        elif dataname in ["story_cloze"]:
            gather_splits = ["validation"]
        elif load_name[0] == "tasksource/mmlu":
            gather_splits = ["validation", "test"] 
        elif load_name[1] == 'ARC-Challenge':
            gather_splits = ["test"]
        all_data = pd.concat([raw_dataset[split].to_pandas() for split in gather_splits])
        
        # sample up to 1000 points for making probing datasets
        have_points = len(all_data)
        if not have_points >= sample_size:
            print(f"Don't have at least {sample_size} labeled points in the {dataname} dataset")
            continue
        sample_idx = data_rng.choice(np.arange(have_points), size=sample_size, replace=False)
        probing_data = all_data.iloc[sample_idx]

        # now take up to max_hardness_points of the remaining points for estimating hardness
        if make_hardness_split:
            remaining_idx = np.setdiff1d(np.arange(have_points), sample_idx)
            remaining_data = all_data.iloc[remaining_idx]
            if min_hardness_points is not None:
                if not len(remaining_data) >= min_hardness_points:
                    print(f"Not enough hardness data. Don't have at least {min_hardness_points} leftover points in the {dataname} dataset")
                    continue
            try_to_get = max_hardness_points if max_hardness_points is not None else len(remaining_data)
            n_sample = min(try_to_get, len(remaining_data))
            sample_idx = data_rng.choice(np.arange(len(remaining_data)), size=n_sample, replace=False)
            hardness_data = remaining_data.iloc[sample_idx]
            if len(hardness_data) < try_to_get:
                print(f"Could only get {len(hardness_data)} points from {dataname} for hardness model after reserving probing points")
        else:
            hardness_data = all_data.iloc[:0]

        # save data
        save_name = f"{short_dataname}_probing-data_sd{seed}.json"
        save_path = os.path.join(data_dir, save_name)
        probing_data.to_json(save_path, orient='records')
        save_name = f"{short_dataname}_hardness-data_sd{seed}.json"
        save_path = os.path.join(data_dir, save_name)
        hardness_data.to_json(save_path, orient='records')

        # add metadata to data_info df
        row = {
            "source": data_source,
            'name': short_dataname,
            "sample_size": sample_size,
            'n_answers': get_max_num_answers(all_data, dataname),
            'n_hardness_points': int(len(hardness_data))
        }
        data_info = data_info.append(row, ignore_index=True)

    if len(data_info) >= 1 and verbose:
        print(f"Have {len(data_info)} tasks with {sample_size} labeled point and at least {min_hardness_points} points for hardness estimation: ")
        print(data_info)
    
    # write metadata if there were gathered datasets
    if len(data_info) > 1:
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        data_info.to_csv(f'outputs/data_info_{data_source}.csv', index=False)

def load_datasets(args, datanames, data_dir, seed):
    '''
    Load datasets written by write_datasets
    args:
        datanames without data_sources, e.g. dataname.split('/')[-1]
    returns
        nested dict with hardness and probing data for each dataset
    '''
    datasets = {}
    for dataname in datanames:
        short_dataname = dataname.split('/')[-1]
        # load dataset hardness splits
        save_name = f"{short_dataname}_hardness-data_sd{seed}.json"
        save_path = os.path.join(data_dir, save_name)
        if os.path.exists(save_path):
            hardness_data = pd.read_json(save_path, orient='records')
        else:
            hardness_data = None
        # load probing split
        save_name = f"{short_dataname}_probing-data_sd{seed}.json"
        save_path = os.path.join(data_dir, save_name)
        probing_data = pd.read_json(save_path, orient='records')
        datasets[short_dataname] = {
            'hardness_data': hardness_data,
            'probing_data': probing_data,
        }
    return datasets

def stratified_test_sampling(args, datasets, seed, precomputed_hidden_states, 
                   hardness_var_name,
                   train_on,
                   n_train,
                   n_test,
                   min_test_size,
                   human_easy_cutoff=None, human_hard_cutoff=None,
                   dataname_to_hardness_config_dict=None):
    '''
    This function is for stratified sampling of test sets when we specify a particular number of test points
    - the bottleneck for this kind of testing will be when training on easy data (And trying to maintain easy test data), or training on hard (and trying to maintain hard test data)
    - we resolve this by:
    - 1. checking that we have at least n_train+min_test_data of the bottleneck data condition (easy or hard)
    - 2. fillling the train data first
    - 3. fill the test data, preferring test data from the bottlneck condition first
    args:
      n_train: should be passed as max(superivision_n) from main.py. The max amount of train data we want. Require to get this
      n_test: -1 means gather all the available data for testing. o/w, try to get the requested amount (but might come up short)
      dataname_to_hardness_config_dict: used for passing different hardness_var_name and cutoffs for different datasets in datasets
    '''
    assert args.no_dev
    assert args.test_on == 'all'
    assert args.stratify_hardness
    assert args.all_data_to_test
    split_datasets = {}
    for dataname, data_dict in datasets.items():
        # get hardness_var_name, human_easy_cutoff, and human_hard_cutoff from the dataname_to_hardness_config_dict if provided
        if dataname_to_hardness_config_dict:
            hardness_var_name, human_easy_cutoff, human_hard_cutoff = dataname_to_hardness_config_dict[dataname]
        # create rng per dataset, so splits do not vary based on number of used datasets
        split_rng = np.random.default_rng(seed)
        # unpack precomputed_hidden_states for particular dataset
        if precomputed_hidden_states is not None:
            hidden_states_dict = precomputed_hidden_states[dataname]
            split_hidden_states = {} # nested dict of prompt_idx: encoder/decoder_hidden_states: train/dev/test: hidden_states
        probing_data = data_dict['probing_data']
        n = len(probing_data)
        # first get eligible easy/hard data idx
        use_automatic_quantiles = 'model' in hardness_var_name or 'prob' in hardness_var_name or 'words' in hardness_var_name or 'chars' in hardness_var_name
        if use_automatic_quantiles:
            hardness_col_name = utils.get_hardness_col_name(hardness_var_name, args.model, model_avg='avg' in args.hardness_var_name)
            print("Getting hardness scores from: ", hardness_col_name)
            hardness_scores = probing_data[hardness_col_name]
            # easy/hard will be 30% easiest/hardest in data according to hardness_scores
            easy_max, hardness_min = np.quantile(hardness_scores, [.3, .7])
            all_easy_idx = np.argwhere(hardness_scores.values <= easy_max).squeeze()
            all_hard_idx = np.argwhere(hardness_scores.values >= hardness_min).squeeze()
            all_medium_idx = np.setdiff1d(np.arange(n), all_hard_idx)
            all_medium_idx = np.setdiff1d(all_medium_idx, all_easy_idx)
        else:
            col_name = hardness_var_name
            hardness_scores = probing_data[col_name]
            all_easy_idx = np.argwhere(hardness_scores.values <= human_easy_cutoff).squeeze()
            all_hard_idx = np.argwhere(hardness_scores.values >= human_hard_cutoff).squeeze()
            all_medium_idx = np.setdiff1d(np.arange(n), all_hard_idx)
            all_medium_idx = np.setdiff1d(all_medium_idx, all_easy_idx)
        # first, get the train_idx
        n_hard = len(all_hard_idx)
        n_medium = len(all_medium_idx)
        n_easy = len(all_easy_idx)
        all_idx = np.arange(n)
        if train_on == 'easy':
            eligible_train_and_dev_idx = np.random.choice(all_easy_idx, size=n_train, replace=False)
            assert n_easy >= n_train + min_test_size
        elif train_on == 'medium':
            eligible_train_and_dev_idx = np.random.choice(all_medium_idx, size=n_train, replace=False)
            assert n_medium >= n_train + min_test_size
        elif train_on == 'hard':
            eligible_train_and_dev_idx = np.random.choice(all_hard_idx, size=n_train, replace=False)
            assert n_hard >= n_train + min_test_size
        elif train_on == 'easy_and_hard':
            sample_size = int(np.ceil(n_train/2))
            eligible_train_and_dev_idx = np.concatenate([
                split_rng.choice(all_easy_idx, size=sample_size, replace=False),
                split_rng.choice(all_hard_idx, size=sample_size, replace=False)
            ])    
            assert n_easy+n_hard >= n_train + min_test_size
        elif train_on == 'all':
            eligible_train_and_dev_idx = np.random.choice(all_idx, size=n_train, replace=False)
        # now fill the test data, stratifying by hardness if necessary. begin by setting train_idx/dev_idx and eligible_test_idx
        n_train = len(eligible_train_and_dev_idx)
        n_dev = 0
        # get train/dev idx/test
        train_idx = split_rng.choice(eligible_train_and_dev_idx, size=n_train, replace=False)
        remaining_idx = np.setdiff1d(eligible_train_and_dev_idx, train_idx)
        dev_idx = split_rng.choice(remaining_idx, size=n_dev, replace=False)
        eligible_test_idx = np.setdiff1d(all_idx, train_idx)
        eligible_test_idx = np.setdiff1d(eligible_test_idx, dev_idx)
        # case 1: include all test data
        if n_test <= 0:
            test_idx = eligible_test_idx
        # case 2: get requested amount of test data. TRY TO STRATIFY BY HARDNESS
        elif n_test > 0:
            eligible_test_idx_hard = np.intersect1d(eligible_test_idx, all_hard_idx)
            eligible_test_idx_medium = np.intersect1d(eligible_test_idx, all_medium_idx)
            eligible_test_idx_easy = np.intersect1d(eligible_test_idx, all_easy_idx)
            eligible_data_list = [eligible_test_idx_easy, eligible_test_idx_medium, eligible_test_idx_hard]
            available_data = [len(x) for x in eligible_data_list]
            bottleneck_idx = np.argmin(available_data).item()
            bottleneck_size = available_data[bottleneck_idx]
            # if we dont have enough of each kind of data to equally fill, then first fill with the least represented group, then mix in the other two
            if 3*bottleneck_size < n_test:
                test_idx = eligible_data_list[bottleneck_idx]
                need_n_more = n_test - len(test_idx)
                get_n_per_remaining_hardness = int(np.ceil(need_n_more / 2))
                non_bottleneck_idx = np.setdiff1d(np.arange(3), bottleneck_idx)
                remaining_data_list = [eligible_data_list[idx] for idx in non_bottleneck_idx]
                add_test_idx = np.concatenate([
                    split_rng.choice(remaining_data, size=get_n_per_remaining_hardness, replace=False) 
                    if len(remaining_data) >= get_n_per_remaining_hardness else remaining_data
                    for remaining_data in remaining_data_list
                ])
                test_idx = np.concatenate([test_idx, add_test_idx])
                split_rng.shuffle(test_idx) # shuffle since we might include an entire chunk of remaining_data above if there's also not enough to fill evenly
            # otherwise, sample args.n_test/3 points from each group
            else:
                get_n_per_hardness = int(np.ceil(n_test / 3))
                test_idx = np.concatenate([
                    split_rng.choice(eligible_data, size=get_n_per_hardness, replace=False) 
                    for eligible_data in eligible_data_list
                ])
        probing_train = probing_data.iloc[train_idx]
        probing_dev = probing_data.iloc[dev_idx]
        probing_test = probing_data.iloc[test_idx]
        split_datasets[dataname] = {
            'probing_data': {
                    'train': probing_train,
                    'dev': probing_dev,
                    'test': probing_test,
                },
            }
        if precomputed_hidden_states is not None:
            for prompt_idx, hidden_states in hidden_states_dict.items():
                probing_hidden_states = hidden_states['probing_data']
                split_hidden_states[prompt_idx] = {
                    'train': probing_hidden_states[train_idx],
                    'dev': probing_hidden_states[dev_idx],
                    'test': probing_hidden_states[test_idx]
                }
            split_datasets[dataname][f"probing_states"] = split_hidden_states
    return split_datasets

def split_datasets(args, datasets, seed, data_type, precomputed_hidden_states=None, 
                   stratify_hardness=False, hardness_var_name=None,
                   train_on='easy', test_on='hard', 
                   human_easy_cutoff=None, human_hard_cutoff=None, human_hardness_exact=False,
                   standardize_sample_sizes=True, 
                   use_extra_easy_data=False,
                   max_n_train=None, min_test_size=None,
                   dataname_to_hardness_config_dict=None,
                   verbose=False):
    '''
    split datasets dict for hardness and probing models
    - probing splits will be 70/10/20 with argument for stratifying by hardness score
    - hardness spits will be 90/10 train/dev, unless this results in fewer than 100 dev points, then get 100 dev and leftover become train
    args:
        data_type is 'hardness' or 'probing'
        datasets should be dataname: {data_type}_data dict
        precomputed_hidden_states: will be indexed to match with the train/dev/test splits of pd dfs, also returned in data dict
        stratify_hardness: this makes train/dev/test splits BASED on hardness
        train_on and test_on should be easy/hard/all
        train_on: 'easy'/'hard'/'all' str, defined as 30% bottom/top/all data according to model-based hardness OR based on human_x_cutoff args if not model_based_hardness (i.e. using human hardness)
        train_on: 'easy'/'hard'/'all' str, defined as 30% bottom/top/all data according to model-based hardness OR based on human_x_cutoff args if not model_based_hardness (i.e. using human hardness)
        human_easy_cutoff: easy points have human hardness scores <= this threshold -- only applies when not model_based_hardness
        human_hard_cutoff: hard points have human hardness scores >= this threshold -- only applies when not model_based_hardness
        human_hardness_exact: select train/test points based on exact match of human hardness score and human_easy_cutoff/human_hard_cutoff. otherwise, threshold s.t. easy points are <=threshold and hard points are >=threshold
        standardize_sample_sizes: train/dev/test splits are bottlenecked when train_on==test_on, so limit data to match our minimum train/dev/test sizes regardless of train_on/test_on values 
                                 (only applicable with stratify_hardness)
        use_extra_easy_data: even when doing standardize_sample_sizes, this will add in any available easy data to the training set, with the view that it is "free" training data
    returns
        nested dict of dataname: data_type: split_name: data
    '''
    split_datasets = {}
    for dataname, data_dict in datasets.items():
        if dataname_to_hardness_config_dict:
            hardness_var_name, human_easy_cutoff, human_hard_cutoff = dataname_to_hardness_config_dict[dataname]
        # create rng per dataset, so splits do not vary based on number of used datasets
        split_rng = np.random.default_rng(seed)
        # unpack precomputed_hidden_states for particular dataset
        if precomputed_hidden_states is not None:
            hidden_states_dict = precomputed_hidden_states[dataname]
            split_hidden_states = {} # nested dict of prompt_idx: encoder/decoder_hidden_states: train/dev/test: hidden_states
        # split hardness data
        if data_type == 'hardness':
            hardness_data = data_dict['hardness_data']
            if hardness_data is not None:
                n = len(hardness_data)
                n_dev = int(max(np.ceil(.1 * n), 100)) if not args.debug else 32
                n_train = n - n_dev
                train_idx = split_rng.choice(np.arange(n), size=n_train, replace=False)
                dev_idx = np.setdiff1d(np.arange(n), train_idx)
                hardness_train = hardness_data.iloc[train_idx]
                hardness_dev = hardness_data.iloc[dev_idx]
                split_datasets[dataname] = {
                    f'hardness_data': {
                        'train': hardness_train,
                        'dev': hardness_dev,
                    },
                }
            else:
                split_datasets[dataname] = {
                    f'hardness_data': {
                        'train': pd.DataFrame(),
                        'dev': pd.DataFrame(),
                    },
                }
            # add hidden states
            if precomputed_hidden_states is not None:
                for prompt_idx, hidden_states in hidden_states_dict.items():
                    if 'hardness_data' in hidden_states:
                        hardness_hidden_states = hidden_states['hardness_data']
                        split_hidden_states[prompt_idx] = {
                            'train': hardness_hidden_states[train_idx],
                            'dev': hardness_hidden_states[dev_idx],
                        }
                    else:
                        split_hidden_states = {}
                split_datasets[dataname][f"{data_type}_states"] = split_hidden_states
        # split probing data
        elif data_type == 'probing':
            probing_data = data_dict['probing_data']
            n = len(probing_data)
            if not stratify_hardness:
                # special cases first
                if args.all_data_to_test:
                    n_train = max_n_train
                    n_test = n - n_train
                    n_dev = 0
                else:
                    n_dev = int(np.ceil(.1 * n))
                    n_test = max(int(np.ceil(.2 * n)), min_test_size)
                    n_train = n - n_dev - n_test
                random_idx = split_rng.permutation(n)
                train_idx = random_idx[:n_train]
                dev_idx = random_idx[n_train:n_train+n_dev]
                test_idx = random_idx[n_train+n_dev:]
            else:
                # first get eligible easy/hard data idx
                use_automatic_quantiles = 'model' in hardness_var_name or 'prob' in hardness_var_name or 'words' in hardness_var_name or 'chars' in hardness_var_name
                if use_automatic_quantiles:
                    hardness_col_name = utils.get_hardness_col_name(hardness_var_name, args.model, model_avg='avg' in args.hardness_var_name)
                    print("Getting hardness scores from: ", hardness_col_name)
                    hardness_scores = probing_data[hardness_col_name]
                    # easy/hard will be 30% easiest/hardest in data according to hardness_scores
                    easy_max, hardness_min = np.quantile(hardness_scores, [.3, .7])
                    all_easy_idx = np.argwhere(hardness_scores.values <= easy_max).squeeze()
                    all_hard_idx = np.argwhere(hardness_scores.values >= hardness_min).squeeze()
                    all_medium_idx = np.setdiff1d(np.arange(n), all_hard_idx)
                    all_medium_idx = np.setdiff1d(all_medium_idx, all_easy_idx)
                else:
                    col_name = hardness_var_name
                    hardness_scores = probing_data[col_name]
                    # scenario 1: x_cutoff args specify exact hardness levels
                    if human_hardness_exact:
                        all_easy_idx = np.argwhere(hardness_scores.values == human_easy_cutoff).squeeze()
                        all_medium_idx = np.arange([])
                        all_hard_idx = np.argwhere(hardness_scores.values == human_hardness_exact).squeeze()
                    # scenario 2: threshold based on human_x_cutoff values
                    else:
                        all_easy_idx = np.argwhere(hardness_scores.values <= human_easy_cutoff).squeeze()
                        all_hard_idx = np.argwhere(hardness_scores.values >= human_hard_cutoff).squeeze()
                        all_medium_idx = np.setdiff1d(np.arange(n), all_hard_idx)
                        all_medium_idx = np.setdiff1d(all_medium_idx, all_easy_idx)
                # determine test sizes, based on whether standardizing the data size bottleneck according to train_on/test_on conditions
                n_hard = len(all_hard_idx)
                n_medium = len(all_medium_idx)
                n_easy = len(all_easy_idx)
                # first, get the test_idx and determine eligible train/dev_idx
                # if we're standardizing the sample sizes across conditions, we fix them like so
                all_idx = np.arange(n)
                if standardize_sample_sizes:
                    bottleneck_is_hard_data = n_hard < n_easy
                    min_test_data = max(int(.25*n_hard), min_test_size) if bottleneck_is_hard_data else max(int(.25*n_easy), min_test_size)
                    n_train_and_dev_data = n_hard - min_test_data if bottleneck_is_hard_data else n_easy - min_test_data
                    n_test = int(min_test_data)
                    # now determine test idx
                    if test_on == 'all':
                        test_idx = split_rng.choice(all_idx, size=n_test, replace=False)
                    if test_on == 'hard':
                        test_idx = split_rng.choice(all_hard_idx, size=n_test, replace=False)
                    if test_on == 'medium':
                        test_idx = split_rng.choice(all_medium_idx, size=n_test, replace=False)
                    if test_on == 'easy':
                        test_idx = split_rng.choice(all_easy_idx, size=n_test, replace=False)
                    if test_on == 'easy_and_hard':
                        # now evenly mix easy and hard data together for training
                        half_n_test = int(np.floor(min_test_data/2))
                        test_idx = np.concatenate([
                            split_rng.choice(all_easy_idx, size=half_n_test, replace=False),
                            split_rng.choice(all_hard_idx, size=half_n_test, replace=False)
                        ])
                    # get eligible train/dev idx, and reset the remaining available data idx
                    eligible_train_and_dev_idx = np.setdiff1d(all_idx, test_idx)
                    eligible_easy_idx = np.setdiff1d(all_easy_idx, test_idx)
                    eligible_medium_idx = np.setdiff1d(all_medium_idx, test_idx)
                    eligible_hard_idx = np.setdiff1d(all_hard_idx, test_idx)
                    if train_on == 'easy':
                        eligible_train_and_dev_idx = np.intersect1d(eligible_train_and_dev_idx, eligible_easy_idx)
                    if train_on == 'medium':
                        eligible_train_and_dev_idx = np.intersect1d(eligible_train_and_dev_idx, eligible_medium_idx)
                    if train_on == 'hard':
                        eligible_train_and_dev_idx = np.intersect1d(eligible_train_and_dev_idx, eligible_hard_idx)
                    if train_on == 'easy_and_hard':
                        # now evenly mix easy and hard data together for training
                        n_easy = len(eligible_easy_idx)
                        n_hard = len(eligible_hard_idx)
                        bottleneck_is_hard_data = n_hard < n_easy
                        bottleneck_size = n_hard if bottleneck_is_hard_data else n_easy
                        eligible_train_and_dev_idx = np.concatenate([
                            split_rng.choice(eligible_easy_idx, size=bottleneck_size, replace=False),
                            split_rng.choice(eligible_hard_idx, size=bottleneck_size, replace=False)
                        ])
                    if train_on == 'all':
                        assert test_on == 'all', "Don't train on all if testing on a subset, because it biases the remaining distribution of 'all' points"
                    # ADD IN AVAILABLE EASY DATA IF REQUESTED
                    if use_extra_easy_data:
                        leftover_easy_idx = np.setdiff1d(eligible_easy_idx, eligible_train_and_dev_idx) # easy_idx already stripped of test_idx
                        eligible_train_and_dev_idx = np.concatenate([eligible_train_and_dev_idx, leftover_easy_idx])
                # otherwise, we will allocate sample sizes by each possible train/test combination
                # if we draw train from the test source, allocate to 70/10/20 splits
                # when doing something like train=easy_and_hard and test=hard, want train to be 50/50 easy+hard
                elif train_on == 'easy' and test_on == 'hard':
                    n_test = int(max(n_hard, min_test_size))
                    test_idx = split_rng.choice(all_hard_idx, size=n_test, replace=False)
                    eligible_train_and_dev_idx = all_easy_idx
                    n_train_and_dev_data = len(eligible_train_and_dev_idx)
                elif train_on == 'hard' and test_on == 'hard':
                    n_test = int(max(.2*n_hard, min_test_size))
                    test_idx = split_rng.choice(all_hard_idx, size=n_test, replace=False)
                    eligible_train_and_dev_idx = np.setdiff1d(all_hard_idx, test_idx)
                    n_train_and_dev_data = len(eligible_train_and_dev_idx)
                elif train_on == 'easy_and_hard' and test_on == 'hard':
                    # try to leave half the hard data for training
                    n_test = int(max(.5*n_hard, min_test_size))
                    test_idx = split_rng.choice(all_hard_idx, size=n_test, replace=False)
                    eligible_hard_idx = np.setdiff1d(all_hard_idx, test_idx)
                    # now evenly mix easy and hard data together for training
                    n_hard_train = n_hard - n_test
                    eligible_hard_idx = split_rng.choice(eligible_hard_idx, size=n_hard_train, replace=False)
                    bottleneck_is_hard_data = n_hard_train < n_easy
                    bottleneck_size = n_hard_train if bottleneck_is_hard_data else n_easy
                    eligible_train_and_dev_idx = np.concatenate([
                        split_rng.choice(all_easy_idx, size=bottleneck_size, replace=False),
                        split_rng.choice(eligible_hard_idx, size=bottleneck_size, replace=False)
                    ])
                    n_train_and_dev_data = len(eligible_train_and_dev_idx)
                elif train_on == 'easy' and test_on == 'easy':
                    n_test = int(max(.2*n_easy, min_test_size))
                    test_idx = split_rng.choice(all_easy_idx, size=n_test, replace=False)
                    eligible_train_and_dev_idx = np.setdiff1d(all_easy_idx, test_idx)
                    n_train_and_dev_data = len(eligible_train_and_dev_idx)
                elif train_on == 'hard' and test_on == 'easy':
                    n_test = int(max(n_easy, min_test_size))
                    test_idx = split_rng.choice(all_easy_idx, size=n_test, replace=False)
                    eligible_train_and_dev_idx = all_hard_idx
                elif train_on == 'easy_and_hard' and test_on == 'easy':
                    # try to leave half the easy data for training
                    n_test = int(max(.5*n_easy, min_test_size))
                    test_idx = split_rng.choice(all_easy_idx, size=n_test, replace=False)
                    eligible_easy_idx = np.setdiff1d(all_easy_idx, test_idx)
                    # now evenly mix easy and hard data together for training
                    n_easy_train = n_easy - n_test
                    eligible_easy_idx = split_rng.choice(eligible_easy_idx, size=n_easy_train, replace=False)
                    bottleneck_is_hard_data = n_hard < n_easy_train
                    bottleneck_size = n_hard if bottleneck_is_hard_data else n_easy_train
                    eligible_train_and_dev_idx = np.concatenate([
                        split_rng.choice(eligible_easy_idx, size=bottleneck_size, replace=False),
                        split_rng.choice(all_hard_idx, size=bottleneck_size, replace=False)
                    ])
                    n_train_and_dev_data = len(eligible_train_and_dev_idx)
                elif train_on == 'easy' and test_on == 'all':
                    n_test = max(.2*n, min_test_size)
                    test_idx = split_rng.choice(all_idx, size=int(n_test), replace=False)
                    eligible_train_and_dev_idx = np.setdiff1d(all_easy_idx, test_idx)
                    n_train_and_dev_data = len(eligible_train_and_dev_idx)
                elif train_on == 'medium' and test_on == 'all':
                    n_test = max(.2*n, min_test_size)
                    test_idx = split_rng.choice(all_idx, size=int(n_test), replace=False)
                    eligible_train_and_dev_idx = np.setdiff1d(all_medium_idx, test_idx)
                    n_train_and_dev_data = len(eligible_train_and_dev_idx)
                elif train_on == 'hard' and test_on == 'all':
                    n_test = max(.2*n, min_test_size)
                    test_idx = split_rng.choice(all_idx, size=int(n_test), replace=False)
                    eligible_train_and_dev_idx = np.setdiff1d(all_hard_idx, test_idx)
                    n_train_and_dev_data = len(eligible_train_and_dev_idx)
                elif train_on == 'easy_and_hard' and test_on == 'all':
                    n_test = max(.2*n, min_test_size)
                    test_idx = split_rng.choice(all_idx, size=int(n_test), replace=False)
                    eligible_easy_idx = np.setdiff1d(all_easy_idx, test_idx)
                    eligible_hard_idx = np.setdiff1d(all_hard_idx, test_idx)
                    # now evenly mix easy and hard data together for training
                    n_easy = len(eligible_easy_idx)
                    n_hard = len(eligible_hard_idx)
                    bottleneck_is_hard_data = n_hard < n_easy
                    bottleneck_size = n_hard if bottleneck_is_hard_data else n_easy
                    eligible_train_and_dev_idx = np.concatenate([
                        split_rng.choice(eligible_easy_idx, size=bottleneck_size, replace=False),
                        split_rng.choice(eligible_hard_idx, size=bottleneck_size, replace=False)
                    ])
                    n_train_and_dev_data = len(eligible_train_and_dev_idx)
                elif train_on == 'all' and test_on == 'all':
                    n_dev = int(np.ceil(.1 * n))
                    n_test = max(int(np.ceil(.2 * n)), min_test_size)
                    n_train = n - n_dev - n_test
                    test_idx = split_rng.choice(all_idx, size=int(n_test), replace=False)
                    eligible_train_and_dev_idx = np.setdiff1d(all_idx, test_idx)
                else:
                    raise NotImplementedError(f"Unexpected combination of train_on/test_on when stratify_hardness==True: train={train_on} + test={test_on}")
                # ADD IN AVAILABLE EASY DATA IF REQUESTED
                if use_extra_easy_data:
                    eligible_easy_idx = np.setdiff1d(all_easy_idx, test_idx)
                    leftover_easy_idx = np.setdiff1d(eligible_easy_idx, eligible_train_and_dev_idx)
                    eligible_train_and_dev_idx = np.concatenate([eligible_train_and_dev_idx, leftover_easy_idx])
                # set initial train/dev/test sample sizes (could be overwritten below)
                if args.no_dev:
                    n_train = len(eligible_train_and_dev_idx)
                    n_dev = 0
                else:
                    n_train_and_dev_data = len(eligible_train_and_dev_idx)
                    n_train = int(np.floor(.8*n_train_and_dev_data))
                    n_dev = int(np.ceil(.2*n_train_and_dev_data)) 
                # get train and dev idx
                # maybe avoid overfilling train_idx if we want to save extra data for test later
                if args.all_data_to_test:
                    n_train = min(max_n_train, n_train)
                train_idx = split_rng.choice(eligible_train_and_dev_idx, size=n_train, replace=False)
                remaining_idx = np.setdiff1d(eligible_train_and_dev_idx, train_idx)
                dev_idx = split_rng.choice(remaining_idx, size=n_dev, replace=False)
                # (re)allocate test data if all_data_to_test
                if args.all_data_to_test:
                    if args.test_on == 'hard':
                        test_idx = np.setdiff1d(all_hard_idx, train_idx)
                        test_idx = np.setdiff1d(test_idx, dev_idx)
                    elif args.test_on == 'easy':
                        test_idx = np.setdiff1d(all_easy_idx, train_idx)
                        test_idx = np.setdiff1d(test_idx, dev_idx)
                    elif args.test_on == 'all':
                        test_idx = np.setdiff1d(all_idx, train_idx)
                        test_idx = np.setdiff1d(test_idx, dev_idx)
                    else:
                        raise NotImplementedError(f"Not expecting args.test_on={args.test_on} with args.all_data_to_test={args.all_data_to_test}")
            probing_train = probing_data.iloc[train_idx]
            probing_dev = probing_data.iloc[dev_idx]
            probing_test = probing_data.iloc[test_idx]
            split_datasets[dataname] = {
                'probing_data': {
                        'train': probing_train,
                        'dev': probing_dev,
                        'test': probing_test,
                    },
                }
            if precomputed_hidden_states is not None:
                for prompt_idx, hidden_states in hidden_states_dict.items():
                    probing_hidden_states = hidden_states['probing_data']
                    split_hidden_states[prompt_idx] = {
                        'train': probing_hidden_states[train_idx],
                        'dev': probing_hidden_states[dev_idx],
                        'test': probing_hidden_states[test_idx]
                    }
                split_datasets[dataname][f"{data_type}_states"] = split_hidden_states
    return split_datasets

def get_sample_efficiency_subsets(supervision_n, split_datasets, seed, data_type='hardness'):
    '''
    Takes nested datasets returned by split_datasets, and subsets the train split while preserving the dev spits. 
    - returns nested_dict with hardness_data and hardness_states, where train splits have n points and dev splits untouched
    - structure is n: dataname: {data_type}_data/{data_type}_states: train/dev/test: data
    - Performs recursive sampling so for a list of supervision_n = [5,10,20], first sample 20 points, then 10 from those, then 5 from those
    '''
    split_rng = np.random.default_rng(seed) # the splits will change between seeds if the number of n or the number of datasets changes, unfortunately
    n_to_datasets = {}
    # break up the text and hidden state subsets for separate recursive function calls
    text_splits = {}
    representation_splits = {}
    for dataname, data_dict in split_datasets.items():
        for k,v in data_dict.items():
            # break up data by type
            if k == f"{data_type}_data":
                text_splits[dataname] = v
            if k == f"{data_type}_states":
                representation_splits[dataname] = v
    # get nested sample idx for each datasets
    dataname_to_nested_idx = {}
    for dataname, data_dict in split_datasets.items():
        dataname_to_nested_idx[dataname] = {}
        n_train = len(data_dict[f"{data_type}_data"]['train'])
        available_idx = np.arange(n_train)
        for n in reversed(sorted(supervision_n)):
            if n_train >= n:
                subset_idx = split_rng.choice(available_idx, size=n, replace=False)
                available_idx = subset_idx
            else:
                print(f"Requesting too much data for {dataname} during subsampling...want {n} but only have {n_train}")
                subset_idx = split_rng.choice(available_idx, size=len(available_idx), replace=False)
            dataname_to_nested_idx[dataname][n] = subset_idx
    # now iterate through and construct dicts with n train points each, per dataset
    for n in reversed(sorted(supervision_n)):
        n_to_datasets[n] = {}
        for dataname in split_datasets.keys():
            n_to_datasets[n][dataname] = {f'{data_type}_data' : {}, f'{data_type}_states' : {}}
            subset_idx = dataname_to_nested_idx[dataname][n]
            # subset text data
            for key in ['train', 'dev', 'test']:
                if key in text_splits[dataname]:
                    if subset_idx is None:
                        new_data = None
                    elif key == 'train':
                        new_data = text_splits[dataname][key].iloc[subset_idx]
                    else: # do not subset dev/test splits
                        new_data = text_splits[dataname][key]
                    n_to_datasets[n][dataname][f'{data_type}_data'][key] = new_data
            # subset representations
            if dataname in representation_splits:
                for prompt_id, prompt_data in representation_splits[dataname].items():
                    n_to_datasets[n][dataname][f'{data_type}_states'][prompt_id] = {}
                    for key in ['train', 'dev', 'test']:
                        if key in prompt_data:
                            if subset_idx is None:
                                new_data = None
                            elif key == 'train':
                                new_data = prompt_data[key][subset_idx]
                            else:
                                new_data = prompt_data[key]
                        n_to_datasets[n][dataname][f'{data_type}_states'][prompt_id][key] = new_data
    return n_to_datasets

def prepare_dataset_for_dataloader(args, probing_method, split_dataset, dataname, prompt, eval_prompt_id, tokenizer, data_type='hardness',
                                   k=0, multitask_training=False, multiprompt_training=False, 
                                   force_test_dataname=None,
                                   use_cot=False, all_split_datasets=None):
    '''
    This function promptifies the data and adds new columns to the data, model_input and model_output, which are used in later Dataloaders
    - returns nested dict of dataname: {data_type}_data/{data_type}_states: train/dev/test: data
    - train data are combined based on multitask and multiprompt args. dev and test data are untouched, meaning they are always task and prompt specific 
    - k is used to get k examples in ICL learning. The way this is works is to make the train_data df empty, and get a new prompt_ex df from it
    - one detail: assumes same number of prompts for each dataset, which is given by prompt.num_prompts
    args:
        split_dataset: a single split dataset, i.e. split_datasets[dataname], which contains hardness/probing_data and hardness/probing_states, split by train/dev/test
        all_split_datasets:  datasets dict returned by split_datasets, or (more commonly) one of the nested dicts from get_sample_efficiency_subsets (used for multitask training)
        force_test_dataname: force test data to be from this dataset. used for task transfer experiments 
    '''
    # make return dict and unpack text_data/hidden_states from the provided split_dataset dict
    return_dataset = {}
    return_dataset[f'{data_type}_data'] = {}
    return_dataset[f'{data_type}_states'] = {}
    train_test_splits = ['train', 'dev'] if data_type == 'hardness' else ['train', 'dev', 'test']
    all_prompt_ids = prompt.dataname_to_prompt_ids[dataname]
    # iterate through train/dev/test spits and make each. the potentially tricky one is train
    for split_name in train_test_splits:
        if split_name == 'train':
            all_text_data = []
            all_hidden_states = []
            collect_dataname_prompt_ids = []
            # maybe collect multiple datasets and prompts based on multitask/multiprompt training
            if multitask_training and multiprompt_training:
                for train_dataname in all_split_datasets.keys():
                    for prompt_id in all_prompt_ids:
                        collect_dataname_prompt_ids.append((train_dataname, prompt_id))
            elif multitask_training and not multiprompt_training:
                for train_dataname in all_split_datasets.keys():
                    collect_dataname_prompt_ids.append((train_dataname, eval_prompt_id))
            elif not multitask_training and multiprompt_training:
                for prompt_id in all_prompt_ids:
                    collect_dataname_prompt_ids.append((dataname, prompt_id))
            else:
                collect_dataname_prompt_ids.append((dataname, eval_prompt_id))
            # make prompt examples first if k>0
            if k is not None and k>0 and probing_method == 'decoding':
                prompt_datasets = []
                for collect_dataname, prompt_id in collect_dataname_prompt_ids:
                    single_dataset = all_split_datasets[collect_dataname][f"{data_type}_data"][split_name]
                    prompt_datasets.append(single_dataset)
                prompt_dataset = pd.concat(prompt_datasets)
                prompt_ex, _ = LM_utils.pull_prompt_from_data(prompt_dataset, k) # do not overwrite the single_dataset
                # noise the prompt labels if requested
                if args.noise_labels_p > 0:
                    assert not (args.hardness_method == 'decoding' and args.estimate_hardness), "This would be a bad condition...don't estimate hardness for decoding using k>0 with random labels"
                    prompt_ex = randomize_dataframe_labels(prompt_ex, 
                                                           args.seed,
                                                           p=args.noise_labels_p,
                                                           randomize_reasoning=use_cot)
            else:
                prompt_ex = None
            for collect_dataname, prompt_id in collect_dataname_prompt_ids:
                # format and accumulate text dataset
                single_dataset = all_split_datasets[collect_dataname][f"{data_type}_data"][split_name]
                if len(single_dataset) > 0: 
                    single_dataset = prepare_dataframe_for_dataloader(single_dataset, prompt_ex,
                                                                    collect_dataname, tokenizer,
                                                                    prompt, prompt_id, 
                                                                    add_reasoning_targets=use_cot and probing_method =='finetuned',
                                                                    max_seq_len=args.max_seq_len)
                all_text_data.append(single_dataset)
                # collect and accumulate hidden representations
                if probing_method == 'learned':
                    hidden_states = all_split_datasets[collect_dataname][f"{data_type}_states"][prompt_id][split_name]
                    all_hidden_states.append(hidden_states)
            text_data = pd.concat(all_text_data)
            all_hidden_states = np.concatenate(all_hidden_states) if len(all_hidden_states) > 0 else None
            # noise labels if requested. These are not prompt_ex as above but the entire training set
            if args.noise_labels_p > 0:
                assert not args.estimate_hardness, "This would be a bad condition...don't estimate hardness with random labels"
                text_data = randomize_dataframe_labels(text_data, args.seed, p=args.noise_labels_p, randomize_reasoning=False)
        elif split_name != 'train':
            # format and accumulate text dataset
            _test_dataname = force_test_dataname if force_test_dataname != 'NA' else dataname
            text_data = all_split_datasets[_test_dataname][f"{data_type}_data"][split_name]
            if len(text_data) > 0: 
                text_data = prepare_dataframe_for_dataloader(text_data, prompt_ex, 
                                                                _test_dataname, tokenizer, 
                                                                prompt, eval_prompt_id, 
                                                                add_reasoning_targets=False,
                                                                max_seq_len=args.max_seq_len)
            # collect and accumulate hidden representations
            if probing_method == 'learned':
                all_hidden_states = all_split_datasets[_test_dataname][f'{data_type}_states'][eval_prompt_id][split_name]
            # reorder dev/test data by pd idx for debugging purposes
            # print("sorting dev/test data for debugging purposes -- note this will break the linear probing method")
            # text_data = text_data.sort_index()
        return_dataset[f'{data_type}_data'][split_name] = text_data
        if probing_method == 'learned':
            assert text_data.shape[0] == all_hidden_states.shape[0], "Uh oh, unequal numbers of text_data datapoints and corresponding hidden states"
            return_dataset[f'{data_type}_states'][split_name] = all_hidden_states
    return return_dataset, prompt_ex

def prepare_dataframe_for_dataloader(dataframe, prompt_ex, dataname, tokenizer, prompt, eval_prompt_id,
                                     max_seq_len, add_reasoning_targets=False):
    '''
    First standardizes the answer choice and label columns for the dataset
    This function promptifies the data and adds new columns to the data, model_input, answer_choices, and label_idx, which are used in later Dataloaders
    It then drops all columns except these three
    '''
    # prepped_data = standardize_column_names(dataname, dataframe.copy())
    # prepped_prompt_ex = standardize_column_names(dataname, prompt_ex.copy()) if prompt_ex is not None else None
    prepped_data = add_model_inputs_to_df(dataframe, 
                            prompt_ex=prompt_ex, 
                            tokenizer=tokenizer, 
                            dataname=dataname, 
                            prompt=prompt, 
                            prompt_id=eval_prompt_id, 
                            max_seq_len=max_seq_len,
                            count_reasoning_in_length=add_reasoning_targets)
    if add_reasoning_targets:
        prepped_data = add_reasoning_targets_to_df(prepped_data, dataname, prompt, eval_prompt_id)
        final_cols = ['model_input', 'answer_choices', 'label_idx', 'letter_labels', 'reasoning_target']
    else:
        final_cols = ['model_input', 'answer_choices', 'label_idx', 'letter_labels']
    prepped_data.loc[:,final_cols]
    return prepped_data

def add_model_inputs_to_df(df, prompt_ex, dataname, tokenizer, prompt, prompt_id, max_seq_len=1024, count_reasoning_in_length=False):
    '''
    This function takes a dataframe and adds a column model_input that is the promptified version of the model inputs
    Does not handle multiple-choice formatting or tokenization
    Does handle max_seq_len by checking if tokenizing the prompt results in an overly long prompt
    returns:
        df of size (n*len(prompt_ids) x n_cols) for input df of len n
    '''
    add_col = []
    prompt_kwargs = prompt.get_prompt_kwargs_from_id(prompt_id, dataname)
    for idx, datapoint in df.iterrows():
        # MANUALLY INSERT A TEST POINT HERE FOR DEBUGGING
        # if idx == 0:
        #     datapoint.input_text = "Will the Albany in Georgia reach a hundred thousand occupants before the one in New York?"
        #     datapoint.answer_idx = 1
        # process point
        input_prompt = prompt.format_prompt_from_df(datapoint, 
                                examples_df=prompt_ex,
                                **prompt_kwargs)
        # check tokenization length of prompt -- this will avoid assertion error for len inside make_LM_batch
        answer_choices = datapoint.answer_choices
        label_idx = datapoint.label_idx
        longest_answer_choice = np.argmax([len(tokenizer.encode(answer)) for answer in answer_choices])
        encode_answer = answer_choices[longest_answer_choice]
        ids_1 = tokenizer.encode(input_prompt, add_special_tokens=False)
        len_ids_1 = len(ids_1)
        len_ids_2 = len(tokenizer.encode(f"{encode_answer}", add_special_tokens=False))
        if count_reasoning_in_length:
            len_ids_2 += len(tokenizer.encode(datapoint.reasoning, add_special_tokens=False))
        if len_ids_1 + len_ids_2 + 1 > max_seq_len: # +1 for bos/eos token
            shorten_to = max_seq_len - 2 - len_ids_2 # -2 for bos+eos token
            # if k is already 0, need to truncate the prompt
            if prompt_ex is None:
                print(f" idx {idx}: Shortening input to fit in max seq len, from {len_ids_1} tokens to {shorten_to}")
                input_prompt = tokenizer.decode(ids_1[:shorten_to])
            # decrease prompt k if prompt is too long
            else:
                _examples_df = prompt_ex.copy()
                while len(tokenizer.encode(input_prompt, add_special_tokens=False)) > shorten_to:
                    print(f" idx {idx}: Prompt too long, shortening from {len(_examples_df)} to {len(_examples_df)-1} examples (consider reducing k)")
                    _examples_df = _examples_df.iloc[:-1,...]
                    input_prompt = prompt.format_prompt_from_df(datapoint, 
                                        examples_df=_examples_df,
                                        **prompt_kwargs)
                    if len(_examples_df) == 0:
                        print("  Completely empty prompt for this test question.")
                        break
        label_idx = datapoint.label_idx
        answer_choices = datapoint.answer_choices
        answer = answer_choices[label_idx]
        # check tokenization length of prompt again and remove the ending now as necessary (the previous step is focused on the prompt, not the test input). This could severely mess up the datapoint
        len_ids_1 = len(tokenizer.encode(input_prompt, add_special_tokens=False))
        len_ids_2 = len(tokenizer.encode(f"{answer}", add_special_tokens=False))
        # ad-hoc adjustment for reasoning length in targets
        if count_reasoning_in_length:
            len_ids_2 += len(tokenizer.encode(datapoint.reasoning, add_special_tokens=False))
        if len_ids_1 + len_ids_2 + 1 > max_seq_len: # +1 for bos/eos token
            shorten_to = max_seq_len - 2 - len_ids_2 # -2 for eos+bos token
            # if k is already 0, need to truncate the prompt
            print(f" idx {idx}: Shortening input to fit in max seq len, from {len_ids_1} tokens to {shorten_to}")
            input_prompt = tokenizer.decode(ids_1[:max(shorten_to, 0)])
        # accumulate input, answer choices, and labels
        add_col.append(input_prompt)
    return_df = df.copy()
    return_df['model_input'] = add_col
    return return_df

def add_reasoning_targets_to_df(df, dataname, prompt, prompt_id):
    '''
    This function takes a dataframe and adds a column model_input that is the promptified version of the model inputs
    Does not handle multiple-choice formatting or tokenization
    Does handle max_seq_len by checking if tokenizing the prompt results in an overly long prompt
    returns:
        df of size (n*len(prompt_ids) x n_cols) for input df of len n
    '''
    add_col = []
    prompt_kwargs = prompt.get_prompt_kwargs_from_id(prompt_id, dataname)
    for idx, datapoint in df.iterrows():
        # process point
        reasoning_target = prompt.format_reasoning_target_from_df(datapoint, **prompt_kwargs)
        add_col.append(reasoning_target)
    return_df = df.copy()
    return_df['reasoning_target'] = add_col
    return return_df
        
    
        

        
