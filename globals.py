# burns_datasets = ["imdb", "amazon_polarity", "ag_news", "dbpedia_14", "copa", "rte", "boolq", "piqa", "qnli", "story_cloze"]
mmlu_datasets = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
arc_datasets = ['ARC-Easy', 'ARC-Challenge']

# mmlu globals
mmlu_subject_levels = ['college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_physics',
                       'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics', 'high_school_physics']
mmlu_subjects = ['mmlu_biology', 'mmlu_chemistry', 'mmlu_computer_science', 'mmlu_mathematics', 'mmlu_physics']
mmlu_combined = ['mmlu_STEM-5']
third_grade_to_college = ['ai2_arc_all', 'mmlu_STEM-5']

# hardness vs. probing data globals
known_hardness_data =  ['ai2_arc', 'ai2_arc_all', 'strategy-qa', 'strategy-qa-dev', 'gsm8k_main', 'gsm8k', 'gsm8k_socratic', 'gsm8k_main_test', 'mmlu_subjects'] + mmlu_combined
still_make_hardness_data =  ['ai2_arc', 'gsm8k_main', 'gsm8k_socratic']
probing_data_only = ['strategy-qa', 'strategy-qa-dev', 'gsm8k_main_test'] # don't make a separate split for hardness model estimation 
# other eligible datasets
eligible_datasets = mmlu_datasets + arc_datasets

# define easy/medium/hard ranges/bounds for the data. default to 30/40/30 percentile chunks if set to None
data_x_hardness_var_to_cutoffs = {
    'ai2_arc': {
        'human_bloom': (2,4),
        'human_difficulty': (1,3),
        'human_grade': (5,8),
        'human_depth_of_knowledge': (1,3),
    },
    'ai2_arc_all': {
        'human_bloom': (2,4),
        'human_difficulty': (1,3),
        'human_grade': (5,8),
        'human_depth_of_knowledge': (1,3),
    },
    'mmlu_STEM-5': {
        'human_hardness': (0,1),
    },
    'strategy-qa': {
        'num_steps': (2,4),
    },
    'gsm8k_main': {
        'num_steps': (4,7),
    },
}
# mmlu extra stats to record
mmlu_subject_stat_cols = [
    'math_prop_TRAIN',
    'physics_prop_TRAIN',
    'chem_prop_TRAIN',
    'bio_prop_TRAIN',
    'cs_prop_TRAIN',
    'math_prop_TEST',
    'physics_prop_TEST',
    'chem_prop_TEST',
    'bio_prop_TEST',
    'cs_prop_TEST',
]
# average hardness scores over these models
hardness_models = [
        "huggyllama/llama-7b",
        "tiiuae/falcon-7b",
        "mistralai/Mistral-7B-v0.1",
        "mosaicml/mpt-7b",
]
llama_models = ['Llama-2-7b', 'Llama-2-13b', 'Llama-2-70b', 'Llama-2-7b-chat', 'Llama-2-13b-chat', 'Llama-2-70b-chat']
base_llama_models = ['Llama-2-70b', 'Llama-2-13b', 'Llama-2-7b']
llama_one_gpu_models = ['Llama-2-7b', 'Llama-2-13b']
one_gpu_models = [model for model in hardness_models + llama_models if not '70b' in model]
four_gpu_models = [model for model in hardness_models + llama_models if '70b' in model]

replicate_models = ['Llama-2-70b', 'Llama-2-70b-chat', 'mistralai/Mixtral-8x7B-v0.1', 'Qwen/Qwen-72B']

# don't use EleutherAI/ or facebook/ etc. prefixes below
model_to_hidden_size = {
    'gpt2-medium': 1024,
    'gpt2-xl': 1600,
    'gpt-j-6B': 4096,
    't5-xl': 1024, # t5 not tested
    't5-xxl': 1024,
    'flan-t5-xl': 1024,
    'flan-t5-xxl': 1024,
    'llama-7b': 4096,
    'llama-13b': 5120,
    'llama-30b': 6656, # really 33b
    'llama-65b': 8192,
    'Llama-2-7b': 4096,
    'Llama-2-13b': 5120,
    'Llama-2-70b': 8192,
    'Llama-2-7b-chat': 4096,
    'Llama-2-13b-chat': 5120,
    'Llama-2-70b-chat': 8192,
    'falcon-7b': 4544,
    'falcon-7b-instruct': 4544,
    'falcon-40b': 8192,
    'falcon-40b-instruct': 8192,
    'persimmon-8b-base': 4096, # this is a 9.3b parameter model...
    'mpt-7b': 4096,
    'Mistral-7B-v0.1': 4096,
    'opt-13b': 5120,
    'Qwen-72B': 8192, 
    'Mixtral-8x7B-v0.1': 8192, # double check?
}
label_dict = {
    "imdb": ["negative", "positive"], # This is for normal IMDB
    "amazon_polarity": ["negative", "positive"],
    "ag_news": ["politics", "sports", "business", "technology"],
    "dbpedia_14": ["company", "educational institution", "artist", "athlete", "office holder", "mean of transportation", "building", "natural place", "village", "animal",  "plant",  "album",  "film",  "written work"],
    "copa": ["choice 1", "choice 2"],
    "rte": ["yes", "no"],   # whether entail
    "boolq": ["false", "true"],
    "qnli": ["yes", "no"],  # represent whether entail
    "piqa": ["solution 1", "solution 2"],
    "story_cloze": ["choice 1", "choice 2"],
}