This is the codebase for the paper [The Unreasonable Effectiveness of Easy Training Data for Hard Tasks](https://peterbhase.github.io/files/easy-to-hard-generalization.pdf).

## Install Requirements

```
pip install -r requirements.txt
```

### Experiment Commands

Below, we describe how to replicate the main experimental results in our paper. 

We begin with a few examples of experiments that one should be able to run with the codebase. Note to use Llama-2 models, the `llama2_path` variable must be set in `utils/utils.py`. 

#### Run llama-7b on ARC Challenge test with a zero-shot prompt

Note that you must supply the `--model_dir` and `--cache_dir` args for saving/storing models by setting the `MODEL_DIR` and `CACHE_DIR` environment variables. Lowering the eval batch size (`-ebs`) to 4 (the minimum value given that ARC is 4-way multiple-choice) should help fit onto a smaller GPU.

```
python main.py --model huggyllama/llama-7b --do_eval true -llm true --probing_style decoding --dataset ai2_arc/ARC-Challenge  --hardness_var_name NA --specific_prompt 0040 -pb 1 -np 1 --stratify_hardness false --k_shot 0 -ebs 8 --all_data_to_test true --model_dir $MODEL_DIR --cache_dir $CACHE_DIR
```

#### Run llama-13b on our combined ARC data with a zero-shot prompt

```
python main.py --model huggyllama/llama-13b --do_eval true -llm true --probing_style decoding --dataset ai2_arc  --hardness_var_name NA --specific_prompt 0040 -pb 1 -np 1 --stratify_hardness false --k_shot 0 -ebs 10 --all_data_to_test true --model_dir $MODEL_DIR --cache_dir $CACHE_DIR
```

#### Run Mixtral-8x7B on college level MMLU-STEM-5 data with a 10-shot prompt containing high school examples, using 5 random seeds

```
python main.py --model mistralai/Mixtral-8x7B-v0.1 --do_eval true -llm true --probing_style decoding --dataset mmlu_STEM-5 --hardness_var_name human_hardness --specific_prompt 0040 -pb 5 -np 1 --stratify_hardness true --train_on easy --test_on hard --k_shot 0 -ebs 10 --all_data_to_test true --model_dir $MODEL_DIR --cache_dir $CACHE_DIR
```

### Paper Research Question Experiments

Now we describe how to replicate the main results in our paper using the `run_jobs.py` file. In general, you have to edit the `use_models` and `use_methods` in this file in order to *not* run experiments across Llama-2-7b, Llama-2-13b, Llama-2-70b, and all relevant training method including ICL, ICL+CoT, linear probing, QLoRA, and QLoRA+CoT. Note that using `Llama-2-70b` requires four 48gb gpus to load in 8bit quantization.

First, if you want to use linear models later on, then write model hidden states to file, which is a precursor to linear modeling. 

```
python run_jobs.py -e write_hidden_states --dataset ai2_arc  
python run_jobs.py -e write_hidden_states --dataset mmlu_STEM-5  
python run_jobs.py -e write_hidden_states --dataset strategyQA
```

If you want to use model-based MDL metrics later on, estimate model-based hardness for these datasets. To use fewer than our four default 7b models, edit `globals.hardness_models`. 

```
python run_jobs.py -e estimate_hardness --dataset ai2_arc  
python run_jobs.py -e estimate_hardness --dataset strategy-qa  
python run_jobs.py -e estimate_hardness --dataset mmlu_STEM-5
```

To get all-to-all performance (comparable to paper Table 4), run the following commands.

```
python run_jobs.py -e all_to_all_table --dataset ai2_arc -nb 5 -lc 0 --n_train 160 --k_shot 10  
python run_jobs.py -e all_to_all_table --dataset mmlu_STEM-5 -nb 5 -lc 0 --n_train 160 --k_shot 10  
python run_jobs.py -e all_to_all_table --dataset strategy-qa -nb 5 -lc 0 --n_train 160 --k_shot 8  
python run_jobs.py -e all_to_all_table --dataset gsm8k_main -nb 5 -lc 0 --n_train 160 --k_shot 8
```

Now to get results for the main easy-to-hard generalization results (RQ2 in the paper), run the below commands. To adjust which hardness measures are used for dataset stratification, adjust the value of `stratify_var_names`.

```
python run_jobs.py -e get_population_table --dataset ai2_arc -nb 5 -lc 0 --n_train 160 --k_shot 10  
python run_jobs.py -e get_population_table --dataset mmlu_STEM-5 -nb 5 -lc 0 --n_train 160 --k_shot 10  
python run_jobs.py -e get_population_table --dataset strategy-qa -nb 5 -lc 0 --n_train 160 --k_shot 8  
python run_jobs.py -e get_population_table --dataset gsm8k_main -nb 5 -lc 0 --n_train 160 --k_shot 8  
```

To get our Figure 1 plot, which measures college test performance for a model prompted with 3rd grade / 8th grade / high school data, run:

```
python run_jobs.py -e third_grade_to_college -nb 5 -lc 0 --n_train 160 --k_shot 10 -rj 0
```

To get results with noisy training labels (RQ3), run:

```
python run_jobs.py -e noisy_labels_table --dataset mmlu_STEM-5 -nb 5 -lc 0 --n_train 160
```

To get learning curves with linear probes, to estimate performance w.r.t. training cost (RQ4), first set `use_methods=['learned_CoT=False]` and `use_models = ['Llama-2-70b']` in `get_population_table`, then run:

```
python run_jobs.py -e get_population_table --dataset ai2_arc -nb 10 -lc 1  
python run_jobs.py -e get_population_table --dataset mmlu_STEM-5 -nb 10 -lc 1  
python run_jobs.py -e get_population_table --dataset strategy-qa -nb 10 -lc 1
```

## Data Analysis

We provide the R markdown file used for data analysis. The above `run_jobs.py` experiments will output .csv's into a `result_sheets` directory. The `analysis.Rmd` file loads results from this directory for plotting.

### bibtex

To cite this work, you can use

```
placeholder
```
















