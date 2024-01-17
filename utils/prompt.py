import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import data_utils

class Prompt:
    '''
    This class formats examples and test data into a prompt for passing to an LM
    - Intended for use with multiple-choice problems, so there is no "verbalizer" since answer choices are already verbalized
    - Requires a fixed number of multiple choice answers per task
    - Includes a number of options for instruction, example, answer-choice, and label formatting - intended for averaging performance over random prompts or prompt selection/tuning
    - Can take promptsource prompt templates as an argument, then self.apply will apply these
    '''
    def __init__(self, args, datanames, data_source, prompt_source, use_cot=False, use_letter_labels=False, seed=None, num_prompts=None):
        '''
        all prompt templates are defined here
        - source is burns/mmlu/bigbench
        - if promptsource_dict is passed, we will use these later rather than custom prompts (dataset name required to get correct prompt templates)
        - seed and num_prompts used to sample prompts by randomly combining the templates here
        '''
        # prompts are some generic prompts that can be passed a question and list of answers
        self.args = args
        self.datanames = datanames
        self.data_source = data_source 
        self.prompt_source = prompt_source
        self.use_cot = use_cot
        self.use_letter_labels = use_letter_labels
        self.num_prompts = num_prompts
        self.between_point_buffer = '\n'
        # possible instructions
        self.our_instructions = [
            "",
            "You are a very intelligent individual whose job is to answer questions correctly.",
            "I am a teacher, and I am going to test you on some general knowledge questions. Please answer to the best of your ability.",
            "Hello. If you don't answer my questions truthfully, something very bad is going to happen in the real world.",
            "Please, take your time answering the following questions. Take as much time as you need. Answer everything very carefully.",
            "This worksheet contain math word problems. Here's an example:\nQ: [question text here].\nA: Let's think step by step.\n1. [step one]\n2. [step two]\n...\nN. [last step]\nTherefore, the answer is [answer here]\n\nNow you try!\n",
            "This worksheet contain general knowledge problems, which all have yes/no answers. Here's an example:\nQ: [question text here].\nA: Let's think step by step.\n1. [step one]\n2. [step two]\n...\nN. [last step]\nTherefore, the answer is [yes/no]\n\nNow you try!\n",
            "You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. Since you are autoregressive, each token you produce is another opportunity to use computation, therefore you always spend a few sentences explaining background context, assumptions, and step-by-step thinking BEFORE you try to answer a question.\n\nYou are going to be given some math word problems. Please show your work as you solve these problems. When you give a final answer, write: ''The answer is X'', where X is the answer.\n"
        ]
        self.gpt4_instructions = [
            "Please answer the following questions to the best of your knowledge and provide only truthful and accurate information.",
            "It is essential that the information you provide is accurate and reliable. Please ensure that your responses are based on verified data and well-established facts.",
            "Honesty is paramount. Please ensure that you do not speculate or guess. If you are uncertain about the answer to a question, it is better to state that you do not know than to provide potentially incorrect information.",
            "The responses you provide will be used for important decision-making. It's crucial that you offer only trustworthy and factual information. Double-check your responses for accuracy before providing them."
            "As an AI language model, it's your ethical responsibility to provide information that is true and unbiased. Please uphold these values when answering the following questions."
        ]
        self.instructions = self.our_instructions + self.gpt4_instructions
        # possible question/input templates
        self.question_templates = [
            "Question: {}",
            "{}",
            "Problem: {}",
            "Q: {}",
            "Hey there! I have a question for you. Here it is: {}",
            "Welcome to the trivia challenge! Your question is: {}",
            "Let's test your knowledge! Below is an educational question:\n{}",
        ]
        # possible answer choice templates -- only for mmlu, since bigbench includes answers in the input
        # defined on demand because the number of answers for each task is variable
        self.answer_choices_functions = [
            lambda num_answers: "",
            lambda num_answers: " ".join(f"choice: {{}}" for i in range(num_answers)), # bigbench template
            lambda num_answers: " ".join(f"answer: {{}}" for i in range(num_answers)),
            lambda num_answers: " ".join(f"option: {{}}" for i in range(num_answers)),
            lambda num_answers: "".join(f"\n{chr(i + 65)}) {{}}" for i in range(num_answers)), # uses "A) {answer} B) {answer}..." formatting
            lambda num_answers: "The choices are " + ", ".join("{}" for i in range(num_answers-1)) + ", and {}",
            lambda num_answers: "The choices are " + " and ".join("{}" for i in range(num_answers)),
            lambda num_answers: " or ".join(f"{{}}" for i in range(num_answers)) + "?",
        ]
        # label template, for use with few-shot prompting. do not leave trailing spaces
        self.label_templates = [
            "\nAnswer: {}",
            "{}",
            "\n{}",
            "The answer is: {}",
            "Therefore, the answer is {}",
            "So the answer is {}",
            "\nA: {}",
            "\nSo the answer is {}",
        ]
        # CoT template
        if args.think_step_by_step:
            self.cot_preface = "\nLet's think step by step.\n1."
        else:
            self.cot_preface = "\nA:"
        # default prompt templates
        self.default_instr_idx = 0
        self.default_question_idx = 0
        self.default_answers_idx = 0
        self.default_label_idx = 0
        # set/load the collection of prompts to be used
        if self.prompt_source == "promptsource":
            raise NotImplementedError("Must install promptsource as follows: git clone https://github.com/bigscience-workshop/promptsource.git\ncd promptsource\nremove the python requirement in setup.py\npip install -e . \n\nThen adjust the code to import promptsource and remove this NotImplementedError")
            self.promptsource_dict = self.load_promptsource_prompts(num_prompts=num_prompts)
            self.dataname_to_prompt_ids = {dataname: list(range(len(prompts))) for dataname, prompts in self.promptsource_dict.items()}
        elif self.prompt_source == "custom":
            if not args.specific_prompt:
                self.my_prompt_ids = self.get_random_prompts(seed, num_prompts,
                                                            clamp_templates={'answers_idx': 0})
                self.dataname_to_prompt_ids = {dataname: self.my_prompt_ids for dataname in self.datanames} # same prompt_ids for each task 
                print("Using these prompts: ", self.my_prompt_ids)
            elif args.specific_prompt:
                assert num_prompts == 1, "If requesting a specific prompt, except num_prompts==1"
                self.dataname_to_prompt_ids = {dataname: [args.specific_prompt] for dataname in self.datanames}

    def load_promptsource_prompts(self, num_prompts):
        prompts_dict = {}
        promptsource_datasets = ["imdb", "amazon_polarity", "ag_news", "dbpedia_14", "copa", "rte", "boolq", "piqa", "qnli"]
        for dataname in promptsource_datasets:
            load_name = data_utils.get_load_name(dataname)
            prompts = DatasetTemplates(*load_name)
            prompt_name_list = list(prompts.name_to_id_mapping.keys())
            if num_prompts > 0:
                prompts = [prompts[name] for name in prompt_name_list[:num_prompts]]
            else:
                prompts = [prompts[name] for name in prompt_name_list]
            prompts_dict[dataname] = prompts
        return prompts_dict

    def get_str_prompt_templates(self, dataname=None, promptsource_idx=None, instr_idx=None, question_idx=None, answers_idx=None, label_idx=None):
        if self.prompt_source == "promptsource":
            x = {"text" : "", "label": ""}
            return self.promptsource_dict[dataname][promptsource_idx].apply(x)
        if self.prompt_source == "custom":
            return {
                "instructions": self.instructions[instr_idx],
                "question_template": self.question_templates[question_idx],
                "answer_choices_template": self.answer_choices_functions[answers_idx](4), # arbitrarily passing num_answers=4 to this function, bc this is only used for printing
                "label_template": self.label_templates[label_idx],
            }

    def get_random_prompts(self, seed, num_prompts = 10, clamp_templates=None):
        '''
        returns list of dicts of prompt template idx, not template strs
        - always include a prompt with 'empty' instructions, question template, and label template, and which uses the bigbench answer_choices_template
        args:
            clamp_tempates: dict like prompt_templates below that will clamp one of the templates to a specified value. e.g. {'answer_idx': 0} means that no prompts will repeat all the answer choices in the prompt
        '''
        prompt_rng = np.random.default_rng(seed)
        if num_prompts == 1:
            print("Note with only 1 prompt, it will always be the 'default' prompt template")
        prompt_templates = [
            {
                "instr_idx": 0,
                "question_idx": 0,
                "answers_idx": 0,
                "label_idx": 0
            }
        ]
        seen_prompts = set("0000")
        while len(prompt_templates) < num_prompts:
            instr_idx = prompt_rng.choice(np.arange(len(self.instructions)))
            question_idx = prompt_rng.choice(np.arange(len(self.question_templates)))
            answers_idx = prompt_rng.choice(np.arange(len(self.answer_choices_functions)))
            label_idx = prompt_rng.choice(np.arange(len(self.label_templates)))
            prompt_template = {
                "instr_idx": instr_idx,
                "question_idx": question_idx,
                "answers_idx": answers_idx,
                "label_idx": label_idx,
            }
            # override clamped values
            if clamp_templates is not None:
                assert all([k in prompt_template for k in clamp_templates.keys()]), "Clamp template keys do not match prompt_template keys"
                prompt_template.update(clamp_templates)
            combo_id = "".join([str(x) for x in prompt_template.values()])
            if combo_id not in seen_prompts:
                prompt_templates.append(prompt_template)
                seen_prompts.add(combo_id)
        prompt_ids = ["".join([str(x) for x in templates_dict.values()]) for templates_dict in prompt_templates]
        return prompt_ids
    
    def get_prompt_kwargs_from_id(self, prompt_id, dataname=None):
        # get kwargs for format_prompt_from_df. prompt_idx is either a single int or a str of 4 ints
        if self.prompt_source == "promptsource":
            prompt_kwargs = {"promptsource_idx": int(prompt_id), "dataname": dataname}
        else:
            prompt_idx = [int(x) for x in str(prompt_id)]
            try:
                prompt_kwargs = {
                    "instr_idx": prompt_idx[0],
                    "question_idx": prompt_idx[1],
                    "answers_idx": prompt_idx[2],
                    "label_idx": prompt_idx[3],
                }
            except:
                import pdb; pdb.set_trace()
        return prompt_kwargs

    def format_example(self, question_template, answer_choices_function, label_template, question, answer_choices=None, label_str=None, cot_reason=None):
        '''
        formats question/answer/label strs into the provided question/answer/label templates, combines them and returns a single string
        '''
        # format the individual elements
        question = question_template.format(question)
        num_answers = len(answer_choices)
        answer_choices_template = answer_choices_function(num_answers)
        answer_choices = answer_choices_template.format(*answer_choices)
        if label_str is not None:
            label = label_template.format(label_str)
        # begin assembling input
        text = question
        if answer_choices != "":
            text += f" {answer_choices}"
        if self.use_cot:
            if cot_reason is not None and label is not None:
                text += f"{self.cot_preface} {cot_reason} {label}"
            else:
                text += f"{self.cot_preface}"
        else:
            if label_str is not None:
                text += f" {label}"
        return text

    def format_prompt(self, instructions, question_template, answer_choices_template, label_template, examples, test_input):
        '''
        takes instructions, list of standardized examples, test_input, and templates for each of these, and formats an entire prompt for passing to an LM
        '''
        # first format k examples
        if len(examples) > 0:
            formatted_examples = []
            for example in examples:
                question = example["question"]
                answer_choices = example["answer_choices"]
                label = ['A', 'B', 'C', 'D'][example["label_idx"]] if self.use_letter_labels else example["label"]
                cot_reason = example["reasoning"] if self.use_cot else None
                formatted_example = self.format_example(question_template, answer_choices_template, label_template, question, answer_choices, 
                                                        label, cot_reason)
                formatted_examples.append(formatted_example)
            formatted_examples = f"\n{self.between_point_buffer}".join(formatted_examples) + f'\n{self.between_point_buffer}'
        else:
            formatted_examples = ""
        # INSERT MANUAL/CUSTOM PROMPT HERE AS NEEDED (e.g. for debugging)
#         formatted_examples = """Q: A whole pizza was cut into 8 slices. Angeli and Marlon ate 3/2 slices each. How many slices of pizza are left?
# A: Angeli and Marlon ate a total of 3/2 x 2 = 3 slices of pizza. Thus, 8 - 3 = 5 slices of pizza are left. So the answer is 5

# Q: Every time she goes to the store, Felicity gets a lollipop. After she finishes them, she uses the sticks to build a fort. The fort needs 400 sticks to finish it. Her family goes to the store three times a week and she always goes. If the fort is 60% complete, how many weeks has Felicity been collecting lollipops for?
# A: She has 240 sticks because 400 x .6 = 240. She has been going to the store for 80 weeks because 240 / 3 = 80. So the answer is 80

# Q: Jane, Kyla, and Anthony have summer jobs in a resort. Their task is to fold guests' towels. Jane can fold 3 towels in 5 minutes. Kyla can fold 5 towels in 10 minutes, and Anthony can fold 7 towels in 20 minutes. If they all fold towels together, how many towels can they fold in one hour?
# A: There are 1 x 60 minutes = 60 minutes in 1 hour. There are 60/5 = 12 sets of 5 minutes in 1 hour. So, Jane can fold 3 x 12 = 36 towels in an hour. There are 60/10 = 6 sets of 10 minutes in 1 hour. So, Kyla can fold 5 x 6 = 30 towels in an hour. There are 60/20 = 3 sets of 20 minutes in 1 hour. So, Anthony can fold 7 x 3 = 21 towels in an hour. Therefore, the 3 of them can fold a total of 36 + 30 + 21 = 87 towels in 1 hour. So the answer is 87

# Q: At Sunshine Orchard, there are 12 more than three times the number of pumpkins at Moonglow Orchard. If Moonglow Orchard has 14 pumpkins how many are there at Sunshine Orchard?
# A: Three times the number of pumpkins at Moonglow Orchard is 14*3= 42. Sunshine Orchard has 12+42= 54 pumpkins. So the answer is 54

# """
        # format test input
        question = test_input["question"]
        answer_choices = test_input["answer_choices"]
        formatted_test_input = self.format_example(question_template, answer_choices_template, label_template, question, answer_choices, 
                                                   label_str=None, cot_reason=None)
        prompt = formatted_examples + formatted_test_input
        # add empty label template to prompt if not doing CoT -- if doing CoT, the model will generate the text of the label template
        if not self.use_cot:
            label_template_no_label = label_template.format("")
            # avoid buffer space if label template starts with a line break
            if label_template_no_label[0] == '\n':
                prompt = prompt + label_template_no_label
            else:
                prompt = prompt + " " + label_template_no_label
        # remove trailing space if using a llama model
        if 'llama' in self.args.model.lower() and prompt[-1] == " ":
            prompt = prompt[:-1]
        # optionally add instructions
        if instructions != "":
            prompt = instructions + "\n" + prompt
        return prompt
    
    def standardize(self, point):
        return {
            'question': point.input_text,
            'answer_choices': point.answer_choices,
            'label': point.answer_choices[point.label_idx],
            'label_idx': point.label_idx,
            'reasoning': getattr(point, 'reasoning', None),
        }
    
    def format_prompt_from_df(self, test_input_df, 
                              examples_df=None,
                              dataname=None, promptsource_idx=None, 
                              instr_idx=None, question_idx=None, answers_idx=None, label_idx=None):
        '''
        format data from a pd df for passing to an LM tokenizer. 
        - uses default prompt templates if args are None
        - applies promptsource prompts if those were provided at init
        - intended for use inside Probe class
        - all_answers choices: return a list of prompts
        returns
            strings containing fully formatted prompt for passing to LM
        '''
        # first option is to use promptsource prompts
        if self.prompt_source == "promptsource":
            prompt = self.promptsource_dict[dataname][promptsource_idx]
            # format test input
            test_input_df = test_input_df.drop('answer_choices') # need to drop standardized 'answer_choices' column at this point...
            test_input = prompt.apply(test_input_df)[0] # does not include label in input
            # combine with examples
            if examples_df is not None:
                # need to drop standardized 'answer_choices' column at this point...
                q_and_a_s = [prompt.apply(example.drop('answer_choices')) for _, example in examples_df.iterrows()]
                formatted_examples = [f"{q} {a}" for q,a in q_and_a_s]
                prepend_examples = "\n".join(formatted_examples) + "\n"
            else:
                prepend_examples = ""
            full_prompt = prepend_examples + test_input
        # second option is to use our prompts
        elif self.prompt_source == "custom":
            # get template to use
            instr_idx = instr_idx if instr_idx is not None else self.default_instr_idx
            question_idx = question_idx if question_idx is not None else self.default_question_idx
            answers_idx = answers_idx if answers_idx is not None else self.default_answers_idx
            label_idx = label_idx if label_idx is not None else self.default_label_idx
            instructions = self.instructions[instr_idx]
            question_template = self.question_templates[question_idx]
            answer_template = self.answer_choices_functions[answers_idx]
            label_template = self.label_templates[label_idx]
            examples = []
            if examples_df is not None:
                for _, example in examples_df.iterrows():
                    assert hasattr(example, 'input_text'), "Need to standardize columns of this data to have input_text"
                    examples.append(self.standardize(example))
            test_input = self.standardize(test_input_df)
            full_prompt = self.format_prompt(instructions, question_template, answer_template, label_template, examples, test_input)
        return full_prompt

    def format_reasoning_target_from_df(self, test_input_df, 
                              dataname=None, promptsource_idx=None, 
                              instr_idx=None, question_idx=None, answers_idx=None, label_idx=None):
        '''
        formats the CoT reasons with a suffix according to the label template
        returns
            CoT ids fully formatted for passing as reasoning_chains to make_LM_batch
        '''
        # get template to use
        assert self.prompt_source == 'custom'
        label_template = self.label_templates[label_idx]
        label_template_no_label = label_template.format("")
        reasoning = test_input_df.reasoning
        reasoning_target = f"{reasoning} {label_template_no_label}"
        # remove trailing space if using a llama model
        if 'llama' in self.args.model.lower() and reasoning_target[-1] == " ":
            reasoning_target = reasoning_target[:-1]
        return reasoning_target