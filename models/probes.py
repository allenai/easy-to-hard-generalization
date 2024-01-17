import torch
from torch import nn
import numpy as np
import pandas as pd
import os
import time
import gc

import bitsandbytes as bnb
from transformers import AutoConfig, BertModel
from transformers import get_scheduler
from torch.optim import SGD, AdamW, Adam, LBFGS
from torch.utils.data import Dataset, DataLoader

from peft import AutoPeftModelForCausalLM

from utils import utils
from utils import LM_utils
from utils import modeling_utils
from utils import data_utils
import copy

class Probe(nn.Module):
    '''
    This is a class for handling both few-shot prompting and supervised probing of LLMs.
    - initialized with a dataset of text inputs, multiple-choice answers, and labels
    - initialized with a Prompt object for formatting the data
    '''
    def __init__(self, args, probing_method, probe_loss,
                 tokenizer,
                 probing_config=None,
                 normalize_representations=False, calibrate_probe=False,
                 model=None, num_classes=None):
        '''
        args:
            args is from argparse in main.py
            datasets is a nested dict of {dataname: dataset}, where dataset contains pd dfs with inputs, multiple-choice answers, and labels.
            prompt is Prompt object from prompt.py
            normalize_representations: z-normalize hidden states used with a parametric probe. see main.py args.normalize_representations
            num_fits: number of times to fit the classifier. used with CCS to select the run with lowest unsupervised loss
        '''
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.normalize_representations = normalize_representations
        self.calibrate_probe = calibrate_probe # kind of a misnomer, this adjusts model preds to be mostly uniform across classes based on a provided dataset
        self.model = model
        self.num_classes = num_classes # num classes when doing classification of question_end hidden state
        # default params, reset by .fit() but used in self.loss, which may be called for prompt selection purposes with ICL
        self.prior_reg, self.prior = 0, 0
        self.l2_reg = 0 if args.optimize_weights != 'ABCD_embeddings' else args.l2_reg
        if args.probing_token_state == 'question_end_token':
            self.l2_prior = self.get_ABCD_embeddings(tokenizer, 'lm_head') # used when doing a regression on question_end_token hidden states
        else:
            self.l2_prior = None
        # set probing strategy
        self.probing_method = probing_method
        # normalization variables, set from train data
        self.mean_for_norming = None
        self.std_for_norming = None
        # calibration param making predicted distribution uniform in aggregate
        self.probs_centroid = None
        # for unsupervised probing
        self.probs_mean = None # will be 'true' or 'false', determine whether pred = argmax(probs) or argmin(probs)
        # probe_loss is used for prompt selection and fitting probes, so it applies to both ICL and probing
        self.probe_loss = probe_loss
        # probing args
        self.probing_config = probing_config
        if self.probing_method == "learned":
            self.probe_model = probing_config['probe_model']
            self.hidden_size = probing_config['hidden_size']
            # extend hidden size by the number of layers we pull representations, and double the size if we pull both encoder and decoder reps
            self.hidden_size *= len(probing_config['features_layers'])
            if len(probing_config['features_enc_dec']) == 2:
                self.hidden_size *= 2
            print(f" probe feature dimensionality is {self.hidden_size}")
            if self.probe_model == 'linear' and args.probing_token_state == 'answer_end_token': 
                probe_model = nn.Linear(in_features=self.hidden_size, out_features=1, bias=False)
            if self.probe_model == 'linear' and args.probing_token_state == 'question_end_token': 
                probe_model = nn.Linear(in_features=self.hidden_size, out_features=self.num_classes, bias=True)
            if self.probe_model == 'MLP': 
                MLP_hidden_size = probing_config['hidden_size'] # use orig model hidden size (as opposed to self.hidden_size) to control parameter growth here
                probe_model = MLP(MLP_hidden_size, in_features=self.hidden_size, dropout_prob=0, num_classes=1)
            if self.probe_model == 'transformer': 
                transformer_config = AutoConfig.from_pretrained('bert-base-uncased', cache_dir=args.cache_dir)
                transformer_config.hidden_size = self.hidden_size
                transformer_config.num_hidden_layers = 1
                transformer_config.hidden_dropout_prob = 0
                MLP_hidden_size = probing_config['hidden_size'] # use orig model hidden size (as opposed to self.hidden_size) to control parameter growth here
                probe_model = TransformerProbe(transformer_config, MLP_hidden_size=MLP_hidden_size, in_features=self.hidden_size)
            proper_normalization = (self.probe_loss != 'CCS') # needed for exact CCS replication
            if args.probing_token_state == 'answer_end_token':
                self.probe = MultipleChoiceClassifier(probe_model, 
                                                    proper_normalization=proper_normalization)
            elif args.probing_token_state == 'question_end_token':
                self.probe = LinearClassifier(probe_model, 
                                              num_classes=self.num_classes)
            if self.args.n_gpu > 0:
                self.probe = self.probe.cuda()

    def set_calibration_params(self, probs=None, dataloader=None, verbose=False):
        '''
        Used to calibrate predictions to be uniformly distributed over label space
        '''
        if verbose:
            print("Calibrating probabilities to be uniform over classes...")
        if dataloader is not None:
            all_probs = []
            for batch in dataloader:
                with torch.no_grad():
                    probs = self.forward(batch) # compute probs here
                    all_probs.append(probs.detach().cpu())
            all_probs = torch.concatenate(all_probs)
        if probs is not None:
            all_probs = probs
        all_preds = torch.argmax(all_probs, dim=1)
        probs_centroid, _ = torch.median(all_probs, dim=0)
        new_probs = all_probs - probs_centroid
        all_new_preds = torch.argmax(new_probs, dim=1)
        old_pred_distr = {y: round(torch.mean((all_preds==y).float()).item(), 2) for y in set(all_preds.cpu().numpy())}
        new_pred_distr = {y: round(torch.mean((all_new_preds==y).float()).item(), 2) for y in set(all_new_preds.cpu().numpy())}
        if verbose:
            print("Old pred distr: ", old_pred_distr)
            print("New pred distr: ", new_pred_distr)
        self.probs_centroid = probs_centroid.cuda()

    def set_normalization_params(self, dataloader):
        states = []
        for batch in dataloader:
            states.append(batch['precomputed_hidden_states'])
        states = torch.concatenate(states, dim=0) 
        states = self.select_hidden_states(states) # shape: n_items x n_answers x self.hidden_size
        # if dataset has fixed label space, make per-label norming params
        if self.args.data_source == 'burns':
            self.mean_for_norming = torch.mean(states, dim=0, keepdim=True)
            self.std_for_norming = torch.std(states, dim=0, keepdim=True)
        # otherwise, for multiple-choice problems, share information across answer choices
        else:
            states = states.view(-1, self.hidden_size) # first collapse the n_items and answer choices dimensions
            self.mean_for_norming = torch.mean(states, dim=0, keepdim=True)
            self.std_for_norming = torch.std(states, dim=0, keepdim=True)

    def safe_fit(self, args, log, dataloader, optimizer_name, epochs=100, l2_reg=1, max_grad_norm=1, prior=None, verbose=False, patience=5):
        # sometimes fit gives nan result, so this refits the model if the final model has nan weightss
        done = False
        counter = 0 
        while not done:
            loss = self.fit(args, log, dataloader, optimizer_name, epochs=epochs, l2_reg=l2_reg, max_grad_norm=max_grad_norm, prior=prior, verbose=verbose)
            if not utils.check_nan_weights(self.probe):
                done = True
            elif counter > patience:
                done = True
                print(f"WARNING: COULD NOT FIT MODEL IN LESS THAN {patience} ATTEMPTS")
            else:
                print(f"WARNING: MODEL FAILED TO FIT. RETRYING...")
                counter += 1
                max_grad_norm /= 4 # cut grad clipping size
                l2_reg *= 4 # large increase to regularization
        return loss

    def repeated_fit(self, args, log, dataloader, optimizer_name, num_fits, prior=None, epochs=100, l2_reg=1, max_grad_norm=1,
                     verbose=False, safe_fit=True):
        # fits a probe num_fits times to dataset, and selects model with best loss
        # for use with CCS
        best_loss = np.inf
        losses = []
        for _ in range(num_fits):
            if safe_fit:
                loss = self.safe_fit(args, log, dataloader, optimizer_name, epochs=epochs, l2_reg=l2_reg, max_grad_norm=max_grad_norm, prior=prior, verbose=verbose)
            else:
                loss = self.fit(args, log, dataloader, optimizer_name, epochs=epochs, l2_reg=l2_reg, max_grad_norm=max_grad_norm, prior=prior, verbose=verbose)
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                prob_meaning = self.probs_mean
            losses.append(loss)
            best_loss = np.argmin(losses)
        losses = sorted(losses)
        self.probe = self.best_probe
        self.probs_mean = prob_meaning
        del self.best_probe
        # print(f"Selecting probe with loss {best_loss} from {num_fits} fits (losses: {losses})")

    def fit(self, args, log, dataloader, optimizer_name, epochs=100, 
            l2_reg=1, max_grad_norm=1, prior=None,
            verbose=False):
        '''
        args
            dataloader: MCTextDataset dataloader
                - ideally includes precomputed hidden states, but we recompute as needed here
            epochs: number of epochs to run over data
        '''
        assert self.probing_method == 'learned', "do not fit a probe if using decoding probing rather than learned probe"
        assert epochs > 0, "Epochs<=0 passed to probe.fit"
        num_answers = data_utils.get_max_num_answers(dataloader.dataset.dataframe)
        if verbose:
            print(f"Fitting probe to data...", end='\r')
        self.l2_reg = l2_reg # used in self.loss
        self.prior = prior # used in self.loss when self.probe_loss == 'unsupervised'
        self.prior_reg = 1
        self.probs_mean = None # indicates whether p(answer) is prob answer is true or prob answer is false
        forward_time = 0
        backward_time = 0
        n_steps = 0
        loss_history = []
        acc_history = []
        mem_history = []
        # re-initialize probe and .train() -- temporarily swap torch seed from args.seed to the provided seed (which should be inherited from boot_idx)
        # THIS IS WHERE WE SET THE PRIOR FOR REGRESSION WEIGHTS TO ABCD EMBEDDINGS WHEN APPLICABLE
        self.probe.apply(self.weights_init)
        self.probe.train()
        # if doing a random probe, do one epoch just to get an accuracy, but don't step
        # random probes will get "flipped" to have accuracy >= .5, for fair comparison with CCS
        if self.probe_loss == 'random':
            epochs = 1

        # compute hidden_states if they are not precomputed
        precomputed_hidden_states = dataloader.dataset.precomputed_hidden_states
        if precomputed_hidden_states is None and self.probing_method == 'learned':
            mc_dataset = dataloader.dataset
            print(f"\nPrecomputing hidden representations for dataset of size {len(mc_dataset)}...")
            bs = 32 // num_answers
            unshuffled_dataloader = DataLoader(mc_dataset, shuffle=False, collate_fn=mc_dataset.collate_fn, pin_memory=False, num_workers=0, batch_size=bs)
            eval_output = modeling_utils.evaluate_model(self.args, log, self.model, unshuffled_dataloader, tokenizer=None, num_answers=num_answers, 
                                                                      return_hidden_states=True)
            hidden_states = eval_output['hidden_states']
            # assign hidden states to main dataloader
            dataloader.precomputed_hidden_states = hidden_states

        # define optimizer and scheduler
        params = self.probe.parameters()
        num_training_steps = epochs * len(dataloader)
        if self.probing_config['probe_model'] == 'linear':
            lr = 5e-2
        else:
            raise NotImplementedError("Check LRs for more expressive probes than linear")
        if optimizer_name == 'sgd':
            optimizer = SGD(params, lr=lr)
        if optimizer_name == 'adamw':
            optimizer = AdamW(params, lr=lr)
        if optimizer_name == 'adam':
            optimizer = Adam(params, lr=lr)
        if optimizer_name == 'LBFGS':
            optimizer = LBFGS(params, max_iter=1, history_size=10)
        if optimizer_name != 'LBFGS':
            lr_decay = self.probing_config['lr_decay']
            if lr_decay == "constant":
                scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
            elif lr_decay in ['linear', '10-percent']:
                percent_of_orig_value = .1 if lr_decay == '10-percent' else 0
                multiplier = 1 / (1-percent_of_orig_value)
                scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=multiplier*num_training_steps)

        # prepare for normalizing representations as requested (see self.forward)
        if self.normalize_representations:
            self.set_normalization_params(dataloader)

        # pre-emptively unpack dataloader if it contains only one batch
        if len(dataloader) == 1:
            dataloader = [batch for batch in iter(dataloader)]
        start_fit_time = time.time()
        for e in range(epochs):
            losses = []
            all_preds = []
            all_labels = []
            for batch in dataloader:
                labels = batch['label_idx']
                if self.args.n_gpu > 0:
                    labels = labels.cuda()
                with torch.enable_grad():
                    start = time.time()
                    probs = self.forward(batch) # compute probs here
                    forward_time += time.time() - start
                    # compute loss and step. split by optimizer type
                    if optimizer_name == 'LBFGS':
                        def closure(): 
                            optimizer.zero_grad()
                            loss = self.loss(labels, probs)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.probe.parameters(), max_grad_norm)
                            return loss
                        start = time.time()
                        if self.probe_loss != 'random':
                            optimizer.step(closure)
                            loss = self.loss(labels, probs).detach()
                        backward_time += time.time() - start
                    else:
                        loss = self.loss(labels, probs)
                        if self.probe_loss != 'random':
                            start = time.time()
                            optimizer.zero_grad()
                            loss.backward()
                            # loss.backward(retain_graph=(self.args.probing_token_state=='question_end_token'))
                            torch.nn.utils.clip_grad_norm_(self.probe.parameters(), max_grad_norm)
                            backward_time += time.time() - start
                            optimizer.step()
                            scheduler.step() # only step on non-lbfgs
                n_steps += 1
                preds = torch.argmax(probs, dim=-1)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
                losses.append(loss.item())
                del loss, probs
            # compute acc
            acc = np.mean(np.array(all_labels)==np.array(all_preds))
            loss_history.append(round(np.mean(losses),2))
            acc_history.append(round(acc,2))
            gpu_mem = utils.get_gpu_utilization() if "cuda" in str(args.device) else None
            mem_history.append(gpu_mem)
        # if doing unsupervised, set whether probs for an answer choice represent prob true or prob false
        # this is some amount of supervision...Burns suggests you could do this step in an unsupervised way
        if self.probe_loss != 'supervised':
            self.probs_mean = 'false' if acc < .5 else 'true'
            acc_history = np.array(acc_history)
            acc_history = 1-acc_history if self.probs_mean == 'false' else acc
        if verbose:
            print(f"Fitting probe to data...took {(time.time() - start_fit_time):.2f} seconds", end='\n')
            # print(f"Loss history: ", loss_history)
            # print(f"Acc history: ", acc_history)
            if len(loss_history) == 0:
                loss_history.append(-1)
        self.probe.eval()
        return loss_history[-1]

    def finetune(self, args, log, train_dataloader, tokenizer, epochs=100, grad_accumulation_factor=1,
                 dev_dataloader=None, eval_every_n_epochs=None, 
                 model_selection='NA',
                 verbose=False):
        '''
        For full model finetuning, or parameter-efficient finetuning
        args
            dataloader: MCTextDataset dataloader
            epochs: number of epochs to run over data
            model_selection: pick best model epoch based on this statistic in log_stats
            break_after_e_epochs: break early after a selected number of epochs
        '''
        assert self.probing_method == 'finetuned'
        num_batches = len(train_dataloader)
        num_items = len(train_dataloader.dataset)
        num_training_steps = epochs * int(np.ceil(num_batches / grad_accumulation_factor))
        best_acc = -1
        # set tmp save/load path
        if model_selection != 'NA':
            tmp_save_load_path = os.path.join(args.model_dir, 'tmp')
        if verbose:
            effective_num_answers = train_dataloader.dataset.get_effective_num_answers()
            print(f"Fitting probe to data...", end='\r')
            print(f"Epochs: {epochs} | Num items: {num_items} | Num answers: {effective_num_answers} | Num points {num_items*effective_num_answers}")
            print(f"Batch size: {args.train_batch_size} | Batches per epoch: {num_batches} | Total opt steps: {num_training_steps}")
            n_items_per_batch = train_dataloader.batch_size
            n_batches_per_step = grad_accumulation_factor
            print("NUM ITEMS PER GRADIENT STEP:",  n_items_per_batch * n_batches_per_step)
        self.l2_reg = 0 # used in self.loss
        compute_mc_probs = args.finetuning_objective == 'MC'
        train_stats = {
            'n_batches': 1,
            'forward_time_sum' : 0,
            'backward_time_sum' : 0,
            'acc': -1,
            'loss': -1,
        }
        total_batches = len(train_dataloader)
        self.model.train()
        # define optimizer and schedules
        if self.args.optimize_weights in ['all', 'LORA']:
            decay_parameters = [name for name, p in self.model.named_parameters() if 'layernorm' not in name.lower() and "bias" not in name.lower()]
            params = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
        elif self.args.optimize_weights == 'embeddings':
            embed_param_names = [n for n,p in self.model.named_parameters() if 'embed' in n or 'lm_head' in n]
            print(" Optimizing these params: ", embed_param_names)
            params = [p for n,p in self.model.named_parameters() if n in embed_param_names]
        elif self.args.optimize_weights == 'ABCD_embeddings':
            embed_param_names = [n for n,p in self.model.named_parameters() if 'embed' in n or 'lm_head' in n]
            assert len(embed_param_names) > 0, f"Couldn't find the lm_head params for {self.args.model}"
            print(" Optimizing these params: ", embed_param_names)
            params = [p for n,p in self.model.named_parameters() if n in embed_param_names]
            ABCD_token_ids = [tokenizer.encode(x, add_special_tokens=False)[0] for x in ['A', 'B', 'C', 'D']]
            ABCD_rows = np.array(ABCD_token_ids)
            zero_grad_rows = np.setdiff1d(np.arange(len(self.tokenizer)), ABCD_rows)
            zero_grad_rows = torch.tensor(zero_grad_rows).cuda()
        lr = args.probing_lr
        if args.quantization == '8bit':
            optimizer = bnb.optim.Adam8bit(params, lr=lr)
        else:
            optimizer = AdamW(params, lr=lr)
        # get scheduler
        lr_decay = self.probing_config['lr_decay']
        if lr_decay == "constant":
            scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        elif lr_decay in ['linear', '10-percent']:
            percent_of_orig_value = .1 if lr_decay == '10-percent' else 0
            multiplier = 1 / (1-percent_of_orig_value)
            scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=multiplier*num_training_steps)

        # start training
        start_fit_time = time.time()
        # pre-emptively unpack dataloader if it contains only one batch
        if len(train_dataloader) == 1:
            train_dataloader = [batch for batch in iter(train_dataloader)]
        for e in range(1, epochs+1):
            epoch_stats = {
                'acc_sum': 0,
                'loss_sum': 0,
                'probe_loss_sum': 0, # used for model selection, may be unsupervised objective
                'n_data_points': 0,
            }
            for batch_num, batch in enumerate(train_dataloader):
                running_time = (time.time()-start_fit_time)
                est_run_time = (running_time/train_stats['n_batches']*total_batches*epochs)
                forward_time = train_stats['forward_time_sum'] / train_stats['n_batches']
                if verbose:
                    gpu_mem = utils.get_gpu_utilization() if "cuda" in str(args.device) else None
                    log.print_training_prog(train_stats, e, epochs, batch_num, len(train_dataloader), running_time, est_run_time, forward_time, gpu_mem=gpu_mem)
                labels = batch['label_idx'].cuda()
                with torch.enable_grad():
                    start = time.time()
                    probs = self.forward(batch, compute_mc_probs=compute_mc_probs) # compute probs here
                    probs = probs.cuda() # syncs probs with labels in multi-gpu case
                    train_stats['forward_time_sum'] += time.time() - start
                    # compute loss, scale by batch size, and step. batch size scaling keeps grad norms consistent when last batch size is smaller than others
                    bs_as_frac_of_max_bs = batch['input_ids'].size(0) / args.train_batch_size
                    labels = labels.cpu()
                    probs = probs.cpu()
                    loss = self.loss(labels, probs) / grad_accumulation_factor * bs_as_frac_of_max_bs
                    start = time.time()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                    train_stats['backward_time_sum'] += time.time() - start
                    if train_stats['n_batches'] % grad_accumulation_factor == 0:
                        # zero non A/B/C/D token embedding rows if necessary
                        if args.optimize_weights == 'ABCD_embeddings':
                            self.model.lm_head.weight.grad[zero_grad_rows,:] = 0
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step() 
                # end of epoch stats
                epoch_stats['loss_sum'] += loss.item()
                epoch_stats['n_data_points'] += len(batch['items'])
                if compute_mc_probs:
                    preds = torch.argmax(probs, dim=-1)
                    binary_correct = preds==labels
                    n_correct = torch.sum(binary_correct).item()
                    epoch_stats['acc_sum'] += n_correct
                    train_stats['acc'] = epoch_stats['acc_sum'] / epoch_stats['n_data_points']
                # update eval stats
                train_stats['loss'] = epoch_stats['loss_sum'] / (batch_num+1)
                train_stats['probe_loss'] = epoch_stats['probe_loss_sum'] / (batch_num+1)
                train_stats['n_batches'] += 1
                # print examples
                if verbose:
                    if (batch_num == 0 and e == 1 and args.num_print > 0): 
                        print_idx = list(range(min(args.num_print, len(batch['items']))))
                    else:
                        print_idx = []
                    if len(print_idx) > 0:
                        print("\n" + "-"*20 + f"\nPrinting examples:")
                        if e == 1:
                            print(f" Input 0     : {tokenizer.decode(batch['input_ids'][0])}")    
                        for i in print_idx:
                            answer_choices = ['A', 'B', 'C', 'D'] if args.use_letter_labels else batch['answers_list'][i]
                            prompt = batch['prompts'][i]
                            print(f" point {i}")
                            print(f" Prompt      : \n{prompt}")
                            if compute_mc_probs: 
                                item_probs = [np.round(x.item(), 4) for x in probs[i].cpu()]
                                print(f" Preds       : {[x for x in zip(answer_choices, item_probs)]}")
                                pred = answer_choices[preds[i]]
                                print(f" Pred        : {pred}")
                                print(f" Label       : {batch['label_strs'][i]}")
                                print(f" Correct     : {binary_correct[i].item()}")
                            if i != print_idx[-1]:
                                print()
                        print("-"*20 + '\n')
                del loss, probs, batch
            # eval model on dev data
            if eval_every_n_epochs > 0 and e % eval_every_n_epochs == 0:
                print(" Evaluating model...")
                dev_stats = modeling_utils.evaluate_model(args=args,
                        log=log,
                        model=self,
                        dataloader=dev_dataloader, 
                        tokenizer=tokenizer,
                        verbose=verbose)
                self.model.train()
                log_stats = {
                    'LR': scheduler.get_last_lr()[0], # two param groups...just take one,
                    'epoch': e,
                    'train_loss': train_stats['loss'],
                    'train_acc': train_stats['acc'] if compute_mc_probs else -1,
                    'dev_loss': dev_stats['loss'],
                    'dev_acc': dev_stats['acc'],
                }
                log.print_epoch_scores(epoch=e, scores=log_stats)
                log.add_to_log(log_stats)
                log.save_plots(n_train=num_items)
            # save best model
            if model_selection != 'NA':
                if model_selection == 'train_acc':
                    select_acc = train_stats['acc']
                elif model_selection == 'dev_acc':
                    assert eval_every_n_epochs >= 1
                    dev_stats_just_calculated = (e % eval_every_n_epochs == 0)
                    select_acc = dev_stats['acc'] if dev_stats_just_calculated else -1
                if select_acc > best_acc:
                    best_acc = select_acc
                    print(f"Saving new best probe at: ", tmp_save_load_path)
                    save_start = time.time()
                    self.save_probe(tmp_save_load_path)
                    print("Saving probe took: ", utils.format_time(time.time()-save_start))
        if verbose:
            print(f"Fitting model to data...took {(time.time() - start_fit_time):.2f} seconds", end='\n')
        if args.dev_eval_every_epochs > 0:
            log.reset_log()
        self.model.eval()
        # load best model
        if model_selection != 'NA':
            self.load_probe(tmp_save_load_path)

    def loss(self, Y, probs):
        # assumes probs of shape n x m, and Y of shape n containing label ids
        if self.probe_loss == 'supervised':
            label_probs = torch.gather(probs, 1, Y.view(-1, 1))
            nll = -torch.log(label_probs).mean()
            l2_norm = 0
            if self.probing_method == 'learned':
                if self.l2_prior is None:
                    for param in self.probe.parameters():
                        l2_norm += torch.linalg.vector_norm(param)
                else:
                    assert self.probing_config['probe_model'] == 'linear'
                    # l2_norm += torch.linalg.norm(self.l2_prior - self.probe.probe_model.weight)
                    l2_norm += self.probe.probe_model.weight.norm()
                    print(l2_norm)
            return nll + self.l2_reg * l2_norm
        elif self.probe_loss in ['LM_loss']:
            nll = -probs.mean() # assuming that finetuning_objective=seq2seq, the probs are already loglikes here
            return nll
        elif self.probe_loss in ['CCS', 'CCS_ours', 'random']: # 'random' condition never actually steps optimizer
            min_probs, _ = torch.min(probs, dim=1)
            informative_loss = (min_probs**2).mean(0)
            consistent_loss = ((1 - probs.sum(1))**2).mean() # always 0 when MC classifier uses proper_normalization=True
            return informative_loss + consistent_loss
        elif self.probe_loss == 'unsupervised':
            max_probs, _ = torch.max(probs, dim=1)
            confidence_loss = -torch.log(max_probs).mean()
            prior_loss = (max_probs.mean() - self.prior)**2 # this is a calibration loss
            l2_norm = 0
            if self.probing_method == 'learned':
                for param in self.probe.parameters():
                    l2_norm += torch.linalg.vector_norm(param)
            return confidence_loss + self.prior_reg * prior_loss + self.l2_reg * l2_norm

    def get_ABCD_embeddings(self, tokenizer, embeds_name='lm_head'):
        assert self.model is not None
        ABCD_token_ids = np.array([tokenizer.encode(x, add_special_tokens=False)[0] for x in ['A', 'B', 'C', 'D']])
        params = [p for n,p in self.model.named_parameters() if embeds_name in n]
        assert len(params) == 1, f"Looking for the {embeds_name} params for {self.args.model} gave != 1 matching value"        
        embeds = params[0]
        return embeds[ABCD_token_ids]
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            if self.args.probing_token_state == 'question_end_token':
                init_weights = self.get_ABCD_embeddings(self.tokenizer, 'lm_head')
                init_weights = init_weights.to(torch.float32)
                m.weight = torch.nn.Parameter(init_weights)
            else:
                m.reset_parameters()
        else:
            if not self.probing_config['probe_model'] in ['linear', 'MLP']:
                print("not re-initializing: ", m)
                import pdb; pdb.set_trace()

    def forward(self, batch, compute_mc_probs=True):
        '''
        branch function output based on self.probing_method
        '''
        if self.probing_method in ['decoding', 'finetuned']:
            assert self.model is not None, "if decoding to score answers, model must be provided to Probe at init"
            main_kwargs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'labels': batch['input_ids'],
                'targets_mask': batch['targets_mask'],
                'answer_choices': batch['answer_choices'],
            }
            utils.move_kwargs_to_gpu(main_kwargs)
            if compute_mc_probs:
                num_answers_list = batch['num_answers_list']
                assert not all(x==1 for x in num_answers_list), "Trying to compute mc probs but num_answers are all 1"
            else:
                num_answers_list = None
            probs = LM_utils.compute_probs_from_batch(self.model, 
                                                      main_kwargs, 
                                                      return_value=self.args.answer_scoring, 
                                                      num_answers_list=num_answers_list)
        if self.probing_method == 'learned':
            # first get hidden states. may compute LLM forward pass as needed
            if 'precomputed_hidden_states' in batch:
                hidden_states = batch['precomputed_hidden_states'] # shape: n_items x n_answers x enc_dec x num_layers x hidden_size
            else:
                assert self.model is not None, "if computing hidden states for data on the fly, model must be provided to Probe at init"
                main_kwargs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                    'labels': batch['input_ids'],
                    'targets_mask': batch['targets_mask'],
                }
                utils.move_kwargs_to_gpu(main_kwargs)
                _, hidden_states_dict = LM_utils.compute_probs_from_batch(self.model, main_kwargs, return_value='probs', return_hidden_states=True)
                hidden_states = LM_utils.get_last_token_hidden_states(hidden_states_dict, 
                                                                      max_num_answers = max(batch['num_answers_list']),
                                                                      num_answers_list=batch['num_answers_list'])
                hidden_states = torch.tensor(hidden_states)
            # select hidden states based on probing feature space config
            hidden_states = self.select_hidden_states(hidden_states)
            # normalize using params obtained during .fit()
            if self.normalize_representations:
                assert self.mean_for_norming is not None, "need to call probe.set_normalization_params before applying this probe"
                hidden_states = (hidden_states - self.mean_for_norming) / self.std_for_norming
            # now send to gpu and do forward pass
            if self.args.n_gpu > 0:
                hidden_states = hidden_states.cuda()
            probs = self.probe(hidden_states)
        # flip and renormalize probs if prob = prob(false) rather than prob(true)
        if self.probs_mean == 'false':
            flip_probs = 1 - probs
            probs = flip_probs / torch.sum(flip_probs, dim=-1, keepdim=True)
        if self.calibrate_probe and self.probs_centroid is not None:
            probs = probs - self.probs_centroid
            min_probs, _ = torch.min(probs, dim=-1, keepdim=True)
            # this step recalibrates the predicted label distribution to be near uniform (see self.set_calibration_params)
            # but the probabilities themselves have to be artificially renormed+smoothed, which we do in a very arbitrary way
            probs = probs - min_probs # first make non-negative
            probs = probs + .01 # artificial choice for smoothing
            probs = probs / torch.sum(probs, dim=1, keepdim=True) # renormalize
        return probs
    
    def select_hidden_states(self, hidden_states):
        # select hidden states before moving to gpu
        enc_dec_idx = []
        for component in self.probing_config['features_enc_dec']:
            if component == 'decoder':
                enc_dec_idx.append(0)
            if component == 'encoder':
                enc_dec_idx.append(1)
            enc_dec_idx = torch.tensor(enc_dec_idx)
        layer_idx = torch.tensor([int(x) for x in self.probing_config['features_layers']])
        hidden_states = torch.index_select(hidden_states, 2, enc_dec_idx) # this indexing keeps indexed dimension
        hidden_states = torch.index_select(hidden_states, 3, layer_idx) # this indexing keeps indexed dimension
        # reshape to n_items x n_answer x probe_hidden_size
        num_items = hidden_states.size(0)
        num_answers = hidden_states.size(1)
        hidden_states = hidden_states.view(num_items, num_answers, self.hidden_size)
        return hidden_states
    
    def save_probe(self, save_path):
        if self.probing_method == 'learned':
            state_dict = self.probe.state_dict()
            torch.save(state_dict, save_path)
        elif self.probing_method == 'finetuned':
            self.model.save_pretrained(save_path)

    def load_probe(self, load_path):
        '''
        This is for loading lightweight probing classifiers or LORA weights
        '''
        assert os.path.exists(load_path), f"Trying to load state dict from {load_path} but does not exist"
        if self.probing_method == 'learned':
            state_dict = torch.load(load_path)
            self.probe.load_state_dict(state_dict, strict=True)
            self.probe.eval()
        elif self.probing_method == 'finetuned':
            # delete the existing model before loading new one
            # if self.args.optimize_weights != 'LORA':
            for x in gc.get_referrers(self.model):
                del x
            del self.model
            self.model = utils.load_model(self.args, load_path)
            self.model.eval()

class MultipleChoiceClassifier(nn.Module):
    '''
    Expects data of shape n x m x d for n samples, m answer choices, and d dimensional features
    Returns probs in forward pass, of shape n x m
    '''
    def __init__(self, probe_model, proper_normalization=True):
        super().__init__()
        self.probe_model = probe_model
        self.proper_normalization = proper_normalization

    def forward(self, X):
        # assume X of shape n x m x d -- n items with m answer choices, d representation size
        scores = self.probe_model(X)
        scores = scores.view(X.size(0), X.size(1)) # drop last hidden dimension, which is now 1
        if self.proper_normalization:
            probs = torch.softmax(scores, dim=-1)
        else:
            probs = torch.sigmoid(scores) # used to exactly replicate CCS
        return probs

class LinearClassifier(nn.Module):
    '''
    Expects data of shape n x d for n samples with d dimensional features
    Returns probs in forward pass, of shape n x num_classes
    '''
    def __init__(self, probe_model, num_classes):
        super().__init__()
        self.probe_model = probe_model
        self.num_classes = num_classes

    def forward(self, X):
        # assume X of shape n x d -- n items, d representation size
        scores = self.probe_model(X)
        scores = scores.view(X.size(0), self.num_classes) # drop last hidden dimension, which is now 1
        probs = torch.softmax(scores, dim=-1)
        return probs


class MLP(nn.Module):
    def __init__(self, hidden_size, in_features=None, dropout_prob=.1, num_classes=2):
        super().__init__()
        if in_features is None:
            in_features = hidden_size
        self.classifier = nn.Sequential(
                                    nn.Linear(in_features, hidden_size),
                                    nn.Tanh(),
                                    nn.Dropout(p=dropout_prob),
                                    nn.Linear(hidden_size, num_classes),
                                )
    def forward(self, hidden_states, **kwargs):
        return self.classifier(hidden_states)
    
class IndexingModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, hidden_states, **kwargs):
        return hidden_states[:,-1,...]

class TransformerProbe(nn.Module):
    def __init__(self, config, num_answers):
        super().__init__()
        transformer = BertModel(config=config) # random weights
        self.indexing = IndexingModule()
        self.transformer = transformer
        self.classifier = MLP(config.hidden_size, config.hidden_dropout_prob, num_classes=1)
    def forward(self, hidden_states, **kwargs):
        outputs = self.transformer(inputs_embeds=hidden_states, attention_mask=kwargs['attention_mask'])
        last_index_rep = self.indexing(outputs.hidden_states[-1])
        scores = self.classifier(last_index_rep)
        return scores