import pandas as pd
import matplotlib.pyplot as plt
import utils
import os

class TrainingLogger():
    """
    Report stores evaluation results during the training process as text files.
    """

    def __init__(self, args, file_path, experiment_name, overwrite_existing=True):
        self.fn = file_path
        self.args = args
        self.experiment_name = experiment_name
        self.max_len = 10
        self.old_running_time = 0
        self.curr_speed = 0
        self.columns = ["LR", "epoch", "train_loss", "dev_loss", "train_acc", "dev_acc"]
        self.log_records = []
        # make training_logs dir if not made yet
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)

    def add_to_log(self, stats):
        assert sorted(list(stats.keys())) == sorted(self.columns), f"please pass stats dict to log.add_to_log with keys: {self.columns} (found keys: {sorted(stats.keys())})"
        self.log_records.append(stats)
        self.log_df = pd.DataFrame.from_records(self.log_records)
        self.save_log()

    def get_last_eval_result(self):
        return self.log_df.iloc[-1]

    def save_log(self):
        self.log_df.to_csv(self.fn, index=False)

    def reset_log(self):
        self.log_records = []
        self.log_df = pd.DataFrame(columns=self.columns)

    def print_training_prog(self, train_stats, epoch, num_epochs, batch_num, 
        n_batches, running_time, est_epoch_run_time, forward_time, current_LR=None, gpu_mem=None):
        last_batch = batch_num == n_batches-1
        print_str = f" Epoch {epoch}/{num_epochs} | Batch: {batch_num+1}/{n_batches}"
        for k, v in train_stats.items():
            if k.lower() == 'loss' or 'acc' in k:
                print_str += f" | {k.capitalize()}: {v:.2f}"
        if current_LR is not None:
            print_str += f" | LR: {current_LR:.6f} | Runtime: {running_time/60:.1f} min. / {est_epoch_run_time/60:.1f} min. | Forward time: {forward_time:.3f} sec."
        else:
            print_str += f" | Runtime: {running_time/60:.1f} min. / {est_epoch_run_time/60:.1f} min. | Forward time: {forward_time:.3f} sec."
        if gpu_mem:
            print_str += f" | Mem: {gpu_mem}"
        print(print_str, end='\r' if not last_batch else '\n')

    def print_epoch_scores(self, epoch, scores):
        epoch_text = ' %6s ' % 'epoch'
        scores = {k:v for k,v in scores.items() if k not in ['n_batches', 'forward_time_sum', 'label_confidence']}
        for n, score_name in enumerate(scores.keys()):
            len_name = len(score_name)
            if len_name > self.max_len:
                score_name = score_name[:self.max_len]
            epoch_text += '| %10s' % score_name if 'acc' not in score_name else '| %11s' % score_name
        epoch_text += '\n %6s ' % str(epoch)
        for score_name, score in scores.items():
            if 'acc' in score_name:
                score *= 100
                epoch_text += '| %10s' % ('%3.2f' % score) + '%'
            elif not isinstance(score, list):
                epoch_text += '| %10s' % ('%1.2f' % score)
        print(epoch_text)

    def save_plots(self, n_train=None):
        # save plots of train_loss/eval_loss/eval_acc vs. steps/n_tokens/n_sentences/epochs
        # outcomes = ['train_loss', 'dev_loss', 'dev_acc']
        outcomes = ['train_acc', 'dev_acc', 'train_loss', 'dev_loss']
        x_vars = ['epoch']
        # make single y vs x plots
        # for outcome in outcomes:
        #     for x_var in x_vars:
        #         plot_name = f"plt_{self.args.dataset}_{self.experiment_name}_{outcome}_vs_{x_var}"
        #         plt.plot(self.log_df[x_var], self.log_df[outcome])
        #         plt.xlabel(x_var)
        #         plt.ylabel(outcome)
        #         plt.title(f"{outcome} vs {x_var}")
        #         # save the plot to a PDF file
        #         filepath = f'training_logs/{plot_name}.pdf'
        #         plt.savefig(filepath)
        #         plt.clf()

        # overlay the eval_acc, train_loss, and eval_loss variables
        for x_var in x_vars:
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            n_train_insert = f"_n-{n_train}" if n_train else ''
            plot_name = f"plt_{self.args.dataset}{n_train_insert}_{self.experiment_name}_results_by_{x_var}"
            for outcome in outcomes:
                if 'loss' in outcome:
                    ax1.plot(self.log_df[x_var], self.log_df[outcome], label=outcome)
                else:
                    ax2.plot(self.log_df[x_var], self.log_df[outcome], label=outcome)
            ax1.set_xlabel(x_var)
            ax1.set_ylabel("Loss", rotation=0)
            max_loss = 10 if self.log_df['dev_loss'].min() > 5 else 5
            ax1.set_ylim(0, max_loss)
            ax2.set_xlabel(x_var)
            ax2.set_ylabel("Acc", rotation=0)
            ax2.set_ylim(0, 1.04)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='best')
            fig.suptitle(f"Model Performance vs. {x_var}")

            # find the peak value of the curve
            peak_value = self.log_df['dev_acc'].max()
            peak_index = self.log_df['dev_acc'].idxmax()
            peak_x_val = self.log_df[x_var].iloc[peak_index]
            ax2.text(.5, 1.03, f'acc: {peak_value:.2f} (at x={int(peak_x_val)})',
                transform=ax2.transAxes, horizontalalignment='center')
            ax2.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
            ax2.axvline(x=peak_x_val, color='red', linestyle='--', linewidth=0.5)

            # save the plot to a PDF file
            filepath = f'training_logs/{plot_name}'
            # plt.savefig(filepath+'.pdf', format='pdf')
            plt.savefig(filepath+'.png', format='png')
            plt.clf()
            ax1.cla()
            ax2.cla()
            plt.close()