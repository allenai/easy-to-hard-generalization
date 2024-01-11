import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import seaborn as sns
import PIL
from matplotlib.transforms import Affine2D

def plot_corr_matrix(corr_matrix, save_name, title):
    plt.figure(figsize=(10, 8))  # Set the size of the figure
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm_r', fmt=".2f")  # Create the heatmap with annotations
    plt.title(title)  # Add a title
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    # plt.yticks(rotation=45)  # Rotate the y-axis labels for better readability
    ax = plt.gca()
    labels = ax.get_xticklabels()
    # Create an offset transform
    dx = -1.5  # shift by -0.5
    offset = Affine2D().translate(dx, 0)
    # Apply the offset transform to each label
    for label in labels:
        label.set_transform(label.get_transform() + offset)
    filepath = f'result_sheets/{save_name}'
    plt.tight_layout()
    plt.savefig(filepath + '.png', format='png', dpi=300)
    plt.clf()

def grid_arrange_pngs(pngs, save_name):
    # takes a list of png plots opened as PIL.Image.open(x), and arranges them in a grid
    n_cols = 4
    n_rows = int(np.ceil(len(pngs) / n_cols))  # Ceil division
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(9, 9), dpi=300)
    axes_flat = axes.flatten()

    # Loop through each subplot and each filepath to display images
    for ax, img in zip(axes_flat, pngs):
        ax.imshow(np.array(img), interpolation='bilinear')
        ax.axis('off')  # Hide axes

    # remove empty subplots
    for remaining_ax in axes_flat[len(pngs):]:
        remaining_ax.axis('off')

    # Display the grid of images
    plt.show()
    filepath = f'result_sheets/{save_name}'
    plt.tight_layout()
    plt.savefig(filepath + '.png', format='png', dpi=300)
    plt.clf()
    

def compare_method_learning_curves(df_results, save_prefix, hardness_var_name, test_split='hard'):
    '''
    To be used on the results of experiments in run_jobs.py (applied to the resulting dfs).
    This function plots the learning curves of different prompting/probing methods, for a particular test split of the data
    args:
        test_split: show test_acc for df rows where test_on==test_split
    '''
    save_dir = 'outputs'
    df_results = df_results.copy() # suppress slicing warnings
    cot_col = df_results['use_cot'].apply(lambda x: '-CoT' if bool(x) else '')
    sup_col = df_results['probe_loss'].apply(lambda x: '-unsup' if 'unsup' in x else '')
    df_results['method'] = df_results['probing_method'] + cot_col + '-' + df_results['train_on'].astype(str) + sup_col
    max_n = 400
    # subset df
    subset = df_results[df_results['test_on'] == test_split]
    subset = subset[subset['n_train'] <= max_n]
    sns.set(style='whitegrid')
    plt.figure(figsize=(14, 7))
    # add error bar
    subset['lower_bound'] = subset['test_acc'] - subset['error_bar']
    subset['upper_bound'] = subset['test_acc'] + subset['error_bar']
    # plot
    sns.lineplot(x='n_train', y='test_acc', hue='method', data=subset, marker='o', markersize=8, linewidth=2, errorbar=None)
    plt.xlabel('Number of Training Samples (n_train)')
    plt.ylabel('Test Accuracy')
    plt.title(f'{hardness_var_name} {test_split.capitalize()} Test Acc')
    plt.legend(title='Method', loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.show()
    save_path = f'{save_dir}/{save_prefix}_test-{test_split}' + '.png'
    plt.savefig(save_path, format='png')
    plt.clf()
    # open image to return 
    img = PIL.Image.open(save_path)
    # now just show individual methods
    for probing_method in ['decoding', 'learned', 'finetuned']:
        max_n = 100 if probing_method == 'decoding' else 1000
        subset = df_results[df_results['test_on'] == test_split]
        subset = subset[subset['n_train'] <= max_n]
        probe_data = subset[subset['probing_method']==probing_method]
        sns.lineplot(x='n_train', y='test_acc', hue='method', data=probe_data, marker='o', markersize=8, linewidth=2, errorbar=None)
        plt.xlabel('Number of Training Samples (n_train)')
        plt.ylabel('Test Accuracy')
        plt.title(f'{hardness_var_name} {test_split.capitalize()} Test Acc')
        plt.legend(title='Method', loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout()
        plt.show()
        save_path = f'{save_dir}/{save_prefix}_test-{test_split}_{probing_method}' + '.png'
        plt.savefig(save_path, format='png')
        plt.clf()
    return img


def plot_acc_vs_hardness(test_stats_df, save_name, hardness_var_name):
    '''
    Plots accuracy against item level hardness for a given dataset
    args:
        test_stats_df: expected to have a hardness variable name and one or more columns with 'accuracy' in the name (possibly representing accuracies from different bootstraps or models)
            accuracy columns may be sparse
        hardness_var_name: name of hardness variable to use
    '''
    plot_df = test_stats_df.copy()
    n_hardness_levels = len(set(plot_df[hardness_var_name].values))
    acc_cols = filter(lambda x : 'acc' in x, plot_df.columns)
    do_bar_plot = n_hardness_levels < 10
    do_line_plot = not do_bar_plot
    # nanmean across the accuracy columns
    plot_df['mean_acc'] = np.nanmean(plot_df.loc[:, acc_cols].copy(), axis=1)
    plot_df = plot_df.loc[:, [hardness_var_name, 'mean_acc']].dropna()
    # make binned hardness variable
    if do_line_plot:
        n_bins = 5
        plot_df['hardness_binned'] = pd.cut(plot_df[hardness_var_name], bins=n_bins, labels=False)
    elif do_bar_plot:
        hardness_min = min(plot_df[hardness_var_name])
        hardness_max = max(plot_df[hardness_var_name])
        plot_df['hardness_binned'] = plot_df[hardness_var_name]
    # calculate group means and CIs
    grouped = plot_df.groupby('hardness_binned')['mean_acc'].agg(['mean', 'std', 'count']).reset_index()
    grouped['CI'] = 1.96 * grouped['std'] / np.sqrt(grouped['count'])
    # drop rows where less than n points
    at_least_n_points = 20
    grouped = grouped[grouped['count'] >= at_least_n_points]
    # plot
    color = '#4287f5'
    if do_bar_plot:
        # make x variable categorical in case there are missing categories after low n filtering
        grouped['hardness_binned'] = pd.Categorical(grouped['hardness_binned'], categories=list(range(hardness_min,hardness_max+1)))
        sns.barplot(x='hardness_binned', y='mean', data=grouped, color=color, errorbar=None)
        plt.errorbar(x=grouped.index, y=grouped['mean'], yerr=grouped['CI'], fmt='none', capsize=5, color='black')
    if do_line_plot:
        # Calculate bin centers
        bins = np.linspace(plot_df[hardness_var_name].min(), plot_df[hardness_var_name].max(), 6)
        bin_centers = bins[:-1] + np.diff(bins) / 2
        grouped['x_center'] = bin_centers[grouped.hardness_binned.values]
        # set x_axis ticks to the halfway points of the lower and upper hardness levels, assuming hardness var is 0-1
        grouped['x_axis'] = (grouped.hardness_binned + 1) / n_bins - (.5 / n_bins)
        sns.lineplot(x='x_center', y='mean', data=grouped, color=color, errorbar=None)
        plt.errorbar(x=grouped['x_center'], y=grouped['mean'], yerr=grouped['CI'], fmt='none', capsize=5, color=color)
    plt.xlabel('Hardness')
    plt.ylabel('Average Accuracy')
    plt.title(f"{hardness_var_name}")
    save_path = f'outputs/{save_name}' + '.png'
    plt.savefig(save_path, format='png')
    plt.clf()

def plot_sample_efficiency(df_results, save_prefix, outcome='eval_acc', x_var_list=None, no_prompt_avg_plot=False, no_multiprompt_plot=False):
    '''
    We plot eval_acc vs. n_train for a given dataset, to visualize hardness estimation results
    args:
    - x_var_list: can include 'n_train' and 'log_x'
    '''
    df_results = df_results.copy() # suppress slicing warnings
    df_results['log_x'] = df_results['n_train'].apply(lambda x: np.log10(x) if x != 0 else 0)
    if x_var_list is None:
        x_var_list = ['n_train', 'log_x']
    def log_axis(x_var_max):
        if x_var_max > 3:
            plt.xticks([1, 2, 3, 3.5], [10, 100, 1000, int(10**3.5)])
        elif x_var_max < 2.5:
            plt.xticks([0, 1, 2, 2.5], [1, 10, 100, int(10**2.5)])
        else:
            plt.xticks([1, 2, 3], [10, 100, 1000])
    # average results over boot idx and prompt idx
    if not no_prompt_avg_plot:
        per_n_train = df_results.groupby(['n_train', 'log_x'])[outcome].mean().reset_index()
        # binomial proportional confidence interval
        if 'error_bar' not in df_results.columns:
            n_dev = df_results['n_dev'][0]
            per_n_train['se'] = np.sqrt(per_n_train[outcome] * (1-per_n_train[outcome]) / n_dev)
            per_n_train['CI'] = 1.96 * per_n_train['se']
        # use pre-calculated error_bar
        else:
            per_n_train = pd.merge(per_n_train, df_results.loc[:,['n_train', 'error_bar']])
            per_n_train['CI'] = per_n_train['error_bar']
        for x_var in x_var_list:
            plt.errorbar(per_n_train[x_var], per_n_train[outcome], yerr=per_n_train['CI'], capsize=5, fmt='o-')
            if x_var == 'log_x':
                log_axis(df_results[x_var].max())
            plt.xlabel('n_train')
            plt.ylabel(outcome)
            plt.title(f"{outcome} vs n_train")
            save_name = f"{save_prefix}" + ('_log' if x_var == 'log_x' else '')
            filepath = f'outputs/{save_name}'
            plt.savefig(filepath + '.png', format='png')
            plt.clf()
    # plot multiple trajectories for different prompts
    if not no_multiprompt_plot:
        for x_var in x_var_list:
            sns.lineplot(x=x_var, y=outcome, data=df_results, hue='prompt_id', palette='flare', errorbar=None) #, errorbar=('ci', 95), err_style='bars')
            if x_var == 'log_x':
                log_axis(df_results[x_var].max())
            plt.xlabel('n_train')
            plt.ylabel(outcome)
            plt.title(f"{outcome} vs n_train\n{save_prefix}")
            plt.legend(title='Prompt ID')
            save_name = f"{save_prefix}_by_prompt" + ('_log' if x_var == 'log_x' else '')
            filepath = f'outputs/{save_name}'
            plt.savefig(filepath + '.png', format='png')
            plt.clf()

def plot_hardness_distribution(hardness_scores, name='hardness_distribution'):
    # plot a single provided vector of per-item hardness scores
    plt.hist(hardness_scores, bins=20, edgecolor='black')
    if 'NORMED' in name:
        plt.xlim(0,1)
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {name}")
    filepath = f'outputs/{name}'
    plt.savefig(filepath + '.png', format='png')
    plt.clf()

def plot_hardness_distributions_facet(hardness_scores, plot_name):
    # facet plot of multiple per-item hardness scores
    n_cols = 4
    n_rows = int(-(-len(hardness_scores.columns) // n_cols))  # Ceil division
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10))

    # make subplots
    for ax, name in zip(axes.flatten(), hardness_scores.columns):
        ax.hist(hardness_scores[name], bins=20, edgecolor='black')
        
        if 'NORMED' in name:
            ax.set_xlim(0, 1)
        
        ax.set_xlabel(name)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{name}")

    # remove empty subplots
    for remaining_ax in axes.flatten()[len(hardness_scores.columns):]:
        remaining_ax.axis('off')

    filepath = f'result_sheets/{plot_name}'
    plt.tight_layout()
    plt.subplots_adjust(hspace=1.2)
    plt.savefig(filepath + '.png', format='png')
    plt.clf()