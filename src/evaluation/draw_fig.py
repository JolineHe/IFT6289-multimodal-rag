#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Miao at 2025-04-15
'''
@File   : draw_fig.py
@Time   : 2025-04-15 10:06 p.m.
@Author : Miao
@Project: IFT6289-multimodal-rag 
@Desc   : Please enter here
'''

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def compare_rags_result_multimodal(df_all):
    # Melt the dataframe to reshape from wide to long format for scorers
    df = df_all[df_all['search_type']=='multimodal']
    df_melted = pd.melt(
        df,
        id_vars=["evaluation",  "alpha_text"],
        value_vars=["context_recall", "faithfulness", "factual_correctness(f1)"],
        var_name="scorer",
        value_name="score"
    )

    # Plot with seaborn
    g = sns.relplot(
        data=df_melted,
        kind="line",
        x="alpha_text",
        y="score",
        hue="evaluation",
        col="scorer",
        ci="sd",  # Show standard deviation
        facet_kws={'sharey': False, 'sharex': True},
        marker='o',
        height=3.8,  # Slightly smaller
        aspect=1.2
    )

    # Remove legend from subplots
    g._legend.remove()

    # Create shared legend at the top
    handles, labels = g.axes[0][0].get_legend_handles_labels()
    g.fig.legend(
        handles,
        labels,
        title="Evaluation",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),  # Adjust this if legend is too far up/down
        ncol=len(labels),
        frameon=False,
        fontsize=16,
        title_fontsize=16
    )

    # Customize tick and title font sizes
    for ax in g.axes.flatten():
        ax.tick_params(labelsize=16)
        ax.set_title(ax.get_title(), fontsize=16)
        ax.set_xlabel(ax.get_xlabel(), fontsize=16)
        ax.set_ylabel(ax.get_ylabel(), fontsize=16)

    # Adjust layout for top legend
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # top=0.93 reserves space for top legend
    plt.savefig("./multimodal_eval_plot.png", dpi=200, bbox_inches="tight")
    plt.show()

def compare_search_strategy(df_all):
    df = df_all[df_all['evaluation']=='text']
    df_melted = pd.melt(
        df,
        id_vars=["search_type", "alpha_text"],
        value_vars=["context_recall", "faithfulness", "factual_correctness(f1)"],
        var_name="scorer",
        value_name="score"
    )

    # Plot with seaborn
    g = sns.relplot(
        data=df_melted,
        kind="line",
        x="alpha_text",
        y="score",
        hue="search_type",
        col="scorer",
        ci="sd",  # Show standard deviation
        facet_kws={'sharey': False, 'sharex': True},
        marker='o',
        # height=3.8,  # Slightly smaller
        # aspect=1.2
    )

    # Remove legend from subplots
    g._legend.remove()

    # Create shared legend at the top
    handles, labels = g.axes[0][0].get_legend_handles_labels()
    g.fig.legend(
        handles,
        labels,
        title="Search Strategy",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),  # Adjust this if legend is too far up/down
        ncol=len(labels),
        frameon=False,
        fontsize=16,
        title_fontsize=16
    )

    # Customize tick and title font sizes
    for ax in g.axes.flatten():
        ax.tick_params(labelsize=16)
        ax.set_title(ax.get_title(), fontsize=16)
        ax.set_xlabel(ax.get_xlabel(), fontsize=16)
        ax.set_ylabel(ax.get_ylabel(), fontsize=16)

    # Adjust layout for top legend
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # top=0.93 reserves space for top legend
    plt.savefig("./searchtype_eval_plot.png", dpi=200, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    sns.set()
    result_path = './evl_results.csv'
    df_all = pd.read_csv(result_path)

    compare_search_strategy(df_all)

    compare_rags_result_multimodal(df_all)