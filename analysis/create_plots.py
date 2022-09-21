import itertools
import sys
import matplotlib
import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import seaborn as sns
import spacy
from wordfreq import word_frequency
from spacy.tokens import Doc
from spacy.pipeline import Tagger

from .format import shorten_model_name, shorten_importance_type, format_corpus_name, format_df

def process_tokens(all_tokens, lang):
    if lang == 'en':
            nlp = spacy.load('en_core_web_md')
    elif lang == 'de':
        nlp = spacy.load('de_core_news_sm')
    elif lang == 'nl':
        nlp = spacy.load('nl_core_news_sm')
    elif lang == 'ru':
        nlp = spacy.load('ru_core_news_sm')
    else:
        raise ValueError("Language '" + lang  +"' is not available.")

    pos_tags = []
    frequencies = []

    for tokens in all_tokens:
        doc = Doc(nlp.vocab, words=tokens)
        processed = nlp(doc)
        sentence_tags = [token.pos_ for token in processed]
        sentence_frequencies = [word_frequency(token.lemma_, lang, wordlist='best', minimum=0.0) for token in processed]
        pos_tags.append(sentence_tags)
        frequencies.append(sentence_frequencies)

    return pos_tags, frequencies

def make_regression_table(fn):
    df = pd.read_excel(io=fn, sheet_name='Regression analysis', engine='openpyxl')
    df['model'] = df['model'].apply(shorten_model_name)
    df['importance_type'] = df['importance_type'].apply(shorten_importance_type)
    df['corpus'] = df['corpus'].apply(format_corpus_name)
    df = df.sort_values(['corpus', 'model', 'importance_type'])
    shared_columns = ['corpus']
    human_columns = ['human~freq', 'human~length', 'human~freq+length']
    neural_columns = [
        'model',
        'importance_type',
        'human~model',
        'human~model+freq',
        'human~model+length',
        'human~model+freq+length'
    ]
    h_df = df[shared_columns + human_columns]
    h_df = h_df.pivot_table(index=['corpus'])
    m_df = df[shared_columns + neural_columns]
    m_df = m_df[~m_df['model'].str.contains('Albert')]
    m_df = m_df[~m_df['model'].str.contains('DistilBert')]

    h_df = h_df.round(decimals=2)
    m_df = m_df.round(decimals=2)

    print(h_df.to_latex(index=False))
    print(m_df.to_latex(index=False))

    with pd.ExcelWriter('linear_regression_results.xlsx') as writer:
        h_df.to_excel(writer, sheet_name='Human linear regression')
        m_df.to_excel(writer, sheet_name='Model linear regression')

    return h_df, m_df

def plot_correlation(df, ax, model : str, column: str, title: str):
    c_df = df
    # Hack to deal with missing Flow value
    if model == 'mBert':
        row = {'corpus': 'English (Geco)', 'model': 'mBert', 'importance_type': 'Flow', column: 0}
        c_df = c_df.append(row, ignore_index=True)

    c_df = c_df.sort_values(by='corpus')
    labels = list(c_df.corpus.unique())


    flow = c_df[(c_df['model'] == model) & (c_df['importance_type'] == 'Flow')][column]
    attn_last = c_df[(c_df['model'] == model) & (c_df['importance_type'] == 'Attn (last)')][column]
    attn_1st= c_df[(c_df['model'] == model) & (c_df['importance_type'] == 'Attn (1st)')][column]
    saliency = c_df[(c_df['model'] == model) & (c_df['importance_type'] == 'Saliency')][column]
    x = np.arange(len(labels))  # the label locations

    width = 0.15  # the width of the bars
    if 'length' in column.lower() or 'freq' in column.lower():
        human = c_df[(c_df['model'] == 'Human') & (c_df['importance_type'] == '-')][column]
        rects1 = ax.bar(x - width * 2.5, flow, width, label='Flow')
        rects2 = ax.bar(x - width * 1.5, attn_1st, width, label='Attn (1st)')
        rects3 = ax.bar(x - width * 0.5, attn_last, width, label='Attn (last)')
        rects4 = ax.bar(x + width * 0.5, saliency, width, label='Saliency')
        rects5 = ax.bar(x + width * 1.5, human, width, label='Human')
    else:
        rects1 = ax.bar(x - width * 2, flow, width, label='Flow')
        rects2 = ax.bar(x - width, attn_1st, width, label='Attn (1st)')
        rects3 = ax.bar(x, attn_last, width, label='Attn (last)')
        rects4 = ax.bar(x + width, saliency, width, label='Saliency')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r'$\rho$')
    ax.set_title(title)
    ax.set_xticks(x, labels)

    return ax

def make_regression_table(fn):
    df = pd.read_excel(io=fn, sheet_name='Regression analysis', engine='openpyxl')
    df['model'] = df['model'].apply(shorten_model_name)
    df['importance_type'] = df['importance_type'].apply(shorten_importance_type)
    df['corpus'] = df['corpus'].apply(format_corpus_name)
    df = df.sort_values(['corpus', 'model', 'importance_type'])
    shared_columns = ['corpus']
    human_columns = ['human~freq', 'human~length', 'human~freq+length']
    neural_columns = [
        'model',
        'importance_type',
        'human~model',
        'human~model+freq',
        'human~model+length',
        'human~model+freq+length'
    ]
    h_df = df[shared_columns + human_columns]
    h_df = h_df.pivot_table(index=['corpus'])
    m_df = df[shared_columns + neural_columns]
    m_df = m_df[~m_df['model'].str.contains('Albert')]
    m_df = m_df[~m_df['model'].str.contains('DistilBert')]

    h_df = h_df.round(decimals=2)
    m_df = m_df.round(decimals=2)

    print(h_df.to_latex(index=False))
    print(m_df.to_latex(index=False))

    with pd.ExcelWriter('linear_regression_results.xlsx') as writer:
        h_df.to_excel(writer, sheet_name='Human linear regression')
        m_df.to_excel(writer, sheet_name='Model linear regression')

    return h_df, m_df


def regression_bar_plot(fn, model : str):
    model_type = 'monolingual' if model == 'Bert' else 'multilingual'
    df = pd.read_excel(io=fn, sheet_name='Regression analysis', engine='openpyxl')
    df['model'] = df['model'].apply(shorten_model_name)
    df['importance_type'] = df['importance_type'].apply(shorten_importance_type)
    df['corpus'] = df['corpus'].apply(format_corpus_name)
    if model_type == 'multilingual':
        row = {'corpus': 'English (Geco)', 'model': 'mBert', 'importance_type': 'Flow'}
        df = df.append(row, ignore_index=True)
    df = df.sort_values(['corpus', 'model', 'importance_type'])
    shared_columns = ['corpus']
    human_columns = ['human~freq', 'human~length', 'human~freq+length']
    neural_columns = [
        'model',
        'importance_type',
        'human~model',
        'human~model+freq',
        'human~model+length',
        'human~model+freq+length'
    ]

    h_df = df[shared_columns + human_columns]
    h_df = h_df.pivot_table(index=['corpus'])
    m_df = df[shared_columns + neural_columns]
    m_df = m_df[~m_df['model'].str.contains('ALBERT')]
    m_df = m_df[~m_df['model'].str.contains('DistilBERT')]

    h_df = h_df.round(decimals=2)
    m_df = m_df.round(decimals=2)

    fig, ax = plt.subplots()
    ax.yaxis.set_tick_params(labelbottom=True, bottom=False)
    plt.rcParams["font.family"] = "Times New Roman"
    importance_types = m_df['importance_type'].unique()
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']
    cmap = matplotlib.cm.get_cmap('tab20')

    variables = {
        'human~model': 'white',
        'human~model+freq': '#a4bfef',
        'human~model+length': '#6a93cb',
        'human~model+freq+length': '#071330'
    }
    it_keys = {
        'Attn (last)': r'$A_l$',
        'Attn (1st)': r'$A_1$',
        'Flow': r'$A_f$',
        'Saliency': r'$S$'
    }
    corpora = m_df['corpus'].unique()
    hatches = ['/', "|", 'x', '.']
    xticks = []
    xlabels = []
    for idx, c in enumerate(corpora):
        c_df = m_df[(m_df['corpus'] == c) & (m_df['model'] == model)]
        importance_types = list([it_keys[it] for it in c_df['importance_type'].unique()])
        idx_xticks = np.array([idx + (idx * 0.25)] * len(importance_types))
        idx_xticks += [n * 0.25 for n in range(len(idx_xticks))]
        xticks += list(idx_xticks)
        if len(importance_types) == 3:
            importance_types[1] += "\n" + c
        elif len(importance_types) == 4:
            importance_types[1] += "\n   " + c
        else:
            raise ValueError("There should be at least three importance types: Attn (last), Attn (1st) and Saliency")
        xlabels += importance_types
        for idx0, v in enumerate(reversed(variables)):
            values = list(c_df[v])
            ind = np.array([idx + (idx * 0.25)] * len(values), dtype=float)
            spacing = [n * 0.25 for n in range(len(values))]
            ind += spacing
            bars = ax.bar(ind, values, color=variables[v], width = 0.25, edgecolor='k')

    ax.set_ylabel(r'$R^2$')
    ax.set_ylim([0, 1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.legend(labels=list(reversed([v[6:] for v in variables.keys()])))
    fig.set_size_inches(9.0, 7.5)

    plt.savefig('plots/regression/regression_barplot_{m}.pdf'.format(m=model_type))
    plt.savefig('plots/regression/regression_barplot_{m}.jpg'.format(m=model_type))

def plot_human_vs_model(fn):
    i_df = pd.read_excel(io=fn, sheet_name='Model Importance', engine='openpyxl')
    i_df = i_df[~i_df['model'].str.contains('albert|distilbert')]
    plt.rcParams["font.family"] = "Times New Roman"
    formatted_i_df = format_df(i_df)
    fig, (ax1, ax2) = plt.subplots(2)
    plot_correlation(formatted_i_df, ax1, 'Bert', 'r_mean', 'Monolingual')
    plot_correlation(formatted_i_df, ax2, 'mBert', 'r_mean', 'Multilingual')
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.set_size_inches(9.0, 7.5)
    fig.savefig('plots/correlations/importance_barplot.jpg')

def plot_baselines_vs_importance(fn, model : str):
    model_type = 'monolingual' if model == 'Bert' else 'multilingual'
    bl_df = pd.read_excel(io=fn, sheet_name='Corpus statistical baselines', engine='openpyxl')
    bl_df = bl_df[~bl_df['model'].str.contains('albert|distilbert')]
    fig, (ax1, ax2) = plt.subplots(2)
    formatted_bl_df = format_df(bl_df)
    plot_correlation(formatted_bl_df, ax1, model, 'length_r_mean', 'Word length')
    plot_correlation(formatted_bl_df, ax2, model, 'freq_r_mean', 'Word frequency')
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.set_size_inches(9.0, 7.5)
    fig.savefig('plots/correlations/baselines_barplot_{m}.jpg'.format(m=model_type))

if __name__ == '__main__':
    filename = sys.argv[1]

    # Human vs. model
    plot_human_vs_model(filename)

    # Baseline (mono- and multilingual models)
    plot_baselines_vs_importance(filename, 'Bert')
    plot_baselines_vs_importance(filename, 'mBert')

    # Regression analysis bar plots
    regression_bar_plot(filename, 'Bert')
    regression_bar_plot(filename, 'mBert')

    # Appendix full results table
    make_regression_table(filename)
