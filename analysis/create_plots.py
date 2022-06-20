import itertools
import sys
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


def visualize_frequencies(et_frequencies, human_saliency, lm_frequencies, machine_saliency, outfile):
    human_data = pd.DataFrame({"Frequency": et_frequencies, "Importance": human_saliency})
    model_data = pd.DataFrame({"Frequency": lm_frequencies, "Importance": machine_saliency})

    all_data = pd.concat([human_data.assign(dataset='Model'), model_data.assign(dataset='Human')])
    print(all_data)
    #fig, ax = plt.subplots(figsize=(8, 4))
    mypalette = sns.diverging_palette(150, 275, s=80, l=55, n=2)
    sns.set(font_scale=1)
    sns.set_style("white")
    plot_ = sns.scatterplot(x='Frequency', y='Importance', data=all_data,
                  hue='dataset', palette=mypalette, alpha=0.75)

    print("Overall correlation: ")
    print("Human - Frequency")
    print(scipy.stats.spearmanr(et_frequencies, human_saliency)[0])
    print("Model - Frequency")
    print(scipy.stats.spearmanr(lm_frequencies, machine_saliency)[0])
    plt.legend(loc='upper right')

    plt.xlabel("Frequency")
    plt.ylabel("Relative Importance")
    plt.savefig(outfile,bbox_inches="tight" )
    plt.close()


def visualize_lengths(et_tokens, human_saliency, lm_tokens, machine_saliency, outfile):
    et_lengths = [len(token) for token in et_tokens]
    lm_lengths = [len(token) for token in lm_tokens]
    human_data = pd.DataFrame({"Length": et_lengths, "Importance": human_saliency})
    model_data = pd.DataFrame({"Length": lm_lengths, "Importance": machine_saliency})

    all_data = pd.concat([human_data.assign(dataset='Model'), model_data.assign(dataset='Human')])
    mypalette = sns.diverging_palette(150, 275, s=80, l=55, n=2)
    sns.set(font_scale=1)
    sns.set_style("white")
    sns.stripplot(x='Length', y='Importance', data=all_data,
                  hue='dataset', dodge=True, palette=mypalette)

    print("Overall correlation: ")
    print("Human - Length")
    print(scipy.stats.spearmanr(et_lengths, human_saliency)[0])
    print("Model - Length")
    print(scipy.stats.spearmanr(lm_lengths, machine_saliency)[0])
    plt.legend(loc='upper right')

    plt.xlabel("Length")
    plt.ylabel("Relative Importance")
    plt.savefig(outfile)
    plt.close()


def visualize_posdistribution(tag2importance, outfile):
    means = []
    stds = []
    num_instances = []

    function_word_tags = ["ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"]
    other_tags = ["PUNCT", "SYM", "X"]
    # Selection and order can be changed for the plot
    #labels = ["ADJ", "ADV", "NOUN", "VERB", "INTJ", "ADP", "AUX", "CONJ", "DET", "PART", "PRON", "NUM"]
    labels = ["ADJ", "ADV", "NOUN", "VERB", "ADP", "AUX", "CONJ", "DET", "PART", "PRON", "NUM"]
    for label in labels:
        if label not in other_tags:
            if label == "CONJ":
                values = tag2importance["CCONJ"] + tag2importance["SCONJ"]
            else:
                values = tag2importance[label]
            #print(label, len(values))
            mean = np.nanmean(values)
            std = np.nanstd(values)

            means.append(mean)
            stds.append(std)
            num_instances.append(len(values))
            #print(f"Mean: {mean:.4f}, {std:.4f}")

    data = pd.DataFrame({"PosTag": labels, "Mean": means, "Std": stds})

    sns.set(font_scale=2)
    fig = sns.catplot(x="PosTag", y="Mean", data=data, kind="bar", height=7, aspect=2)
    plt.ylim(0.0,0.055)
    plt.xlabel("")
    plt.ylabel("Relative Importance")

    # We added this type of plot (including number of instances per plot) to the appendix
    ax = fig.facet_axis(0, 0)
    for i,p in enumerate(ax.patches):
        ax.text(p.get_x() - 0.01, p.get_height() * 1.02, str(num_instances[i]), color='black',rotation='horizontal',fontsize=16)

    # Save the figure and show
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def calculate_saliency_by_wordclass(pos_tags, saliencies):

    tag2importance = defaultdict(list)

    for i, tags in enumerate(pos_tags):

        # if i % 500 == 0:
        #     print(i, len(pos_tags))
        try:
            if not len(tags) == len(saliencies[i]):
                pass

            else:
                salience = saliencies[i]

                for k, tag in enumerate(tags):
                    # We normalize importance by sentence length
                    # Otherwise tokens which occur more often in shorter sentences would receive much higher importance
                    # We found this to be reasonable but there might be a better way.
                    try:
                        tag2importance[tag].append(salience[k] / len(tags))
                    except:
                        print("Tokenisation ERROR!: ")
                        continue
        except TypeError:
            pass
    return tag2importance


def visualize_sentence(i, et_tokens, human_saliency, lm_saliency, outfile):
    print(et_tokens[i], human_saliency[i], lm_saliency[i])
    if i == 153:
        # hardcoded uppercasing for better looking plot
        tokens = ["Oh,", "Sherlock", "Holmes", "by", "all", "means."]
    else:
        if len(et_tokens[i]) == len(human_saliency[i]) == len(lm_saliency[i]):
            tokens = et_tokens[i]
        else:
            print("Mismatched tokenisation for sentence", i)
            return

    human_data = pd.DataFrame({"Tokens": tokens, "Importance": human_saliency[i]})
    model_data = pd.DataFrame({"Tokens": tokens, "Importance": lm_saliency[i]})

    all_data = pd.concat([human_data.assign(dataset='Human'), model_data.assign(dataset='Model')])
    fig, ax = plt.subplots(figsize=(8, 4))
    mypalette = sns.diverging_palette(150, 275, s=80, l=55, n=2)
    sns.set(font_scale=1)
    sns.set_style("white")
    sns.lineplot(x="Tokens", y="Importance", data=all_data, hue="dataset", style="dataset", markers=False,
                 dashes=[(3, 3), (3, 3)], palette=mypalette)
    sns.scatterplot(x="Tokens", y="Importance", data=all_data, hue="dataset", size="Importance", markers=["o", "o"],
                    legend=False, sizes=(50, 300), palette=mypalette)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)
    plt.savefig(outfile)
    plt.close()

def barplot_for_example(et_tokens, human_importance, outfile):
    i = 22

    human_data = pd.DataFrame({"Tokens": range(0, 9), "Importance": human_importance[i]})

    fig, ax = plt.subplots(figsize=(8, 2))
    sns.set_style("white")
    sns.set(font_scale=5)
    g = sns.barplot(x="Tokens", y="Importance", data=human_data, color="mediumaquamarine")

    # Some hard-coded finetuning for this particular example
    g.set_xticks(range(0, 9))
    g.set_xticklabels(["The"] + et_tokens[i][1:])
    g.set_yticks([0, 0.1, 0.2])
    g.set_yticklabels([0, 0.1, 0.2])
    g.tick_params(labelsize=10)
    g.set(xlabel=None)
    g.set_ylabel("Relative Importance", fontsize=10)

    plt.savefig(outfile)
    plt.close()

def flatten(mylist):
    return [item for sublist in mylist for item in sublist]


def flatten_saliency(mylist):
    flattened_salience = []
    for salience in mylist:
        normalized_salience = [s / len(salience) for s in salience]
        flattened_salience.extend(normalized_salience)
    return flattened_salience

def shorten_model_name(mp):
    if 'multilingual' in mp:
        short_name = 'mBert'
    elif mp.startswith('bert') or mp.startswith('rubert'):
        short_name = 'Bert'
    elif mp.startswith('albert'):
        short_name = 'Albert'
    elif mp.startswith('distilbert'):
        short_name = 'DistilBert'
    elif mp == 'human':
        short_name = 'Human'
    else:
        print(mp)
        raise ValueError
    return short_name

def shorten_importance_type(it):
    return {
        '-': '-',
        'attention': 'Attn (last)',
        'saliency': 'Saliency',
        'flow': 'Flow',
        'attention_1st_layer': 'Attn (1st)'}[it]

def format_corpus_name(c):
    return {
        'geco_nl': 'Dutch',
        'geco': 'English (Geco)',
        'potsdam': 'German',
        'russsent': 'Russian',
        'zuco':  'English (ZuCo)'
    }[c]

def format_df(df):
    df['model'] = df['model'].apply(shorten_model_name)
    df['importance_type'] = df['importance_type'].apply(shorten_importance_type)
    df['corpus'] = df['corpus'].apply(format_corpus_name)
    df = df.sort_values(['corpus', 'model', 'importance_type'])
    return df

def plot_correlation_heatmap(correlations, column_labels, row_labels, corpus, pvalues):

    pvalues = np.array(pvalues)
    title = corpus

    df = pd.DataFrame(correlations, columns = column_labels, index=row_labels)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # mark significant correlations
    labels = np.zeros_like(correlations, dtype=object)
    for ir,row in enumerate(correlations):
        for iv, value in enumerate(row):
            # significance threshold
            #import pdb
            #pdb.set_trace()
            if not value:
                labels[ir, iv] = "0.45"
            elif pvalues[ir, iv] < 0.05 and pvalues[ir, iv]!=1:
                labels[ir, iv] = "{:.2f}".format(value)+"***"
            elif pvalues[ir, iv] < 0.1 and pvalues[ir, iv]!=1:
                labels[ir, iv] = "{:.2f}".format(value)+"**"
            elif pvalues[ir, iv] < 0.2 and pvalues[ir, iv]!=1:
                labels[ir, iv] = "{:.2f}".format(value)+"*"
            else:
                labels[ir, iv] = "{:.2f}".format(value)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.set(font_scale = 1)
    ax = sns.heatmap(df, cmap=cmap, annot=labels, vmin=-1, vmax=1, fmt="", annot_kws={"fontsize":8})
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 12)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 12)
    plt.title(title, fontsize=24)
    plt.savefig("plots/heatmaps/"+corpus+"-correlations.png")

def plot_results_file(fn):
    df = pd.read_excel(io=fn, sheet_name='Model Importance', engine='openpyxl')
    df = format_df(df)
    for c in df['corpus'].unique():
        c_df = df[df['corpus'] == c]
        important_types = c_df['importance_type'].unique()
        models = c_df['model'].unique()
        correlations = []
        pvalues = []
        for m in models:
            m_df = c_df[c_df['model'] == m]
            correlation_row = []
            pvalue_row = []
            for t in important_types:
                t_df = m_df[m_df['importance_type'] == t]
                values = t_df['r_mean'].values
                value = values[0] if len(values) > 0 else None
                correlation_row.append(value)
                values_p = t_df['p_mean'].values
                value_p = values_p[0] if len(values_p) > 0 else 0
                pvalue_row.append(value_p)
            pvalues.append(pvalue_row)
            correlations.append(correlation_row)
        plot_correlation_heatmap(correlations, important_types, models, c, pvalues)

def plot_baselines(fn):
    df = pd.read_excel(io=fn, sheet_name='Corpus statistical baselines', engine='openpyxl')
    df = format_df(df)
    c_df = df[df['importance_type'] != '-']
    c_df = c_df.pivot_table(['length_r_mean', 'freq_r_mean'], ['corpus', 'model',], 'importance_type')
    human_df = df[df['importance_type'] == '-'].pivot_table(['length_r_mean', 'freq_r_mean'], ['corpus', 'model'], 'importance_type')

    c_df = c_df.round(decimals=2)
    human_df = human_df.round(decimals=2)

    with pd.ExcelWriter('baseline_results.xlsx') as writer:
        c_df.to_excel(writer, sheet_name='Model baselines')
        human_df.to_excel(writer, sheet_name='Human baselines')

def plot_linear_regression(X, y, y_pred):
    plt.scatter(X, y, color="black")
    plt.plot(X, y_pred, color="blue", linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.savefig("plots/linear_regression.png")

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
    model_type = 'monolingual' if model == 'Bert' else 'multilingual'
    c_df = df
    # Hack to deal with missing Flow value
    if model_type == 'multilingual':
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
    ax.set_ylabel('R')
    ax.set_title(title)
    ax.set_xticks(x, labels)

    return ax

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
    bl_df = pd.read_excel(io=fn, sheet_name='Corpus statistical baselines', engine='openpyxl')
    bl_df = bl_df[~bl_df['model'].str.contains('albert|distilbert')]
    fig, (ax1, ax2) = plt.subplots(2)
    formatted_bl_df = format_df(bl_df)
    plot_correlation(formatted_bl_df, ax1, model, 'length_r_mean', 'Length')
    plot_correlation(formatted_bl_df, ax2, model, 'freq_r_mean', 'Frequency')
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.set_size_inches(9.0, 7.5)
    fig.savefig('plots/correlations/baselines_barplot_{m}.jpg'.format(m=model))

if __name__ == '__main__':
    filename = sys.argv[1]

    # Human vs. model
    plot_human_vs_model(filename)

    # Baseline (mono- and multilingual models)
    plot_baselines_vs_importance(filename, 'Bert')
    plot_baselines_vs_importance(filename, 'mBert')
