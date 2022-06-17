import argparse
import os.path
import time
import scipy.stats
import sklearn.metrics


from ast import literal_eval
from sklearn.linear_model import LinearRegression

from analysis.create_plots import *
from analysis.calculate_baselines import calculate_freq_baseline, calculate_len_baseline, calculate_wordclass_baseline, calculate_permutation_baseline, calculate_linear_regression
from extract_model_importance.tokenization_util import merge_symbols, merge_albert_tokens, merge_hyphens

def extract_human_importance(dataset):
    with open("results/" + dataset + "_sentences.txt", "r") as f:
        sentences = f.read().splitlines()

    # split and lowercase
    tokens = [s.split(" ") for s in sentences]
    tokens = [[t.lower() for t in tokens] for tokens in tokens]

    human_importance = []
    with open("results/" + dataset + "_relfix_averages.txt", "r") as f:
        for line in f.read().splitlines():
            fixation_duration = np.fromstring(line, dtype=float, sep=',')
            human_importance.append(fixation_duration)

    return tokens, human_importance

# Importance type is either "saliency" or "attention"
def extract_model_importance(dataset, model, importance_type):
    lm_tokens = []
    lm_salience = []
    fname = "results/" + dataset + "_" + model + "_" + importance_type + ".txt"
    if not os.path.isfile(fname):
        error_message = fname + " does not exist. Have you run extract_all.py?"
        raise FileNotFoundError(error_message)
    with open(fname, "r") as f:
        for line in f.read().splitlines():
            tokens, heat = line.split("\t")
            try:
                tokens = list(literal_eval(tokens))
            except:
                lm_tokens.append([])
                lm_salience.append(np.array([]))
                continue
            salience = np.array(literal_eval(heat))

            # remove CLR and SEP tokens, this is an experimental choice
            lm_tokens.append(tokens[1:-1])
            salience = salience[1:-1]

            # Apply softmax over remaining tokens to get relative importance
            salience = scipy.special.softmax(salience)
            lm_salience.append(salience)
    return lm_tokens, lm_salience


def align_sentences(
    et_tokens, human_salience,
    lm_tokens, lm_salience,
    importance_type, corpus, modelname,
    log_tokenization_errors = False
):
    count_tok_errors = 0
    aligned_et_tokens, aligned_human_salience, aligned_lm_tokens, aligned_lm_salience = [],[],[],[]
    with open("results/correlations/" + corpus + "_" + modelname + "_" + importance_type + "_correlations.txt", "w") as outfile:
        outfile.write("Spearman\tKendall\tMutualInformation\n")
        for i, sentence in enumerate(et_tokens):
            if len(et_tokens[i]) < len(lm_tokens[i]):
                # TODO: some merge operations are already performed when extracting saliency. Would be better to have them all in one place.
                if modelname.startswith("albert"):
                    lm_tokens[i], lm_salience[i] = merge_albert_tokens(lm_tokens[i], lm_salience[i])
                    lm_tokens[i], lm_salience[i] = merge_hyphens(lm_tokens[i], lm_salience[i])

                lm_tokens[i], lm_salience[i] = merge_symbols(lm_tokens[i], lm_salience[i])
            if len(et_tokens[i]) == len(lm_tokens[i]) == len(human_salience[i]) == len(lm_salience[i]):
                aligned_et_tokens.append(et_tokens[i])
                aligned_human_salience.append(human_salience[i])
                aligned_lm_tokens.append(lm_tokens[i])
                aligned_lm_salience.append(lm_salience[i])
            else:
                # Uncomment if you want to know more a bout the tokenization alignment problems
                if log_tokenization_errors:
                    print("Tokenization Error:")
                    print(len(et_tokens[i]), len(lm_tokens[i]), len(human_salience[i]), len(lm_salience[i]))
                    print(et_tokens[i], lm_tokens[i])
                    print()
                count_tok_errors += 1
    return aligned_et_tokens, aligned_human_salience, aligned_lm_tokens, aligned_lm_salience



parser = argparse.ArgumentParser()
parser.add_argument('--skip-model-if-not-exist', action='store_true')


corpora_modelpaths = {
    # 'geco': [
    #     'albert-base-v2',
    #     'bert-base-uncased',
    #     'distilbert-base-uncased',
    #     'bert-base-multilingual-cased'
    # ],
    # 'geco_nl': [
    #     'GroNLP/bert-base-dutch-cased',
    #     'bert-base-multilingual-cased'
    #  ],
    # 'zuco': [
    #      'bert-base-uncased',
    #      'distilbert-base-uncased',
    #      'albert-base-v2',
    #      'bert-base-multilingual-cased'
    # ],
    'potsdam': [
        'dbmdz/bert-base-german-uncased',
        'distilbert-base-german-cased',
        'bert-base-multilingual-cased'
    ],
    # 'russsent': ['DeepPavlov/rubert-base-cased', 'bert-base-multilingual-cased']
}


def populate_dataframes():
    corpora_languages = {'geco': 'en',
                         'geco_nl': 'nl',
                         'zuco': 'en',
                         'potsdam': 'de',
                         'russsent': 'ru'}
    types = ["saliency", "attention", "flow"]
    human_words_columns = (
        'et_token', 'et_importance', 'frequency', 'length', 'sentence', 'corpus')
    aligned_words_columns = (
        'et_token', 'et_importance', 'lm_token', 'lm_importance', 'frequency',
        'length', 'sentence', 'corpus', 'importance_type', 'model')
    # Human only relative word importance
    human_words_df = pd.DataFrame(columns=human_words_columns)
    # Human and model aligned relative word importance
    aligned_words_df = pd.DataFrame(columns=aligned_words_columns)

    # Populate dataframes
    for corpus, modelpaths in corpora_modelpaths.items():
        print(corpus)
        et_tokens, human_importance = extract_human_importance(corpus)
        lang = corpora_languages[corpus]

        pos_tags, frequencies = process_tokens(et_tokens, lang)
        lengths = [[len(token) for token in sent] for sent in et_tokens]

        for idx, zipped in enumerate(zip(et_tokens, human_importance, frequencies, lengths)):
            data = {
                'et_token': zipped[0],
                'et_importance': zipped[1],
                'frequency': zipped[2],
                'length': zipped[3]
            }
            sent_df = pd.DataFrame(data, columns=human_words_df.columns)
            sent_df['sentence'], sent_df['corpus'] = idx, corpus
            human_words_df = human_words_df.append(sent_df)

        for importance_type in types:
            print(importance_type)
            for mp in modelpaths:
                modelname = mp.split("/")[-1]
                try:
                    lm_tokens, lm_importance = extract_model_importance(corpus, modelname, importance_type)
                except FileNotFoundError:
                    skip = parser.parse_args().skip_model_if_not_exist
                    if skip:
                        print("Skipping ", mp, " results file does not exist")
                        continue
                    else:
                        raise
                aligned = align_sentences(et_tokens, human_importance,
                                          lm_tokens, lm_importance,
                                          importance_type, corpus, modelname)
                aligned_et_tokens, aligned_human_salience, aligned_lm_tokens, aligned_lm_salience = aligned
                lm_lengths = [[len(token) for token in sent] for sent in aligned_lm_tokens]
                _, lm_frequencies = process_tokens(aligned_lm_tokens, lang)
                for idx, zipped in enumerate(zip(aligned[0], aligned[1], aligned[2], aligned[3], lm_frequencies, lm_lengths)):
                    data = {
                        'et_token': zipped[0],
                        'et_importance': zipped[1],
                        'lm_token': zipped[2],
                        'lm_importance': zipped[3],
                        'frequency': zipped[4],
                        'length': zipped[5]
                    }
                    sent_df = pd.DataFrame(data, columns=aligned_words_df.columns)
                    sent_df['sentence'], sent_df['corpus'], sent_df['importance_type'], sent_df['model'] = idx, corpus, importance_type, modelname
                    aligned_words_df = aligned_words_df.append(sent_df)
        # Force baseline values to be numeric
        # TODO: Move this to when creating dataframes.
        human_words_df['length'] = pd.to_numeric(human_words_df['length'])
        human_words_df['frequency'] = pd.to_numeric(human_words_df['frequency'] )
        aligned_words_df['length'] = pd.to_numeric(aligned_words_df['length'])
        aligned_words_df['frequency'] = pd.to_numeric(aligned_words_df['frequency'])

        return human_words_df, aligned_words_df

def calculate_correlation(
    words_df: pd.DataFrame, X_column: str, y_column: str, groupby: list,
    by_sentence: bool = True
    ):

    def spearmanr_pval(x,y):
        return scipy.stats.spearmanr(x,y)[1]

    extra_sentence_column = ['sentence'] if by_sentence else []
    grouped = words_df.groupby(groupby + extra_sentence_column)[[X_column, y_column]]
    spearman = grouped.corr('spearman').iloc[0::2, -1]
    pvalues = grouped.corr(method=spearmanr_pval).iloc[0::2, -1]

    spearman = spearman.groupby(groupby).agg(['mean', np.nanstd]).add_prefix('spearman_')
    pvalues = pvalues.groupby(groupby).agg(['mean', np.nanstd]).add_prefix('pvalues_')
    return spearman.join(pvalues)


def calculate_regression(
    words_df: pd.DataFrame,
    observations: list, outcomes: list,
    groupby: list,
    apply_log : bool = False,
    ):

    def apply_linear_regression(df):
        r_squares = {}
        for obs in observations:
            for out in outcomes:
                # TODO: fix apply_log -> causes errors
                if apply_log:
                    df[obs + [out] ] = df[obs + [out] ].apply(np.log)
                    df[obs + [out] ] = df[obs + [out] ].replace([np.inf, -np.inf], np.nan).dropna()
                X, y = df[obs], df[out]
                reg = LinearRegression().fit(X, y)
                r_sq = reg.score(X, y)
                column = "+".join(obs) + "->" + out
                r_squares[column] = r_sq
        return pd.Series(r_squares)

    return words_df.groupby(groupby).apply(apply_linear_regression)

""" Calculate correlations (Spearman)"""

# Human vs. model
human_words_df, aligned_words_df = populate_dataframes()
human_vs_model = calculate_correlation(aligned_words_df, 'et_importance', 'lm_importance', groupby = ['importance_type', 'corpus', 'model'])

# Human baselines
human_vs_len = calculate_correlation(human_words_df, 'et_importance', 'length', groupby = ['corpus'])
human_vs_freq = calculate_correlation(human_words_df, 'et_importance', 'frequency', groupby = ['corpus'])
#human_vs_baselines = human_vs_len.join(human_vs_freq)

# Model baselines
model_vs_len = calculate_correlation(aligned_words_df, 'lm_importance', 'length', groupby = ['importance_type', 'corpus', 'model'])
model_vs_freq = calculate_correlation(aligned_words_df, 'et_importance', 'frequency', groupby = ['importance_type', 'corpus', 'model'])
#model_vs_baselines = model_vs_len.join(model_vs_freq)

""" Calculate linear regression models"""

# Regression
observations = [
    ['length'],
    ['frequency'],
    ['length', 'frequency'],
]
outcomes = ['et_importance']
human_vs_regression = calculate_regression(human_words_df, observations, outcomes, ['corpus'])

observations = [
    ['length'],
    ['frequency'],
    ['lm_importance'],
    ['length', 'frequency'],
    ['length', 'lm_importance'],
    ['frequency', 'lm_importance'],
    ['length', 'frequency', 'lm_importance'],
]
outcomes = ['et_importance']
model_vs_regression = calculate_regression(
    aligned_words_df, observations, outcomes,
    groupby = ['importance_type', 'corpus', 'model']
)
