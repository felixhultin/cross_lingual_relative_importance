import argparse
import os.path
import time
import scipy.stats
import sklearn.metrics


from ast import literal_eval
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

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
        if dataset == 'geco' and importance_type == 'flow' and model != 'bert-base-uncased':
            print("Skipping ", fname)
        else:
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
        for i, sentence in tqdm(enumerate(et_tokens)):
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

def populate_dataframes(corpora_modelpaths, types):
    human_words_fn = 'results/words/' + 'human_words.csv'
    aligned_words_fn = 'results/words/' + 'aligned_words.csv'
    if os.path.isfile(human_words_fn) and os.path.isfile(aligned_words_fn):
        print("Word-level files found.")
        human_words_df = pd.read_csv(human_words_fn)
        aligned_words_df = pd.read_csv(aligned_words_fn)
        return human_words_df, aligned_words_df
    print("Creating word-level files...")
    corpora_languages = {'geco': 'en',
                         'geco_nl': 'nl',
                         'zuco': 'en',
                         'potsdam': 'de',
                         'russsent': 'ru'}
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
                # unique_path = ['results', corpus, importance_type, modelname + '_words.csv']
                # aligned_words_fn = "/".join(unique_path)
                # if os.path.isfile(aligned_words_fn):
                #     pass
                try:
                    lm_tokens, lm_importance = extract_model_importance(corpus, modelname, importance_type)
                except FileNotFoundError:
                    skip = parser.parse_args().skip_model_if_not_exist
                    if skip or corpus == 'geco' and importance_type == 'flow' and modelname != 'bert-base-uncased':
                        print("Skipping ", modelname, " results file does not exist")
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

    human_words_df.to_csv(human_words_fn)
    aligned_words_df.to_csv(aligned_words_fn)

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

    spearman = spearman.groupby(groupby).agg(['mean', np.nanstd]).add_prefix(y_column + '_spearman_')
    pvalues = pvalues.groupby(groupby).agg(['mean', np.nanstd]).add_prefix(y_column + '_pvalues_')
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
                filtered_df = df[ obs + [out] ]
                column = "+".join(obs) + "~" + out
                if apply_log:
                    nof_entries = len(filtered_df)
                    filtered_df = filtered_df\
                        .apply(np.log)\
                        .replace([np.inf, -np.inf], np.nan)\
                        .dropna()
                    nof_entries_dropped = nof_entries - len(filtered_df)
                    if nof_entries_dropped:
                        print(column, ":", nof_entries_dropped, "entries dropped from", nof_entries, "to", len(filtered_df))
                X, y = filtered_df[obs], filtered_df[out]
                reg = LinearRegression().fit(X, y)
                r_sq = reg.score(X, y)
                r_squares[column] = r_sq
        return pd.Series(r_squares)

    return words_df.groupby(groupby).apply(apply_linear_regression)

def calculate_results(
    human_words_df : pd.DataFrame,
    aligned_words_df: pd.DataFrame,
    by_sentence : bool = True,
    apply_log: bool = False
):
    # Human vs. model (importance)
    human_vs_model = calculate_correlation(
        aligned_words_df,
        'et_importance', 'lm_importance',
        groupby = ['importance_type', 'corpus', 'model'],
        by_sentence = by_sentence
    )

    # Human vs. baselines
    human_vs_len = calculate_correlation(
        human_words_df,
        'et_importance', 'length',
        groupby = ['corpus'],
        by_sentence = by_sentence
    )
    human_vs_freq = calculate_correlation(
        human_words_df,
        'et_importance', 'frequency',
        groupby = ['corpus'],
        by_sentence = by_sentence
    )
    human_vs_baselines = human_vs_len.join(human_vs_freq)

    # Model vs. baselines
    model_vs_len = calculate_correlation(
        aligned_words_df,
        'lm_importance', 'length',
        groupby = ['importance_type', 'corpus', 'model'],
        by_sentence = by_sentence
    )
    model_vs_freq = calculate_correlation(
        aligned_words_df,
        'et_importance', 'frequency',
        groupby = ['importance_type', 'corpus', 'model'],
        by_sentence = by_sentence
    )
    model_vs_baselines = model_vs_len.join(model_vs_freq)

    # Regression
    observations = [
        ['length'],
        ['frequency'],
        ['length', 'frequency'],
    ]
    outcomes = ['et_importance']
    human_vs_regression = calculate_regression(
        human_words_df,
        observations, outcomes,
        ['corpus'],
        apply_log = apply_log
    )

    observations = [
        ['lm_importance'],
        ['lm_importance', 'length'],
        ['lm_importance', 'frequency'],
        ['lm_importance', 'length', 'frequency'],
    ]
    outcomes = ['et_importance']
    model_vs_regression = calculate_regression(
        aligned_words_df,
        observations, outcomes,
        groupby = ['importance_type', 'corpus', 'model'],
        apply_log = apply_log
    )
    results = {
        'human_vs_model': human_vs_model,
        'human_vs_baselines': human_vs_baselines,
        'model_vs_baselines': model_vs_baselines,
        'human_vs_regression': human_vs_regression,
        'model_vs_regression': model_vs_regression,
    }

    return results

def write_results_to_excel(results):
    """ Store results to excel file with timestamp. """
    timestr = time.strftime("%Y-%m-%d-%H:%M:%S")
    fname = "results/analysis/all_results-" + timestr + ".xlsx"
    fname = "results/analysis/test.xlsx"
    with pd.ExcelWriter(fname) as writer:
        # Write human_vs_model to excel
        results['human_vs_model']\
            .reset_index()\
            .rename(columns = lambda c: c.replace('lm_importance_', ''))\
            .to_excel(writer, sheet_name='Model Importance', index=False)
        # Write baselines to excel
        human_vs_baselines = results['human_vs_baselines'].reset_index()
        human_vs_baselines['importance_type'] = 'it'
        human_vs_baselines['model'] = 'model'
        model_vs_baselines = results['model_vs_baselines'].reset_index()
        model_vs_baselines\
            .append(human_vs_baselines)\
            .rename(columns =\
                lambda c: c.replace('frequency_', '').replace('length_', '')
            )\
            .to_excel(writer, sheet_name='Corpus statistical baselines', index=False)
        # Write regression to excel
        human_vs_regression = results['human_vs_regression'].reset_index()
        model_vs_regression = results['model_vs_regression'].reset_index()
        model_vs_regression\
            .merge(human_vs_regression, on = ['corpus'])\
            .rename(columns = lambda c: c\
                .replace('frequency', 'freq')\
                .replace('length', 'len')\
                .replace('lm_importance', 'model')\
                .replace('et_importance', 'human')
            )\
            .to_excel(writer, sheet_name='Regression analysis', index=False)

def check_result_files_exist(corpora_modelpaths, types):
    for corpus, model_paths in corpora_modelpaths.items():
        for mp in model_paths:
            modelname = mp.split("/")[-1]
            for type in types:
                fname = "results/" + corpus + "_" + modelname + "_" + type + ".txt"
                if not os.path.isfile(fname):
                    # Skip 'flow' files which have not been computed
                    if corpus == 'geco' and type == 'flow' and mp != 'bert-base-uncased':
                        print("Skipping ", fname)
                        continue
                    error_message = fname + " does not exist. Have you run extract_all.py?"
                    raise FileNotFoundError(error_message)



if __name__ == '__main__':

    corpora_modelpaths = {
        'geco': [
            'albert-base-v2',
            'bert-base-uncased',
            'distilbert-base-uncased',
            'bert-base-multilingual-cased'
        ],
        'geco_nl': [
            'GroNLP/bert-base-dutch-cased',
            'bert-base-multilingual-cased'
         ],
        'zuco': [
             'bert-base-uncased',
             'distilbert-base-uncased',
             'albert-base-v2',
             'bert-base-multilingual-cased'
        ],
        'potsdam': [
            'dbmdz/bert-base-german-uncased',
            'distilbert-base-german-cased',
            'bert-base-multilingual-cased'
        ],
        'russsent': ['DeepPavlov/rubert-base-cased', 'bert-base-multilingual-cased']
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-model-if-not-exist', action='store_true')
    parser.add_argument('--by-tokens', action='store_true', default=False)
    parser.add_argument('--apply-log-to-regression', action='store_true', default=False)
    apply_log = parser.parse_args().apply_log_to_regression
    by_sentence = False if parser.parse_args().by_tokens else True
    types = ["saliency", "attention", "attention_1st_layer", "flow"]
    check_result_files_exist(corpora_modelpaths, types)
    human_words_df, aligned_words_df = populate_dataframes(corpora_modelpaths, types)
    results = calculate_results(human_words_df, aligned_words_df, by_sentence=by_sentence, apply_log=apply_log)
    write_results_to_excel(results)
