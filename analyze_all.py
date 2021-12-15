import os.path
import time
import scipy.stats
import sklearn.metrics
from ast import literal_eval
from analysis.create_plots import *
from analysis.calculate_baselines import calculate_freq_baseline, calculate_len_baseline, calculate_permutation_baseline
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

            tokens = list(literal_eval(tokens))
            salience = np.array(literal_eval(heat))

            # remove CLR and SEP tokens, this is an experimental choice
            lm_tokens.append(tokens[1:-1])
            salience = salience[1:-1]

            # Apply softmax over remaining tokens to get relative importance
            salience = scipy.special.softmax(salience)
            lm_salience.append(salience)

    return lm_tokens, lm_salience


def compare_importance(et_tokens, human_salience, lm_tokens, lm_salience, importance_type):
    count_tok_errors = 0

    spearman_correlations = []
    kendall_correlations = []
    mutual_information = []
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
                # Calculate the correlation
                spearman = scipy.stats.spearmanr(lm_salience[i], human_salience[i])[0]
                spearman_correlations.append(spearman)
                kendall = scipy.stats.kendalltau(lm_salience[i], human_salience[i])[0]
                kendall_correlations.append(kendall)
                mi_score = sklearn.metrics.mutual_info_score(lm_salience[i], human_salience[i])
                mutual_information.append(mi_score)
                outfile.write("{:.2f}\t{:.2f}\t{:.2f}\n".format(spearman, kendall, mi_score))

            else:
                # Uncomment if you want to know more about the tokenization alignment problems
                print("Tokenization Error:")
                print(len(et_tokens[i]), len(lm_tokens[i]), len(human_salience[i]), len(lm_salience[i]))
                print(et_tokens[i], lm_tokens[i])
                print()
                count_tok_errors += 1


    print(corpus, modelname)
    print("Token alignment errors: ", count_tok_errors)
    print("Spearman Correlation Model: Mean, Stdev")
    mean_spearman = np.nanmean(np.asarray(spearman_correlations))
    std_spearman = np.nanstd(np.asarray(spearman_correlations))
    print(mean_spearman, std_spearman)

    print("\n\n\n")

    return mean_spearman, std_spearman


corpora_modelpaths = {
    'geco': [
        'bert-base-uncased',
        'distilbert-base-uncased',
        'albert-base-v2',
        'bert-base-multilingual-cased'
    ],
    'geco_nl': [
        'GroNLP/bert-base-dutch-cased',
        'bert-base-multilingual-cased'
    ],
    'zuco':
        ['bert-base-uncased',
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

corpora_languages = {'geco': 'en',
                     'geco_nl': 'nl',
                     'zuco': 'en',
                     'potsdam': 'de',
                     'russsent': 'ru'}

types = ["saliency", "attention"]


baseline_columns = ('corpus', 'model', 'length_mean_correlation',
                    'length_std_correlation', 'freq_mean_correlation')
results_columns = ('importance_type', 'corpus', 'model', 'mean_correlation',
                   'std_correlation')
baseline_results = pd.DataFrame(columns=baseline_columns)
results = pd.DataFrame(columns=results_columns)
permutation_results = pd.DataFrame(
    columns=('importance_type', 'corpus', 'model', 'mean_correlation', 'std_correlation'))

for corpus, modelpaths in corpora_modelpaths.items():
    print(corpus)
    et_tokens, human_importance = extract_human_importance(corpus)
    lang = corpora_languages[corpus]

    for importance_type in types:
        print(importance_type)

        # Human baselines
        et_tokens, human_importance = extract_human_importance(corpus)
        pos_tags, frequencies = process_tokens(et_tokens, lang)
        len_mean, len_std = calculate_len_baseline(et_tokens, human_importance)
        freq_mean, freq_std = calculate_freq_baseline(frequencies, human_importance)
        row = {
            'corpus': corpus,
            'modelname': 'human',
            'length_mean_correlation': len_mean,
            'length_std_correlation': len_std,
            'freq_mean_correlation': freq_mean,
            'freq_std_correlation': freq_std
        }
        baseline_results = baseline_results.append(row, ignore_index=True)

        for mp in modelpaths:
            modelname = mp.split("/")[-1]
            lm_tokens, lm_importance = extract_model_importance(corpus, modelname, importance_type)

            # Model Correlation
            spearman_mean, spearman_std = compare_importance(et_tokens, human_importance, lm_tokens, lm_importance, importance_type)
            results = results.append( {'importance_type': importance_type, 'corpus': corpus, 'model': modelname, 'mean_correlation': spearman_mean, 'std_correlation': spearman_std}, ignore_index=True)

            #Permutation Baseline
            spearman_mean, spearman_std = calculate_permutation_baseline(human_importance, lm_importance)
            permutation_results = permutation_results.append(
                {'importance_type': importance_type, 'corpus': corpus, 'model': mp, 'mean_correlation': spearman_mean, 'std_correlation': spearman_std},
                ignore_index=True)

            # Plots
            lm_tokens, lm_importance = extract_model_importance(corpus, modelname, importance_type)

            # Plot length vs saliency
            flat_et_tokens = flatten(et_tokens)
            flat_lm_tokens = flatten(lm_tokens)
            flat_human_importance = flatten_saliency(human_importance)
            flat_lm_importance = flatten_saliency(lm_importance)
            # visualize_lengths(flat_et_tokens, flat_human_importance, flat_lm_tokens, flat_lm_importance, "plots/" + corpus + "_" + model + "_length.png")

            # Plot an example sentence
            i = 10
            visualize_sentence(i, et_tokens, human_importance, lm_importance, "plots/" + modelname + "_" + str(i) + ".png")

            # Linguistic pre-processing (POS-tagging, word frequency extraction)
            #lm_tokens and et_tokens differ slightly because there are some cases which cannot be perfectly aligned.
            lm_pos_tags, lm_frequencies = process_tokens(lm_tokens, lang)

            # Plot POS distribution with respect to saliency
            tag2machineimportance = calculate_saliency_by_wordclass(lm_pos_tags, lm_importance)
            visualize_posdistribution(tag2machineimportance, "plots/" + corpus + "_" + modelname + "_wordclasses.png")

            # Plot POS distribution with respect to human importance
            tag2humanimportance = calculate_saliency_by_wordclass(pos_tags, human_importance)
            visualize_posdistribution(tag2humanimportance, "plots/" + corpus + "_human_wordclasses.png")

            # Plot frequency vs saliency
            flat_frequencies= flatten(frequencies)
            flat_lm_frequencies = flatten(lm_frequencies)
            visualize_frequencies(flat_frequencies, flat_human_importance, flat_lm_frequencies,
                                      flat_lm_importance, "plots/" + corpus + "_" + modelname + "_frequency.png")

            # LM baselines
            len_mean, len_std = calculate_len_baseline(et_tokens, lm_importance)
            freq_mean, freq_std = calculate_freq_baseline(frequencies, lm_importance)
            row = {
                'corpus': corpus,
                'modelname': modelname,
                'length_mean_correlation': len_mean,
                'length_std_correlation': len_std,
                'freq_mean_correlation': freq_mean,
                'freq_std_correlation': freq_std
            }
            baseline_results = baseline_results.append(row, ignore_index=True)


    # Store results
    timestr = time.strftime("%Y-%m-%d-%H:%M:%S")
    with open("results/all_results-" + timestr + ".txt", "w") as outfile:
        outfile.write("Model Importance: \n")
        outfile.write(results.to_latex())

        outfile.write("\n\nPermutation Baselines: \n")
        outfile.write(permutation_results.to_latex())

        outfile.write("\n\nLen-Freq Baselines: \n")
        outfile.write(baseline_results.to_latex())

        print(results)
        print()
        print(permutation_results)
        print()
        print(baseline_results)
        print()
