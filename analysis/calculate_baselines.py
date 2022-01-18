import sklearn
import numpy as np
import scipy.stats
import random

from sklearn.linear_model import LinearRegression

# Here we calculate length, frequency, and permutation baselines on the sentence level
# Note that the correlation functions yield a warning if one of the list is constant (stdev = 0)
# For example, the phrase "you did not" would yield the length vector [3,3,3]
# and then correlation cannot be calculated
def calculate_len_baseline(tokens, importance):
    spearman = []
    kendall = []
    mi_scores = []

    for i, sent in enumerate(tokens):
        lengths = [len(token) for token in sent]

        if len(lengths) > 1:
            mi_scores.append(sklearn.metrics.mutual_info_score(lengths, importance[i]))
            spearman.append(scipy.stats.spearmanr(lengths, importance[i])[0])
            kendall.append(scipy.stats.kendalltau(lengths, importance[i])[0])

    print("---------------")
    print("Length Baseline")
    print("Spearman Correlation: Mean: {:0.2f}, Stdev: {:0.2f}".format(np.nanmean(np.asarray(spearman)),
                                                                       np.nanstd(np.asarray(spearman))))
    # print("Kendall Tau: Mean: {:0.2f}, Stdev: {:0.2f}".format( np.nanmean(np.asarray(kendall)), np.nanstd(np.asarray(kendall))))
    # print("Mutual Information: Mean: {:0.2f}, Stdev: {:0.2f}".format( np.nanmean(np.asarray(mi_scores)), np.nanstd(np.asarray(mi_scores))))
    print("---------------")
    print()
    spearman_mean = np.nanmean(np.asarray(spearman))
    spearman_std = np.nanstd(np.asarray(spearman))
    return spearman_mean, spearman_std


def calculate_freq_baseline(frequencies, importance):
    spearman = []
    kendall = []
    mi_scores = []

    for i in range(len(frequencies)):
        if len(frequencies[i])>0:
            mi_scores.append(sklearn.metrics.mutual_info_score(frequencies[i], importance[i]))
            spearman.append(scipy.stats.spearmanr(frequencies[i], importance[i])[0])
            kendall.append(scipy.stats.kendalltau(frequencies[i], importance[i])[0])

    spearman_mean = np.nanmean(np.asarray(spearman))
    spearman_std = np.nanstd(np.asarray(spearman))
    print("---------------")
    print("Frequency Baseline")
    print("Spearman Correlation: Mean: {:0.2f}, Stdev: {:0.2f}".format(spearman_mean,
                                                                       spearman_std))
    # print("Kendall Tau: Mean: {:0.2f}, Stdev: {:0.2f}".format( np.nanmean(np.asarray(kendall)), np.nanstd(np.asarray(kendall))))
    # print("Mutual Information: Mean: {:0.2f}, Stdev: {:0.2f}".format( np.nanmean(np.asarray(mi_scores)), np.nanstd(np.asarray(mi_scores))))
    print("---------------")
    print()
    return spearman_mean, spearman_std

def calculate_wordclass_baseline(wordclasses, importance):
    ttest = []
    kendall = []
    mi_scores = []

    for i in range(len(wordclasses)):
        if len(wordclasses[i])>0:
            mi_scores.append(sklearn.metrics.mutual_info_score(wordclasses[i], importance[i]))
            le = sklearn.preprocessing.LabelEncoder()
            le.fit(wordclasses[i])
            encoded = le.transform(wordclasses[i])
            ttest.append(scipy.stats.ttest_ind(encoded, importance[i])[0])
            kendall.append(scipy.stats.kendalltau(wordclasses[i], importance[i])[0])
    ttest_mean = np.nanmean(np.asarray(ttest))
    ttest_std = np.nanstd(np.asarray(ttest))
    print("---------------")
    print("Wordclass Baseline")
    print("T-test Correlation: Mean: {:0.2f}, Stdev: {:0.2f}".format(ttest_mean,
                                                                       ttest_std))
    # print("Kendall Tau: Mean: {:0.2f}, Stdev: {:0.2f}".format( np.nanmean(np.asarray(kendall)), np.nanstd(np.asarray(kendall))))
    # print("Mutual Information: Mean: {:0.2f}, Stdev: {:0.2f}".format( np.nanmean(np.asarray(mi_scores)), np.nanstd(np.asarray(mi_scores))))
    print("---------------")
    print()
    return ttest_mean, ttest_std

def calculate_linear_regression(independent, *dependent):
    c = 0
    print(len(independent), len(dependent[0]), len(dependent[1]))
    new_independent = []
    new_dependent = [[] for d in dependent]
    for idx, l in enumerate(independent):
        if any(len(l) != len(d[idx]) for d in dependent):
            continue
        new_independent.append(l)
        for d_idx, d in enumerate(new_dependent):
            try:
                print(dependent[d_idx])
            except:
                import pdb
                pdb.set_trace()
            new_dependent[d_idx].append(dependent[d_idx][idx])
    independent, dependent = new_independent, new_dependent
    for idx, l in enumerate(dependent):
        dependent[idx] = [item for sublist in l for item in sublist]
    X = np.array(dependent).swapaxes(0,1)
    y = np.array([item for sublist in independent for item in sublist])
    #import pdb
    #pdb.set_trace()
    #Y = [[] for y in dependent]

    reg = LinearRegression().fit(X, y)
    r_sq = reg.score(X, y)
    coef, intercept_ =  r_sq.coef_, r_sq.intercept_
    print(coef, intercept_)
    return r_sq

def calculate_permutation_baseline(human_importance, model_importance, num_permutations=100, seed=35):
    all_random_correlations = []
    for i in range(len(human_importance)):
        if not len(human_importance[i]) == len(model_importance[i]):
            pass
            #  print("Alignment Error: " + str(i))
        else:
            # Ignore sentences of length 1
            if len(human_importance[i])>1:
                random_correlations = []
                for k in range(num_permutations):
                    shuffled_importance = random.sample(list(model_importance[i]), len(model_importance[i]))
                    spearman = scipy.stats.spearmanr(shuffled_importance, human_importance[i])[0]
                    random_correlations.append(spearman)
                mean_sentence = np.nanmean(np.asarray(random_correlations))
                all_random_correlations.append(mean_sentence)

    spearman_mean = np.nanmean(np.asarray(all_random_correlations))
    spearman_std = np.nanstd(np.asarray(all_random_correlations))
    print("---------------")
    print("Permutation baseline: Mean: {:0.2f}, stdev: {:0.2f}".format(spearman_mean, spearman_std))
    print("---------------")
    print()
    return spearman_mean, spearman_std
