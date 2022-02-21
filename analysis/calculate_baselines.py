import sklearn
import numpy as np
import scipy.stats
import random
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Here we calculate length, frequency, and permutation baselines on the sentence level
# Note that the correlation functions yield a warning if one of the list is constant (stdev = 0)
# For example, the phrase "you did not" would yield the length vector [3,3,3]
# and then correlation cannot be calculated
def calculate_len_baseline(tokens, importance):
    pvalues = []
    spearman = []
    kendall = []
    mi_scores = []

    for i, sent in enumerate(tokens):
        lengths = [len(token) for token in sent]

        if len(lengths) > 1:
            mi_scores.append(sklearn.metrics.mutual_info_score(lengths, importance[i]))
            corr, pvalue = scipy.stats.spearmanr(lengths, importance[i])
            spearman.append(corr)
            pvalues.append(pvalue)
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
    pvalues_mean = np.nanmean(np.asarray(pvalues))
    pvalues_std = np.nanmean(np.asarray(pvalues))
    return spearman_mean, spearman_std, pvalues_mean, pvalues_std


def calculate_freq_baseline(frequencies, importance):
    pvalues = []
    spearman = []
    kendall = []
    mi_scores = []

    for i in range(len(frequencies)):
        if len(frequencies[i])>0:
            mi_scores.append(sklearn.metrics.mutual_info_score(frequencies[i], importance[i]))
            corr, pvalue = scipy.stats.spearmanr(frequencies[i], importance[i])
            spearman.append(corr)
            pvalues.append(pvalue)
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
    pvalues_mean = np.nanmean(np.asarray(pvalues))
    pvalues_std = np.nanmean(np.asarray(pvalues))
    return spearman_mean, spearman_std, pvalues_mean, pvalues_std

def calculate_wordclass_baseline(wordclasses, importance):
    pvalues = []
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

def calculate_linear_regression(dependent, *independent, plot=False, title=None):
    new_dependent = []
    new_independent = [[] for ind in independent]
    for idx, l in enumerate(dependent):
        if any(len(l) != len(ind[idx]) for ind in independent):
            continue
        new_dependent.append(l)
        for ind_idx, ind in enumerate(new_independent):
            new_independent[ind_idx].append(independent[ind_idx][idx])
    dependent, independent = new_dependent, new_independent
    for idx, l in enumerate(independent):
        independent[idx] = [item for sublist in l for item in sublist]
    X = np.array(independent).swapaxes(0,1)
    y = np.array([item for sublist in dependent for item in sublist])
    reg = LinearRegression().fit(X, y)
    r_sq = reg.score(X, y)
    if plot:
        y_pred = reg.predict(X)
        X, y = X.flatten(), y.flatten()
        plt.scatter(X, y, color="black", s=3)
        plt.plot(X, y_pred, color="blue", linewidth=1)
        plt.xlabel("human")
        plt.ylabel("model")
        plt.title(title)
        plt.savefig("plots/linear_regression-"+title+".png")
    return r_sq

def calculate_permutation_baseline(human_importance, model_importance, num_permutations=100, seed=35):
    pvalues = []
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
                    spearman, pvalue = scipy.stats.spearmanr(shuffled_importance, human_importance[i])
                    random_correlations.append(spearman)
                    pvalues.append(pvalue)
                mean_sentence = np.nanmean(np.asarray(random_correlations))
                all_random_correlations.append(mean_sentence)

    spearman_mean = np.nanmean(np.asarray(all_random_correlations))
    spearman_std = np.nanstd(np.asarray(all_random_correlations))
    print("---------------")
    print("Permutation baseline: Mean: {:0.2f}, stdev: {:0.2f}".format(spearman_mean, spearman_std))
    print("---------------")
    print()
    pvalues_mean = np.nanmean(np.asarray(pvalues))
    pvalues_std = np.nanmean(np.asarray(pvalues))
    return spearman_mean, spearman_std, pvalues_mean, pvalues_std
