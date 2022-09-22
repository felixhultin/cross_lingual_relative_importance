# Cross-lingual Comparison of Human and Model Relative Word Importance

This project contains the code for the paper "Cross-lingual Comparison of Human and Model Relative Word
Importance"


### 1 Running the experiments

If running `analyze_all.py` for the first time, the script will take a bit longer to create two `.csv`-files: `aligned_words.csv` and `human_words.csv`. These contain all word-level information needed to run the experiments, e.g. token, relative importance, word length and word frequency. These are saved in the `results/words` folder and can be used for additional analysis.

Once `analyze_all.py` has finished, a final Excel file will be created: `all_results-<timestamp>.xlsx`. It contains all of the results organized into four tabs.

- **Model Importance**: Correlation results (Spearman R) between human and model relative word importance.
- **Permutation Baselines** Correlation results between model relative word importance and random numbers. Used as a sanity check, but not presented in paper.
- **Corpus statistical baselines** Correlation results (Spearman R) between human and model relative word importance and the two corpus statistical baselines: Word frequency and word length.
- **Regression analysis**: Results of the linear regression analysis (out-of-date: look at section 5).  


### 2. Pre-proccessing (optional)

`analyze_all.py` (see section 1) require several intermediate result files to run. These can be found in the `results` folder and consist of files with the following formats `<corpus>_<hf_modelpath>_<importance type>.txt`, which align model relative importance to words in a corpus, `<corpus>_relfix_averages.txt`, which align words with human relative importance, and `<corpus>_sentences` which are simply the sentences in the corpus.

These files are created by running `extract_all.py`, which use the files created by the data extractor scripts in the folder `extract_human_fixations`. See its specific [README](extract_human_fixations/README.md) for information on how to add a new corpus or re-run the scripts.

### 3. Generating plots

Run `python -m analysis.create_plots all_results-<timestamp>.xlsx` on the Excel file (see section 1) to create the respective plots. The plots are saved in the `plots` folder.

### 4. Folder structure

- **extract_human_fixations**: code to extract the relative fixation duration from the five eye-tracking corpora and average it over all subjects.

- **extract_model_importance**: code to extract saliency-based, attention-based importance from transformer-based language models.

- **analysis**: code to compare and analyze patterns of importance in the human fixation durations and the model data. Also contains code to replicate the plots in the paper.

- **plots**: contains all plots.

- **results**: contains intermediate results.

### 4. Requirements

Python should be <= 3.8.

We use the following packages (see requirements.txt):  
numpy (1.19.5), tensorflow (2.4.1), transformers (4.2.2), scikit-learn (0.22.2), spacy (2.3.5), wordfreq (2.3.2), scipy (1.4.1)

Note that later versions of transformers might lead to errors.

To install, create and activate a virtual environment and run:  
`pip3 install -r requirements.txt`

### 5. Regression analysis using Linear mixed models (LMM)

Since LMMs are not readily available in Python, the results of the regression analysis in the paper was done in Stata. To run the script, run `stata mixed-effects/lmm.do`. If you want to create the plots, run `convert_tables_to_results.py`, which will create a `conversion.xlxs` Excel file. Move the `with_reffect` tab of the `conversion.xlsx` Excel file to the `Regression analysis` tab of the original `all_results-<timestamp>.xlsx` Excel file. Then you can run `python -m analysis.create_plots all_results-<timestamp>.xlsx`.

### Acknowledgements

A large part of the code base of https://github.com/beinborn/relative_importance has been re-purposed for this project. Alexander Koplenig wrote the `mixed-effects/lmm.do` file.

### Paper and citation

The paper can be found here: https://aclanthology.org/2022.clasp-1.2/. In BibTex, use the  citation below.

```
@inproceedings{morger-etal-2022-cross,
    title = "A Cross-lingual Comparison of Human and Model Relative Word Importance",
    author = "Morger, Felix  and
      Brandl, Stephanie  and
      Beinborn, Lisa  and
      Hollenstein, Nora",
    booktitle = "Proceedings of the 2022 CLASP Conference on (Dis)embodiment",
    month = sep,
    year = "2022",
    address = "Gothenburg, Sweden",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.clasp-1.2",
    pages = "11--23",
    abstract = "Relative word importance is a key metric for natural language processing. In this work, we compare human and model relative word importance to investigate if pretrained neural language models focus on the same words as humans cross-lingually. We perform an extensive study using several importance metrics (gradient-based saliency and attention-based) in monolingual and multilingual models, including eye-tracking corpora from four languages (German, Dutch, English, and Russian). We find that gradient-based saliency, first-layer attention, and attention flow correlate strongly with human eye-tracking data across all four languages. We further analyze the role of word length and word frequency in determining relative importance and find that it strongly correlates with length and frequency, however, the mechanisms behind these non-linear relations remain elusive. We obtain a cross-lingual approximation of the similarity between human and computational language processing and insights into the usability of several importance metrics.",
}
```
