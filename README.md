# Cross-lingual Comparison of Human and Model Relative Word Importance

This project contains the code for the paper "Cross-lingual Comparison of Human and Model Relative Word
Importance"


### 1 Running the experiments

If all required data files are available (see section 2), simply run: `python analyze_all.py`



If running `analyze_all.py` for the first time, the script will take a bit longer to create two `.csv`-files: `aligned_words.csv` and `human_words.csv`. These contain all word-level information needed to run the experiments, e.g. token, relative importance, word length and word frequency. These are saved in the `results/words` folder and can be used for additional analysis.

Once `analyze_all.py` has finished, a final Excel file will be created: `all_results-<timestamp>.xlsx`. It contains all of the results organized into four tabs.

- **Model Importance**: Correlation results (Spearman R) between human and model relative word importance.
- **Permutation Baselines** Correlation results between model relative word importance and random numbers. Used as a sanity check, but not presented in paper.
- **Corpus statistical baselines** Correlation results (Spearman R) between human and model relative word importance and the two corpus statistical baselines: Word frequency and word length.
- **Regression analysis**: Results of the linear regression analysis (out-of-date: look at section 6).  


### 2. Pre-proccessing steps (optional)


`analyze_all.py` (see section 1) require several data files. The first are intermediate results files, which can be found in the `results` folder. These are files, which align model relative importance to words in a corpus: `<corpus>_<hf_modelpath>_<importance type>.txt`), human relative importance `<corpus>_relfix_averages.txt`. These are created by running (`extract_all.py`).

### 3. Adding a corpus.

TODO

### 4. Generating plots

Run `python analysis/create_plots.py` on the Excel file (see section 1) to create the respective plots. 

### 5. Folder structure

- **extract_human_fixations**: code to extract the relative fixation duration from two eye-tracking corpora and average it over all subjects. The two corpora are [GECO](https://expsy.ugent.be/downloads/geco/) and [ZUCO](https://osf.io/q3zws/).

- **extract_model_importance**: code to extract saliency-based and attention-based importance from transformer-based language models.

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

For the more fine-grained analyses (POS-tags, word frequencies), you need to download the English spaCy model en_core_web_md to your virtual environment:  
`python -m spacy download en_core_web_md`

### Acknowledgements

A large part of the code base of https://github.com/beinborn/relative_importance has been re-purposed for this project.
