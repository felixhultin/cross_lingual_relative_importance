# Eye-tracking corpora

The experiments for this paper uses several eye-tracking corpora. From these human *relative word importance* are extracted (total reading time) and the same sentences are used to feed pretrained language models and extract *model relative word importance*.

Every corpus has a `data_extractor_<corpus_name>.py` script and contains a `<corpus_name>` folder with two subfolders: `raw` and `relfix`. `raw` contains the raw eye-tracking corpora and `relfix` the `relative word importance` values of each human subject.

## Downloading the raw corpora

### GECO

Download the **GECO** corpus from [https://expsy.ugent.be/downloads/geco/]()
and place the file "MonolingualReadingData.xlsx" in data/geco.

The published version of GECO contains an alignment error which we manually corrected in the file geco/EnglishMaterial_corrected.csv:
The word_id 1-41-87 ("better") has been assigned the wrong sentence id. It is coded as 1-284 but it is actually the last word of sentence 1-283. As a consequence, all following sentence_ids are off by 1. We manually inserted a cell with 1-283. Then all cells below are moved one down for part 1.

### ZuCo

You can download the ZuCo corpus under the following links:
ZuCo 1.0: [https://osf.io/q3zws/]()
ZuCo 2.0: [https://osf.io/2urht/]()
From both repositories, make sure to download the preprocessed files in the “Matlab files” folder of each task (for ZuCo 1.0 “task 1 - SR” and “task 2 - NR”, and for ZuCo 2.0 “task 1 - NR”). There is one file for each participant.

Place the files in the task folders ("zuco1_SR", "zuco1_NR", "zuco2_NR") in data/zuco.

### Russsent

The Russian eye-tracking corpus Russian Sentence Corpus Stimulus (Russsent) can be downloaded on the following link: https://osf.io/x5q2r/



### Potsdam

The German eye-tracking corpus Potsdam can be downloaded on the following http://read.psych.uni-potsdam.de/.

## Adding another corpus

To add another corpus, a data_extractor_<corpus_name>.py script needs to be created which creates the relevant `relfix-feats.cvs` files. Also the corpus and relevant models need to be added to `extract_all.py` and `analyze_all.py`. 
