# GramTypix
> :warning: The docs are still being completed. In particular, example data will be provided soon, as well as scripts to do hierarchical clustering of languages.

## Project structure
```
gramtypix/
├── notebooks/
│   ├── gmm_clustering.ipynb
│   ├── ngram_analysis.ipynb
│   └── semantic_mapping.ipynb
├── src/
│   ├── analysis/
│   │   └── ngram_analysis.py
│   ├── utils/
│   │   └── text_processing.py
│   ├── ngram_detection_main.py  # main script for char n-gram detection
│   └── semantic_mapping_main.py  # main script for semantic mapping/Kriging
├── requirements.txt
└── README.md
```

## Install requirements

> NB: tested on Python 3.10.3.

Install the requirements (use virtual environment if possible):

```
pip install -r requirements.txt
```

## Define your variables

Variables including the paths to the datasets, the source and target language codes, and the words being investigated must be defined in the `config.yaml` file provided under /src/. Follow the comments in the file for descriptions of what each variable indicates. 

## Datasets required

1) A CSV file (to be defined in `advdf_path`), where each line is an occurrence of the source_token of interest in language A (e.g. English), and each column is the parallel in other languages, besides a `sent_id` column (arbitrary, as long as internally meaningful and unique for each context) and a `context` column (with the source text).
2) A list of target languages in their ISO-code, whose column you want to refine. You'll get both a copy of the original CSV file with the respective columns modified (and the others left untouched) and a copy of the same but only with the `sent_id`, `context`, source language, and target languages that were modified.
3) The alignment models in a CSV format, with the first column being a `sent_id` that can be mapped to the CSV in `advdf_path`, the second column being a `context` column corresponding to the `sent_id` and the third column a mapping from source to target language context in the format word1 (parallel1), word2 (parallel2), etc. For instance, the header of the alignment model for [aai] is:

```
sent_id,context,targ
40001003,and judah the father of perez and zerah by tamar and perez the father of hezron and hezron the father of ram,and (naatu) judah (judah) the (NOMATCH) father (NOMATCH) of (natun) perez (perez) and (naatu) zerah (zerah) by (NOMATCH) tamar (natunatun) and (naatu) perez (perez) the (NOMATCH) father (NOMATCH) of (natun) hezron (NOMATCH) and (naatu) hezron (NOMATCH) the (NOMATCH) father (NOMATCH) of (natun) ram (ram) 
```

The location of the alignment models is to be defined in `alignments_parent_path`.

:warning: Make sure that 'empty'/null-token alignments are marked as 'NOMATCH' rather than 'NULL' or other variations, to clearly distinguish them from empty parallels due to the lack of target text (as opposed to lack of a lexical counterpart).

You can define some stopwords on the source language, which will automatically be associated with the respective target tokens, to ensure they are not mistaken as parallels to your word of interest if they occur very often.

## Scripts and notebooks

Notebooks are a good place to start to play around.
1) `ngram_analysis.ipynb`: check this first if you want to refine a parallel dataset for a word with n-gram information (morphological markers). The input dataset should be as defined in the previous section.
2) `semantic_mapping.ipynb`: check this after running the other notebook. Or it can be used to generate semantic maps/kriging directly from the parallel dataset.

Else, you can run:

```
python ngram_detection_main.py

```

if you want to refine your parallel dataset with n-gram associations or 

```
python semantic_mapping_main.py

```

if you are ready to generate semantic maps, either based on the initial parallel dataset or the refined one.

In both scripts you'll need to adjust the variables which define paths to the data according to the environment where you're running the scripts (in the `config.yaml` file).
