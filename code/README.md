## Overview

This repository provides notebooks and scripts for the course 'Machine Learning in NLP'.  
This directory is the main directory for finding notebooks and scripts.   
It has one subfolder:
* `/settings` : for configuration files that support the code 

--------------

### Scripts:

`feature_extraction.py` is used to extract features from a conll data file with the path `[data_input_path]`, which are saved as new columns in a conll file with path `'[data_output_path]'`. The script can be called as follows:  
`python .\feature_extraction.py '[data_input_path]', '[data_output_path]'`  
as for example:  
`python .\feature_extraction.py '..\data\conll2003.dev-preprocessed.conll', '..\data\conll2003.dev-preprocessed-features.conll'`  
The following features are extracted and saved for each token: 
 * capitalization (all characters / the first character / mixed / none), 
 * whether the token includes numbers (1 / 0)
 * whether the token includes punctuation (1 / 0)
 * the previous token
 * the previous POS tag  
 For each feature, a column is added to the input data including this feature, and this is saved in the output file.  
 I extracted features from the files `'conll2003.train.conll'` and `'..\data\conll2003.dev-preprocessed.conll'`, and use their output files to train my models on.
-----------

`ner_machine_learning.py` can be used to train models. Calling this file with its arguments should look as follows:  
`python` `'.\ner_machine_learning.py'` `'[train_file_path]'` `'[test_file_path]'` `'[model_name]'` `'[training_setting]'`  
as for instance:  
`python '.\ner_machine_learning.py' '..\data\conll2003.train-features.conll' '..\data\conll2003.dev-preprocessed-features.conll' 'SVM' 'extended'`  
Here the `[train_file_path]` is the path to the conll training data, and `[test_file_path]` is the path to the conll test data. In order to use all possible features, features should first be extracted from the original conll train/dev/test data using the script `feature_extraction.py`.

There are 3 `[model_name]`'s to select from:
* `'logreg'` for a logistic regression classifier
* `'SVM'` for a linear SVM
* `'NB'` for a complement Naive Bayes classifier

Further there are 3 options for the `training_settings`:
* in the `basic` setting, only the token (one-hot) is included
* in the `extended` setting the token (one-hot), POS, capitalization, numerical, punctuation, previous token and previous POS are included
* the `embeddings` setting is similar to the `extended` setting, but the token is represented by word embeddings and the previous token is excluded. **note:** In order to use the embeddings, the file `'GoogleNews-vectors-negative300.bin.gz'` should be added to the `..\models` folder 

The file further has two optional arguments :   
* `[selected_features]` is a string of features to be included in the model, separated by white spaces. I used this script for my feature ablation study this way, by incrementally selecting a different combination of features.
* `[output_path]` the path where the output file (input file + a column with predictions) should be saved to.
**Note:** if one of these additional features is defined, the other should be defined as well.  
If the `[output_path]` is not defined the output path will be `'../data/[model]_[training_setting].conll'`. The output file is a copy of the `[test_file_path]` file, with a column is added containing the predictions of the model.

An example way to call this function for an ablation experiment (in this case, where the feature `capitatlization` is excluded) would look as follows:  
`python '.\ner_machine_learning.py' '..\data\conll2003.train-features.conll', '..\data\conll2003.dev-preprocessed-features.conll' 'SVM' 'extended' 'Token Pos  Number Punct PrevToken PrevPOS' '..\data\SVM_extended_noCap.conll'`  

-----------

`error_analysis.py`: this file gives an overview of quantitative information about the confusions for a true `[target_label]` predicted as a `'[predicted_label]'` by a model whose predictions are to be found at the path `'[model_annotations_path]'`. The script can be called by:  
`python .\error_analysis.py '[model_annotations_path]' '[target_label]' '[predicted_label]'`  
For example:  
`python .\error_analysis.py '..\data\SVM_extended.conll' 'I-ORG' 'I-PER'`

-----------

`error_analysis_featurecount_perlabel.py` counts for a feature of interest `'[feature]'` how often each feature value occurs for two labels (`'[label1]'` and `'[label2]'`) in the data.

`python .\error_analysis_featurecount_perlabel.py '[model_annotations_path]' '[label1]' '[label2]' '[feature]'`
For example:  
`python .\error_analysis_featurecount_perlabel.py '..\data\SVM_embeddings.conll' 'B-ORG' 'B-LOC' 'Cap'`

-----------
### Notebooks

`Preprocessing_conll.ipynb`:  this file can be used to preprocess the labels in a file containing machine annotations, so that they match the labels. I used this notebook to preprocess the files `..\data\conll2003.dev.conll`, `..\data\stanford_out.dev.conll` and `..\data\spacy_out.dev.conll`. Their preprocessed files are called `..\data\conll2003-preprocessed.dev.conll`, `..\data\stanford_out.dev-preprocessed.conll` and `..\data\spacy_out.dev-preprocessed.conll`.

-----------
`basic_system.ipynb`: The code in this file creates a logistic regression model using the token itself (one-hot encoded) as its only feature.

-----------

`basic_evaluation.ipynb`: this notebook can be used to evaluate the performance of a model, using a file with the gold labels and a file with machine annotations. In this file, the results of the models I trained can also be found. 



