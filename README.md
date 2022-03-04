## Named Entity Recognition Project

This repository includes code and data to train and compare several Named Entity Recognition Machine Learning systems.    
The project was carried out for the course 'Machine Learning in NLP' at the Vrije Universiteit Amsterdam, taught by Antske Fokkens and José Angel Daza.  
I adapted code provided by Antske Fokkens and José Angel Daza to train and evaluate my systems.   
The data I trained my systems on is from the e CoNLL-2003 Named Entity dataset. This data can also be found in the data folder.   
The repository also includes the report I wrote in which my results can be found.   

The repository is structured as follows:

#### `/code`

This is the main directory for finding notebooks and scripts.  
Further explanations on how to run the code can be found in the README in this folder.  
It contains one subfolder:  
* `/settings` : for configuration files that support the code  

#### `/data`

This filder includes train, dev and test data.

#### `/models`

In this folder, the `'GoogleNews-vectors-negative300.bin.gz'` word embeddings model should be located, but this was excluded for the submission.  
In order to successfully run all code, this model should be placed in this folder, which can be downloaded from https://code.google.com/archive/p/word2vec/

