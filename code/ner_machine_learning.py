#import models
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn import svm

import sys
import numpy as np
import csv

#import gensim to deal with word embeddings
import gensim


#defines the column in which each feature is located
feature_to_index = {'Token': 0, 'Pos': 1, 'Gold': 3, 'Cap': 4, 'Number': 5, 'Punct': 6, 'PrevToken':7, 'PrevPOS': 8}

def extract_feature_values(row, selected_features):
    '''
    Function that extracts feature value pairs from row
    
    :param row: row from conll file
    :param selected_features: list of selected features
    
    :type row: string
    :type selected_features: list of strings

    :returns: dictionary of feature value pairs
    '''
    feature_values = {}
    #only extract selected features
    for feature_name in selected_features:
        #get index of current feature to extract it
        r_index = feature_to_index.get(feature_name)
        feature_values[feature_name] = row[r_index]
        
    return feature_values

def extract_word_embedding(token, word_embedding_model):
    '''
    Function that returns the word embedding for a given token out of a distributional semantic model.
    If the token is not present in the embeddings model, a 300-dimension vector of 0s is returned.
    
    :param token: the token
    :param word_embedding_model: the distributional semantic model
    :type token: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :returns a vector representation of the token
    '''
    if token in word_embedding_model:
        vector = word_embedding_model[token]
    else:
        vector = np.zeros(300)
    return vector

def create_vectorizer_traditional_features(feature_values):
    '''
    Function that creates vectorizer for set of feature values, and vectorizes these
    
    :param feature_values: list of dictionaries containing feature-value pairs
    :type feature_values: list of dictionairies (key and values are strings)
    
    :returns vectorizer: vectorizer fitted on feature values
    :returns vec_feature_values: vectorized feature values
    '''
    vectorizer = DictVectorizer()

    vec_feature_values = vectorizer.fit_transform(feature_values)
    return vectorizer, vec_feature_values

def combine_sparse_and_dense_features(dense_vectors, sparse_features):
    '''
    Function that takes sparse and dense feature representations and appends their vector representation
    
    :param dense_vectors: list of dense vector representations
    :param sparse_features: list of sparse vector representations
    :type dense_vector: list of arrays
    :type sparse_features: list of lists
    
    :returns: list of arrays in which sparse and dense vectors are concatenated
    '''
    
    combined_vectors = []
    sparse_vectors = np.array(sparse_features.toarray())
    
    #concatenate sparse and dense vectors
    for index, vector in enumerate(sparse_vectors):
        combined_vector = np.concatenate((vector,dense_vectors[index]))
        combined_vectors.append(combined_vector)
    return combined_vectors

def extract_embeddings_as_features_and_gold(conllfile, word_embedding_model, selected_features):
    '''
    Function that extracts features and gold labels using word embeddings
    
    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :param selected_features: the features to include in the models
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    :type selected_features: list
    
    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    :return vectorizer: vectorizer fitted on feature values
    '''
    ### This code was partially inspired by code included in the HLT course, obtained from https://github.com/cltl/ma-hlt-labs/, accessed in May 2020.
    labels = []
    features = []
    embedding_features = []
    sparse_features = []

    #if the set of features to include is not specified, select all of them except the next token
    if selected_features == None:
        selected_features = ['Pos', 'Cap', 'Number', 'Punct', 'PrevPOS']
    
    #read inputfile
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    firstline = True
    for row in csvreader:
        #skip the first line, as this contains the headers of the columns
        if firstline:
            firstline = False
            continue
        #check for cases where empty lines mark sentence boundaries (which some conll files do).
        if len(row) > 3:
            #extract embedding for the token
            embedding = extract_word_embedding(row[feature_to_index.get('Token')], word_embedding_model)

            #extract other features
            feature_dict = extract_feature_values(row, selected_features)

            embedding_features.append(embedding)
            sparse_features.append(feature_dict)
            labels.append(row[feature_to_index.get('Gold')])
    print('data extracted')
    
    #vectorize sparse features
    vectorizer, sparse_vec_features = create_vectorizer_traditional_features(sparse_features)
    print('sparse features vectorized')

    #combine embeddings and sparse features
    features = combine_sparse_and_dense_features(embedding_features, sparse_vec_features)
    return features, labels, vectorizer

def extract_embeddings_as_features(conllfile, vectorizer, word_embedding_model, selected_features):
    '''
    Function that extracts features and gold labels using word embeddings
    
    :param conllfile: path to conll file
    :param vectorizer: vectorizer fitted on training data
    :param word_embedding_model: a pretrained word embedding model
    :param selected_features: the features to include in the model
    :type conllfile: string
    :type vectorizer: sklearn.feature_extraction._dict_vectorizer.DictVectorizer
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    :type selected_features: list
    
    :return features: list of vector representation of tokens
    '''
    features = []
    embedding_features = []
    sparse_features = []

    #if the set of features to include is not specified, select all of them
    if selected_features == None:
        selected_features = ['Pos', 'Cap', 'Number', 'Punct', 'PrevPOS']
    
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    firstline = True
    for row in csvreader:
        #skip the first line, as this contains the headers of the columns
        if firstline:
            firstline = False
            continue
        #check for cases where empty lines mark sentence boundaries (which some conll files do).
        if len(row) > 3:
            embedding = extract_word_embedding(row[0], word_embedding_model)
            features.append(embedding)
            #extract other features
            feature_dict = extract_feature_values(row, selected_features)

            embedding_features.append(embedding)
            sparse_features.append(feature_dict)
    #vectorize sparse features
    sparse_features_vectors = vectorizer.transform(sparse_features)

    #combine embeddings and sparse features
    features = combine_sparse_and_dense_features(embedding_features, sparse_features_vectors)
    return features

def extract_features_and_labels(trainingfile, selected_features, setting='extended'):
    '''
    Function that extracts features and gold labels
    
    :param inputfile: path to conll inputfile
    :param selected_features: the features to include in the model
    :param setting: setting that defines the features that are included in case they are not exlpicitly defined.
                    In the 'basic' setting only the token is extracted, in the 'extended' setting all features are
    :type trainingfile: string
    :type selected_features: list
    :type setting: string
    
    :return vectorized_data: list of dictionaries containing feature information
    :return targets: list of gold labels
    :return vectorizer: vectorizer fitted on training data
    '''
    data = []
    targets = []

    #if the set of features to include is not specified, select the ones that belong to the setting
    if selected_features == None:
        if setting == 'extended':
            selected_features = ['Token', 'Pos', 'Cap', 'Number', 'Punct', 'PrevToken', 'PrevPOS']
        if setting == 'basic':
            selected_features = ['Token']

    #open data
    firstline = True
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            #skip the first line, as this contains the headers of the columns
            if firstline:
                firstline = False
                continue
            row = line.rstrip('\n').split()
            #skip empty lines, extract features according to setting
            if len(row) > 0:
                feature_dict = extract_feature_values(row, selected_features)
                data.append(feature_dict)
                #extract gold
                targets.append(row[feature_to_index.get('Gold')])
    
    #vectorize the features
    vectorizer, vecotrized_data = create_vectorizer_traditional_features(data)

    return vecotrized_data, targets, vectorizer
    
def extract_features(inputfile, vectorizer, selected_features, setting='extended'):
    '''
    Function that extracts features
    
    :param inputfile: path to conll inputfile
    :param vectorizer: vectorizer fitted on training data
    :param selected_features: the features to include in the model
    :param setting: setting that defines the number of features that are saved. 
                    In the 'basic' setting only the token is extracted, in the 'extended' setting all features are
    :type inputfile: string
    :type vectorizer: sklearn.feature_extraction._dict_vectorizer.DictVectorizer
    :type selected_features: list
    :type setting: string
    
    :return vectorized_data: list of dictionaries containing feature information
    '''
    data = []

    #if the set of features to include is not specified, select the ones that belong to the setting
    if selected_features == None:
        if setting == 'extended':
            selected_features = ['Token', 'Pos', 'Cap', 'Number', 'Punct', 'PrevToken', 'PrevPOS']
        if setting == 'basic':
            selected_features = ['Token']
    #open data
    firstline = True
    with open(inputfile, 'r', encoding='utf8') as infile:
        for line in infile:
            #skip the first line, as this contains the headers of the columns
            if firstline:
                firstline = False
                continue
            row = line.rstrip('\n').split()
            #skip empty lines, extract features according to setting
            if len(row) > 0:
                feature_dict = extract_feature_values(row, selected_features)
                data.append(feature_dict)

    #vectorize the features
    vecotrized_data = vectorizer.transform(data)
    return vecotrized_data
    
def create_classifier(train_features, train_targets, modelname):
    '''
    Function that creates a classifier trained on the training data
    
    :param train_features: vectorized training features
    :param train_targets: gold labels for training
    :param modelname: the type of classifier to be trained

    :type train_features: list
    :type train_targets: list of strings (gold labels)
    :type modelname: str
    
    :return model: trained classifier
    '''
   #select and create requested model
    if modelname ==  'logreg':
         model = LogisticRegression(max_iter=10000)
    elif modelname == 'SVM':
        #create a LINEAR SVM
        model = svm.LinearSVC()
    elif modelname == 'NB':
        #complement NB version of Naive Bayes classifier because this method is suitable for text data and 
        # usually outperforms the other method that suits text data
        model = ComplementNB()

    #train model
    model.fit(train_features, train_targets)
    return model
    
    
def classify_data(model, test_features, testfile, outputfile):
    '''
    Function that makes prediction on the test data, given a trained model and saves the predictions to the outputfile
    
    :param model: trained classifier
    :param test_features: vectorized test features
    :param inputdata: path to conll file containing test data
    :param outputfile: path to conll output file

    :type model: sklearn model
    :type test_features: list
    :type inputdata: string
    :type outputfile: string
    '''

    #make predictions
    predictions = model.predict(test_features)

    #save predictions 
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(testfile, 'r'):
        #add header to the newly added column of predictions
        if counter == 0:
            lastcolumn = int(line.rstrip('\n')[-1])
            new_colum = str(lastcolumn + 1)
            outfile.write(line.rstrip('\n') + '\t' + new_colum  + '\n')
            counter += 1
        else:
            #add predictions in a new column
            if len(line.rstrip('\n').split()) > 0:
                outfile.write(line.rstrip('\n') + '\t' + predictions[counter-1] + '\n')
                counter += 1
    outfile.close()

def main(argv=None):
    '''
    Main function that creates and trains a model on provided training data, classifies the test data, and saves the predictions.

    :param my_arg : a list containing the following parameters:
                    args[1] : the path (str) to the conll training data 
                    args[2] : the path (str) to the conll test data 
                    args[3] : the name (str) of the model to be trained
                    args[4] : the setting (str) for the features to be included
                                in the 'basic' setting, only the token is included and is one-hot encoded
                                in the 'extended' setting, all features in the data are included
                                in the 'embeddings' setting, the token is included as a dense vector, and all other features except the previous token are included
                    then there are two optional parameters:
                    args[5] : a string of selected features (separated by enters), to test a combination of features different from the settings
                    args[6] : path (str) to output conll file. if not defined a standard location is used.
    '''
    #picking up commandline arguments
    if argv is None:
        argv = sys.argv

    trainingfile = argv[1]
    testfile = argv[2]
    model = argv[3]
    setting = argv[4]
    #if features to include and output path are defined, extract them. otherwise use standard values
    try:
        selected_features = argv[5].split()
        print(selected_features)
        outfile_path = argv[6]
        print(outfile_path)
    except:
        selected_features = None
        outfile_path = '../data/' + model + '_' + setting + '.conll'

    print(f"start training {model} model {setting} setting")

    #extract training and test data
    if setting == 'embeddings':
        ## load word embeddings
        language_model = gensim.models.KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin.gz', binary=True)
        #extract train and test featers and gold labels
        training_features, gold_labels, vec = extract_embeddings_as_features_and_gold(trainingfile, language_model, selected_features)
        test_features =  extract_embeddings_as_features(testfile, vec, language_model, selected_features)
    else:        
        #extract train and test featers and gold labels
        training_features, gold_labels, vec = extract_features_and_labels(trainingfile, selected_features, setting)
        test_features =  extract_features(testfile, vec, selected_features, setting)

    #build and train model
    ml_model = create_classifier(training_features, gold_labels, model)
    print('done training')
    #classify test data and save the predictions
    classify_data(ml_model, test_features, testfile, outfile_path)
    print('done classifying')
    
if __name__ == '__main__':
    main()
