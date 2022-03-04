import sys
from string import punctuation

def get_capitalization(token: str) -> str:
    '''
    Function that the capitalization in the token
    
    :param token: the input token (str)
    
    :return capitalization: a catigorical variable for the type of capitalization present in the token. 
                        Categories are defined as follows:
                            'all': entire word is capitalized
                            'mixed': the word contains capital letters other than the first letter (e.g. eBay)
                            'first': the first letter of the word is capitalized
                            'none': the word is not capitalized
    '''
    #check if entire word is capitalized
    if token.isupper():
        capitalization = 'all'
    #check if there is mixed capitalization
    elif any([char.isupper() for char in token[1:]]):
        capitalization = 'mixed'
    #check if the first letter is capitalized
    elif token[0].isupper():
        capitalization = 'first'
    #else, there is no capitalizion
    else:
        capitalization = 'none'
    return capitalization

def get_contains_number(token: str) -> int:
    '''
    Function that checks if the token contains numbers
    
    :param token: the input token (str)
    
    :return binary variable: 1 if token contains a digit, 0 otherwise
    '''
    # checks if the token contains any digits
    return 1 if any(char.isdigit() for char in token) else 0
    
def get_punctuation(token: str) -> int:
    '''
    Function that checks if the token contains punctution
    
    :param token: the input token (str)
    
    :return binary variable: 1 if token contains , 0 otherwise
    '''
    # checks if the token contains any punctuation
    return 1 if any(p in token for p in punctuation) else 0

def extract_features(inputfile):
    '''
    Function that extracts features
    
    :param inputfile: path (str) to conll inputfile
    
    :return data: list of dictionaries containing feature information
    '''
    data = []
    firstline = True
    #open data
    with open(inputfile, 'r', encoding='utf8') as infile:
        token = '<s>'
        pos = '.'
        for line in infile:
            #ignore first line which contains column numbers
            if firstline:
                firstline = False
                continue
            components = line.rstrip('\n').split()
            #extract features, ignore empty lines
            if len(components) > 1:
                prev_token = token
                prev_pos = pos

                token = components[0]
                capitalized = get_capitalization(token)
                contains_number = get_contains_number(token)
                punct = get_punctuation(token)
                pos = components[1]
                
                #save features
                feature_dict = {'capitalized':capitalized, 
                                'contains_number': str(contains_number), 
                                'punctuation': str(punct),
                                'prev_token': prev_token, 
                                'prev_pos': prev_pos}
                data.append(feature_dict)
    return data

def main(argv=None):
    '''
    Main function extracts features from an inputfile and save each feature in a separate column in an outputfile.

    :param my_arg : a list containing the following parameters:
                    args[1] : the path (str) to the conll input file, for instance the test data: '..\data\conll2003.dev-preprocessed.conll'
                    args[2] : the path (str) to the conll output file, for instance '..\data\conll2003.dev-preprocessed-features.conll'
    '''
    #pick up commandline arguments
    if argv is None:
        argv = sys.argv

    #get input and output files from the arguments    
    inputfile = argv[1]
    outputfile = argv[2]

    #extract and features
    features = extract_features(inputfile)

    #save features to output file
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputfile, 'r'):
        #add column numbers to the first line
        if counter == 0:
            counter += 1
            lastcolumn = line.rstrip('\n')[-1]
            #if the file already contains headers, add headers for the new columns
            try:
                lastcolumn = int(lastcolumn)
                outfile.write(line.rstrip('\n') + 
                                '\t' + str(lastcolumn + 1)  +
                                '\t' + str(lastcolumn + 2)  +
                                '\t' + str(lastcolumn + 3)  +
                                '\t' + str(lastcolumn + 4)  +
                                '\t' + str(lastcolumn + 5)  + '\n')   
      
            #if the file does not contain headers yet, add headers
            except:
                n_columns = len(line.rstrip('\n').split())+5
                outfile.write(''.join([str(i)+'\t' for i in range(n_columns)]) + '\n')
                outfile.write(line.rstrip('\n') + 
                            '\t' + features[counter-1]['capitalized'] + \
                            '\t' + features[counter-1]['contains_number'] + \
                            '\t' + features[counter-1]['punctuation'] + \
                            '\t' + features[counter-1]['prev_token'] + \
                            '\t' + features[counter-1]['prev_pos'] + '\n')
        else:
            #skip empty lines
            if len(line.rstrip('\n').split()) > 1:
                #separate different features in the outputfile by a tab
                outfile.write(line.rstrip('\n') + 
                            '\t' + features[counter-1]['capitalized'] + \
                            '\t' + features[counter-1]['contains_number'] + \
                            '\t' + features[counter-1]['punctuation'] + \
                            '\t' + features[counter-1]['prev_token'] + \
                            '\t' + features[counter-1]['prev_pos'] + '\n')
                counter += 1
    outfile.close()

if __name__ == '__main__':
    main()