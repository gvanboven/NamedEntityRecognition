import sys
from collections import defaultdict

feature_to_index = {'Token': 0, 'Pos': 1, 'Gold': 3, 'Cap': 4, 'Number': 5, 'Punct': 6, 'PrevToken':7, 'PrevPOS': 8}

def main(argv=None):
    '''
    Main function that counts for a feature of interest how often each feature value occurs for 
    two labels (target_label and pred_label) in the data

    :param my_arg : a list containing the following parameters:
                    args[1] : the path (str) to the conll file with system predictions, e.g. '..\data\logreg_embeddings.conll'
                    args[2] : the true target label (string), for which the misclassifications are made
                    args[3] : the predicted label (string)
                    args[4] : a feature (str) that is present in the conll input file
    '''
    
    #picking up commandline arguments
    if argv is None:
        argv = sys.argv

    #get relevant info
    file = argv[1]
    target_label = argv[2]
    pred_label = argv[3]
    feature = feature_to_index.get(argv[4])

    feature_counts = defaultdict(lambda: defaultdict(int))
    with open(file, 'r') as f:
        data = list(f)
    for line in data:
        row = line.rstrip('\n').split()
        #count the feature distribution for the labels of interest
        if row[feature_to_index.get('Gold')] == target_label or row[feature_to_index.get('Gold')] == pred_label:
            feature_counts[row[feature_to_index.get('Gold')]][row[feature]] += 1
                   
    #print the feature distribution
    print(f"distribution of feature values for feature {feature} for labels {target_label} and {pred_label}")
    print(dict(feature_counts))

    #get % of occurrence of each feature value
    print("in percentages, the distributions are as follows:")
    for label, feature_values in feature_counts.items():
        d= {feature_value:(round((count/sum(feature_values.values()))*100 ,2)) for feature_value, count in feature_values.items()}
        print(f"{label} : {dict(sorted(d.items(), key=lambda item: item[1], reverse=True))}\n")
if __name__ == '__main__':
    main()