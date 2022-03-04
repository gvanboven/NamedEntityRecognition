import sys
from collections import defaultdict, Counter

#get indices for information of interest
TOKEN_COLUMN = 0
GOLD_COLUMN = 3
PRED_COLUMN = 9

def main(argv=None):
    '''
    Main function analyses errors of mixups from a target label as another (pred_label) label

    :param my_arg : a list containing the following parameters:
                    args[1] : the path (str) to the conll file with system predictions, e.g. '..\data\logreg_embeddings.conll'
                    args[2] : the true target label (string), for which the misclassifications are made
                    args[3] : the predicted label (string)
    '''
    #picking up commandline arguments
    if argv is None:
        argv = sys.argv
    
    file = argv[1]
    target_label = argv[2]
    pred_label = argv[3]

    mixups = defaultdict(int)

    print(f"Sentences in which token with the label {target_label} are classified as {pred_label} are:")
    firstline = True
    with open(file, 'r') as f:
        data = list(f)
    for i, line in enumerate(data):
        #skip the first line, here the header is located
        if firstline:
            firstline = False
            continue
        row = line.rstrip('\n').split()
        #retrieve classification mistakes
        if row[GOLD_COLUMN] != row[PRED_COLUMN] :
                #save mistake if the confusion is between the correct labels
                if row[GOLD_COLUMN] == target_label and row[PRED_COLUMN] == pred_label:
                    mixups[row[TOKEN_COLUMN]] += 1
                    #print sentece of the mistake
                    print(" ".join([x.rstrip('\n').split()[0] for x in data[i-10:i+10]]))

    #print the tokens for this type of error and their counts
    print(f"\n tokens + their counts, for which this error is made are:")
    print(mixups)

    #print the number of total mistakes for this error type
    mixup_words = len(mixups.keys())
    total_mixup_count = sum(mixups.values())
    print(f"total mixup words: {mixup_words}, together creating {total_mixup_count} mistakes")

    #for all the words for wich mistakes of this type are made, count for each of their true labels, how often they are predicted as each prediction label
    instances = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
    sents = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))

    prev_pos_mistakes, prev_pos_correct = [],[]

    for i, line in enumerate(data):
        row = line.rstrip('\n').split()
        if row[TOKEN_COLUMN] in mixups.keys() :
            instances[row[TOKEN_COLUMN]][row[GOLD_COLUMN]][row[PRED_COLUMN]] += 1
            sents[row[TOKEN_COLUMN]][row[GOLD_COLUMN]][row[PRED_COLUMN]].append(f"{data[i-1].split()[0]} {data[i-1].split()[8]} {row[0]} ({row[1]}, {row[4]})")
               # " ".join([x.rstrip('\n').split()[0] for x in data[i-2:i+1]]))
            if row[GOLD_COLUMN] == row[PRED_COLUMN] == target_label:
                prev_pos_correct.append(data[i-1].split()[8])
            if row[GOLD_COLUMN] == target_label and row[PRED_COLUMN] == pred_label:
                prev_pos_mistakes.append(data[i-1].split()[8])
    
    # print("correct:")
    # print(Counter(prev_pos_correct))
    # print("mistake:")
    # print(Counter(prev_pos_mistakes))


    print(dict(sents))
    print("MISTAKES:")
    for token, labels in dict(sents).items():
        if token == 'of':
            for label, instance in labels.items():
                for prediction, string in instance.items():
                    if label == target_label and pred_label == prediction:
                        print(f"{label} - {prediction} : {string}")
    print()
    print("CORRECT:")
    for token, labels in dict(sents).items():
        if token == 'of':
            for label, instance in labels.items():
                for prediction, string in instance.items():
                    if label == prediction == target_label:
                        print(f"{label} : {string}")

    mixups_with_pred_as_true = 0
    more_pred_than_true = 0
    also_correct = []
    also_correct_count = 0 
    correct_as_target = 0

    #go through all the mistake  tokens
    for key, true in instances.items():
        # count how often the mistake tokens also have the mixup label in the gold
        if pred_label in true.keys():
            mixups_with_pred_as_true += 1
            #count how often the mixup label is more common for mistake tokens in the gold
            if sum(true[pred_label].values()) > sum(true[target_label].values()):
                more_pred_than_true += 1

        for truelabel, prediction in true.items():
            #count how often these mistake tokens also are classified correctly 
            if truelabel == target_label:
                if target_label in prediction:
                    also_correct.append(key* prediction[target_label])
                    also_correct_count += 1
                    #count the total number of correct predictions for this token for the target label
                    correct_as_target += prediction[target_label]

    #print an overall overview
    print(f"mixup words that also have the predicted label {pred_label}: {round(100*(mixups_with_pred_as_true/mixup_words),2)}%")
    print(f"mixup words where the true label {pred_label} occurs more often than target {target_label}: {round(100*(more_pred_than_true/mixup_words),2)}%")
    print(f"out of {mixup_words} words that are mistaken, {also_correct_count} words are also classified correctly as {target_label}")
    print(f"while there are {total_mixup_count} misclassifications for {target_label} as {pred_label}, {correct_as_target} times in total {target_label} is classified correctly")
    print(also_correct)
if __name__ == '__main__':
    main()