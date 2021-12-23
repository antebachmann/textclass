import nltk
import csv
import numpy as np

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def document_features(document):
    all_words = " ".join(x for x in document if x not in ("?", ".", ";", ":", "!", '"')).split()
    all_words = nltk.FreqDist(w.lower() for w in all_words)
    word_features = list(all_words)[:2000]
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


def tweet_analysis(document):
    print(document)

def dothefeatures(address, sentiment):
    neg_tweet_list = []
    with open(address, 'r') as fh:
        for line in fh:
            neg_tweet_list.append(line)

    all_wordsneg = " ".join(x for x in neg_tweet_list if x not in ("?", ".", ";", ":", "!", '"')).split()
    all_wordsneg = nltk.FreqDist(w.lower() for w in all_wordsneg)

    featuresetsneg = [(document_features(d), sentiment) for d in neg_tweet_list]

    return featuresetsneg, neg_tweet_list


#test addresses
featuresetspos, _ = dothefeatures('data/train_pos.txt', 1)
featuresetsneg, _ = dothefeatures('data/train_neg.txt', -1)
featuresets=featuresetspos+featuresetsneg



train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

#test work
_, testtweets, = dothefeatures('data/test_data.txt', 0)
ids=list(range(1,len(testtweets)+1))

ypred=nltk.classify(classifier,testtweets )
name="done.csv"
create_csv_submission(ids, ypred, name)