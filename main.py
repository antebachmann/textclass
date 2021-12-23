import nltk
from sklearn.feature_extraction.text import CountVectorizer


def document_features(document):
    word_features = list(all_words)[:2000]
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


def tweet_analysis(document):
    print(document)


good_tweet_list = []
with open('data/train_pos.txt', 'r') as fh:
    for line in fh:
        good_tweet_list.append(line)
neg_tweet_list = []
with open('data/train_neg.txt', 'r') as fh:
    for line in fh:
        neg_tweet_list.append(line)

all_wordsneg = " ".join(x for x in neg_tweet_list if x not in ("?", ".", ";", ":", "!", '"')).split()
all_wordsneg = nltk.FreqDist(w.lower() for w in all_wordsneg)

featuresetsneg = [(document_features(d), -1) for d in neg_tweet_list]

all_words = " ".join(x for x in good_tweet_list if x not in ("?", ".", ";", ":", "!", '"')).split()
all_words = nltk.FreqDist(w.lower() for w in all_words)

featuresets = [(document_features(d), 1) for d in good_tweet_list]

train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
