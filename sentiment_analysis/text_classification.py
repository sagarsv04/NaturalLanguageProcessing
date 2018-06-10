import nltk
import os
import random
import pickle
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


if not os.path.exists("../pickled/"):
	os.mkdir("../pickled/")


class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)

		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf


if not(os.path.exists("../pickled/documents.pickle") and os.path.exists("../pickled/word_features5k.pickle")):
	short_pos = open("../positive.txt","r").read()
	short_neg = open("./negative.txt","r").read()
	# move this up here
	all_words = []
	documents = []
	# j is adject, r is adverb, and v is verb
	# allowed_word_types = ["J","R","V"]
	allowed_word_types = ["J"]
	for p in short_pos.split('\n'):
	    documents.append( (p, "pos") )
	    words = word_tokenize(p)
	    pos = nltk.pos_tag(words)
	    for w in pos:
	        if w[1][0] in allowed_word_types:
	            all_words.append(w[0].lower())

	for p in short_neg.split('\n'):
	    documents.append( (p, "neg") )
	    words = word_tokenize(p)
	    pos = nltk.pos_tag(words)
	    for w in pos:
	        if w[1][0] in allowed_word_types:
	            all_words.append(w[0].lower())

	save_documents = open("../pickled/documents.pickle","wb")
	pickle.dump(documents, save_documents)
	save_documents.close()

	all_words = nltk.FreqDist(all_words)

	word_features = list(all_words.keys())[:5000]

	save_word_features = open("../pickled/word_features5k.pickle","wb")
	pickle.dump(word_features, save_word_features)
	save_word_features.close()
else:
	save_documents_file = open("../pickled/documents.pickle", "rb")
	documents = pickle.load(save_documents_file)
	save_documents_file.close()

	word_features_file = open("../pickled/word_features5k.pickle", "rb")
	word_features = pickle.load(word_features_file)
	word_features_file.close()


def find_features(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)
	return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

# set that we'll train our classifier with
training_set = featuresets[:10000]

# set that we'll test against.
testing_set = featuresets[10000:]


if not os.path.exists("../pickled/naivebayes.pickle"):
	classifier = nltk.NaiveBayesClassifier.train(training_set)
	print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
	save_classifier = open("../pickled/naivebayes.pickle","wb")
	pickle.dump(classifier, save_classifier)
	save_classifier.close()
else:
	open_file = open("../pickled/naivebayes.pickle", "rb")
	classifier = pickle.load(open_file)
	open_file.close()


classifier.show_most_informative_features(15)


if not os.path.exists("../pickled/MNB_classifier.pickle"):
	MNB_classifier = SklearnClassifier(MultinomialNB())
	MNB_classifier.train(training_set)
	print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
	save_classifier = open("../pickled/MNB_classifier.pickle","wb")
	pickle.dump(classifier, save_classifier)
	save_classifier.close()
else:
	open_file = open("../pickled/MNB_classifier.pickle", "rb")
	MNB_classifier = pickle.load(open_file)
	open_file.close()


if not os.path.exists("../pickled/BernoulliNB_classifier.pickle"):
	BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
	BernoulliNB_classifier.train(training_set)
	print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
	save_classifier = open("../pickled/BernoulliNB_classifier.pickle","wb")
	pickle.dump(classifier, save_classifier)
	save_classifier.close()
else:
	open_file = open("../pickled/BernoulliNB_classifier.pickle", "rb")
	BernoulliNB_classifier = pickle.load(open_file)
	open_file.close()


if not os.path.exists("../pickled/LogisticRegression_classifier.pickle"):
	LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
	LogisticRegression_classifier.train(training_set)
	print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
	save_classifier = open("../pickled/LogisticRegression_classifier.pickle","wb")
	pickle.dump(classifier, save_classifier)
	save_classifier.close()
else:
	open_file = open("../pickled/LogisticRegression_classifier.pickle", "rb")
	LogisticRegression_classifier = pickle.load(open_file)
	open_file.close()


if not os.path.exists("../pickled/SGDClassifier_classifier.pickle"):
	SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
	SGDClassifier_classifier.train(training_set)
	print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
	save_classifier = open("../pickled/SGDClassifier_classifier.pickle","wb")
	pickle.dump(classifier, save_classifier)
	save_classifier.close()
else:
	open_file = open("../pickled/SGDClassifier_classifier.pickle", "rb")
	SGDClassifier_classifier = pickle.load(open_file)
	open_file.close()


if not os.path.exists("../pickled/LinearSVC_classifier.pickle"):
	LinearSVC_classifier = SklearnClassifier(LinearSVC())
	LinearSVC_classifier.train(training_set)
	print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
	save_classifier = open("../pickled/LinearSVC_classifier.pickle","wb")
	pickle.dump(classifier, save_classifier)
	save_classifier.close()
else:
	open_file = open("../pickled/LinearSVC_classifier.pickle", "rb")
	LinearSVC_classifier = pickle.load(open_file)
	open_file.close()


if not os.path.exists("../pickled/NuSVC_classifier.pickle"):
	NuSVC_classifier = SklearnClassifier(NuSVC())
	NuSVC_classifier.train(training_set)
	print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
	save_classifier = open("../pickled/NuSVC_classifier.pickle","wb")
	pickle.dump(classifier, save_classifier)
	save_classifier.close()
else:
	open_file = open("../pickled/NuSVC_classifier.pickle", "rb")
	NuSVC_classifier = pickle.load(open_file)
	open_file.close()


voted_classifier = VoteClassifier(classifier, NuSVC_classifier, LinearSVC_classifier, SGDClassifier_classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier)


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)


print(sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))
print(sentiment("This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"))
