from nltk.tag import StanfordNERTagger
from nltk.metrics.scores import accuracy
from nltk.tokenize import word_tokenize
from nltk.internals import config_java
from nltk import pos_tag, ne_chunk, chunk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# download link for Stanford NER tagger
# https://nlp.stanford.edu/software/CRF-NER.shtml#Download
# to use you will require Java v1.8+.

# doesnt worked for me
# config_java("C:/Program Files/Java/jdk1.8.0_172/bin/java.exe")

java_path = "C:/Program Files/Java/jdk1.8.0_172/bin/java.exe"
os.environ['JAVAHOME'] = java_path


####################### Basic Example using Stanford NER Tagger #######################


# using 3 class model for recognizing locations, persons, and organizations
st = StanfordNERTagger('./stanford_ner/classifiers/english.all.3class.distsim.crf.ser.gz',
					   './stanford_ner/stanford-ner.jar', encoding='utf-8')


text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'

tokenized_text = word_tokenize(text)
classified_text = st.tag(tokenized_text)

print(classified_text)


####################### Testing NLTK and Stanford NER Taggers for Accuracy #######################


raw_annotations = open("./wikigold.conll.txt", encoding="utf8").read()
split_annotations = raw_annotations.split()

# Amend class annotations to reflect Stanford's NERTagger
for n,i in enumerate(split_annotations):
	if i == "I-PER":
		split_annotations[n] = "PERSON"
	if i == "I-ORG":
		split_annotations[n] = "ORGANIZATION"
	if i == "I-LOC":
		split_annotations[n] = "LOCATION"

# Group NE data into tuples
def group(lst, n):
	for i in range(0, len(lst), n):
		val = lst[i:i+n]
		if len(val) == n:
			yield tuple(val)

reference_annotations = list(group(split_annotations, 2))

pure_tokens = split_annotations[::2]


def test_nltk_classifier():
	tagged_words = pos_tag(pure_tokens)
	nltk_unformatted_prediction = ne_chunk(tagged_words)
	# Convert prediction to multiline string and then to list (includes pos tags)
	multiline_string = chunk.tree2conllstr(nltk_unformatted_prediction)
	listed_pos_and_ne = multiline_string.split()

	# Delete pos tags and rename
	del listed_pos_and_ne[1::3]
	listed_ne = listed_pos_and_ne

	# Amend class annotations for consistency with reference_annotations
	for n,i in enumerate(listed_ne):
		if i == "B-PERSON":
			listed_ne[n] = "PERSON"
		if i == "I-PERSON":
			listed_ne[n] = "PERSON"
		if i == "B-ORGANIZATION":
			listed_ne[n] = "ORGANIZATION"
		if i == "I-ORGANIZATION":
			listed_ne[n] = "ORGANIZATION"
		if i == "B-LOCATION":
			listed_ne[n] = "LOCATION"
		if i == "I-LOCATION":
			listed_ne[n] = "LOCATION"
		if i == "B-GPE":
			listed_ne[n] = "LOCATION"
		if i == "I-GPE":
			listed_ne[n] = "LOCATION"

	# Group prediction into tuples
	nltk_formatted_prediction = list(group(listed_ne, 2))
	nltk_accuracy = accuracy(reference_annotations, nltk_formatted_prediction)
	print(nltk_accuracy)
	return nltk_accuracy


def test_stanford_classifier():

	st = StanfordNERTagger('./stanford_ner/classifiers/english.all.3class.distsim.crf.ser.gz',
							'./stanford_ner/stanford-ner.jar', encoding='utf-8')
	stanford_prediction = st.tag(pure_tokens)
	stanford_accuracy = accuracy(reference_annotations, stanford_prediction)
	print(stanford_accuracy)
	return stanford_accuracy


def compare_accuracy(nltk_accuracy, stanford_accuracy):

	style.use('fivethirtyeight')
	N = 1
	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig, ax = plt.subplots()

	stanford_percentage = stanford_accuracy * 100
	rects1 = ax.bar(ind, stanford_percentage, width, color='r')

	nltk_percentage = nltk_accuracy * 100
	rects2 = ax.bar(ind+width, nltk_percentage, width, color='y')

	# add some text for labels, title and axes ticks
	ax.set_xlabel('Classifier')
	ax.set_ylabel('Accuracy (by percentage)')
	ax.set_title('Accuracy by NER Classifier')
	ax.set_xticks(ind+width)
	ax.set_xticklabels( ('') )

	ax.legend( (rects1[0], rects2[0]), ('Stanford', 'NLTK'), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

	def autolabel(rects):
		# attach some text labels
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x()+rect.get_width()/2., 1.02*height, '%10.2f' % float(height),
					ha='center', va='bottom')

	autolabel(rects1)
	autolabel(rects2)

	plt.show()


nltk_accuracy = test_nltk_classifier()
stanford_accuracy = test_stanford_classifier()
compare_accuracy(nltk_accuracy, stanford_accuracy)
