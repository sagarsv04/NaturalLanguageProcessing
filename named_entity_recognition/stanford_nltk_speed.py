import nltk
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from nltk import pos_tag
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize


java_path = "C:/Program Files/Java/jdk1.8.0_172/bin/java.exe"
os.environ['JAVAHOME'] = java_path


# Process text
def process_text():
	raw_text = open("./news_article.txt").read()
	token_text = word_tokenize(raw_text)
	return token_text

# Stanford NER tagger
def stanford_tagger(token_text):
	st = StanfordNERTagger('../stanford_ner/classifiers/english.all.3class.distsim.crf.ser.gz',
							'../stanford_ner/stanford-ner.jar', encoding='utf-8')
	ne_tagged = st.tag(token_text)
	return(ne_tagged)

# NLTK POS and NER taggers
def nltk_tagger(token_text):
	tagged_words = nltk.pos_tag(token_text)
	ne_tagged = nltk.ne_chunk(tagged_words)
	return(ne_tagged)

def stanford_main():
	print(stanford_tagger(process_text()))
	return 0

def nltk_main():
	print(nltk_tagger(process_text()))
	return 0

def time_plot(stanford_total_time, nltk_total_time):

	style.use('fivethirtyeight')
	N = 1
	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars
	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, stanford_total_time, width, color='r')
	rects2 = ax.bar(ind+width, nltk_total_time, width, color='y')

	# Add text for labels, title and axes ticks
	ax.set_xlabel('Classifier')
	ax.set_ylabel('Time (in seconds)')
	ax.set_title('Speed by NER Classifier')
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


if __name__ == '__main__':
	stanford_t0 = time.clock()
	stanford_main()
	stanford_t1 = time.clock()
	stanford_total_time = stanford_t1 - stanford_t0

	nltk_t0 = time.clock()
	nltk_main()
	nltk_t1 = time.clock()
	nltk_total_time = nltk_t1 - nltk_t0

	time_plot(stanford_total_time, nltk_total_time)
