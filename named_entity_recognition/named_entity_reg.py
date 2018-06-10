from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

import os


# download link for Stanford NER tagger
# https://nlp.stanford.edu/software/CRF-NER.shtml#Download
# to use you will require Java v1.8+.

# doesnt worked for me
# config_java("C:/Program Files/Java/jdk1.8.0_172/bin/java.exe")

java_path = "C:/Program Files/Java/jdk1.8.0_172/bin/java.exe"
os.environ['JAVAHOME'] = java_path


####################### Basic Example using Stanford NER Tagger #######################


# using 3 class model for recognizing locations, persons, and organizations
st = StanfordNERTagger('../stanford_ner/classifiers/english.all.3class.distsim.crf.ser.gz',
					   '../stanford_ner/stanford-ner.jar', encoding='utf-8')


text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'

tokenized_text = word_tokenize(text)
classified_text = st.tag(tokenized_text)

print(classified_text)
