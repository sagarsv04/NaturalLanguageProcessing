import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords, state_union
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer



################################ Tokenize ################################

EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."
stop_words = set(stopwords.words("english"))

word_tokens = word_tokenize(EXAMPLE_TEXT)

filtered_sentence = [w for w in word_tokens if not w in stop_words]


################################ Stemming ################################

ps = PorterStemmer()
# snowboll , lemetizer
new_text = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
words = word_tokenize(new_text)
example_words = ["python","pythoner","pythoning","pythoned","pythonly"]
# stemmed = [ps.stem(w) for w in example_words]
stemmed = [ps.stem(w) for w in words]


################################ Part of Speech Tagging ################################

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
	try:
		for i in tokenized[:5]:
			words = word_tokenize(i)
			tagged = nltk.pos_tag(words)
			print(tagged)
	except Exception as e:
		print(str(e))

process_content()


################################ Chunking ################################

sentence = [("a","DT"), ("beautiful","JJ"), ("young","JJ"), ("lady","NN"), ("is","VBP"), ("walking","VBP"), ("on","IN"), ("the","DT"), ("Pavement","NN")]
grammar = "NP: {<DT>?<JJ>*<NN>}"

parser = nltk.RegexpParser(grammar)
output = parser.parse(sentence)
output.draw()


train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
	try:
		for i in tokenized[:5]:
			words = word_tokenize(i)
			tagged = nltk.pos_tag(words)
			chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
			chunkParser = nltk.RegexpParser(chunkGram)
			chunked = chunkParser.parse(tagged)
			chunked.draw()
			print(tagged)
	except Exception as e:
		print(str(e))

process_content()


################################ Chinking ################################

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
	try:
		for i in tokenized[:5]:
			words = word_tokenize(i)
			tagged = nltk.pos_tag(words)
			chinkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""
			chinkParser = nltk.RegexpParser(chinkGram)
			chinked = chinkParser.parse(tagged)
			chinked.draw()
			print(tagged)
	except Exception as e:
		print(str(e))

process_content()
