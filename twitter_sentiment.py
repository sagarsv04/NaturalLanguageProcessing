from tweepy import Stream, API
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import configparser
import time
import text_classification as txt_clf

# read credentials from ini file
config = configparser.ConfigParser()
config.read('./credentials.ini')


# Authenticate
consumer_key    = config['Consumer']['API Key']
consumer_secret = config['Consumer']['API Secret']

access_token    = config['Access']['Token']
access_token_secret = config['Access']['Token Secret']


def get_auth():

	auth = OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)

	return auth


class listener(StreamListener):

	def on_data(self, data):
		try:
			all_data = json.loads(data)

			tweet = all_data["text"]
			sentiment_value, confidence = txt_clf.sentiment(tweet)
			print(tweet, sentiment_value, confidence)

			if confidence*100 >= 80:
				output = open("./twitter-out.txt","a")
				output.write(sentiment_value)
				output.write('\n')
				output.close()
			time.sleep(0.5)
			return True
		except Exception as ex:
			return True

	def on_error(self, status):
		print(status)


twitterStream = Stream(get_auth(), listener())
twitterStream.filter(track=["happy"])
