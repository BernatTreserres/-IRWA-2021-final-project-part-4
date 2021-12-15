import datetime
import json
from random import random

from faker import Faker

fake = Faker()


# fake.date_between(start_date='today', end_date='+30d')
# fake.date_time_between(start_date='-30d', end_date='now')
#
# # Or if you need a more specific date boundaries, provide the start
# # and end dates explicitly.
# start_date = datetime.date(year=2015, month=1, day=1)
# fake.date_between(start_date=start_date, end_date='+30y')

def get_random_date():
    """Generate a random datetime between `start` and `end`"""
    return fake.date_time_between(start_date='-30d', end_date='now')


def get_random_date_in(start, end):
    """Generate a random datetime between `start` and `end`"""
    return start + datetime.timedelta(
        # Get a random amount of seconds between `start` and `end`
        seconds=random.randint(0, int((end - start).total_seconds())), )


class Document:
    def __init__(self, id, title, description, doc_date, user, likes, rt):
        self.id = id
        self.title = title
        self.description = description
        self.doc_date = doc_date
        self.user = user
        self.likes = likes
        self.rt = rt
        

    def set_title(self, title):
        self.title = title

def load_documents_corpus():
    """
    Load documents corpus from dataset_tweets_WHO.txt file
    :return:
    """

    ##### demo replace ith your code here #####
    docs = {}
    
    docs_path = 'dataset_tweets_WHO.txt'
    with open(docs_path) as fp:
        tweets = json.loads(fp.read())
    
    for tweet_id in list(tweets.keys()):
        tweet = tweets[tweet_id]
        docs[tweet_id] = Document(tweet_id, tweet['full_text'], tweet['full_text'], tweet['created_at'], tweet['user']['screen_name'],
                                  tweet["favorite_count"], tweet["retweet_count"] )
    
    return docs


    
    # for i in range(200):
    #     docs[i] = Document(fake.uuid4(), fake.text(), fake.text(), fake.date_this_year(), fake.email(), fake.ipv4())
    # return docs
