import os
from json import JSONEncoder

import httpagentparser  # for getting the user agent as json
import nltk
import time
from flask import Flask, render_template, session
from flask import request
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import MaxNLocator 

from app.analytics.analytics_data import AnalyticsData, Click
from app.search_engine.load_corpus import load_corpus
from app.search_engine.objects import Document, StatsDocument
from app.core import utils
from app.search_engine.search_engine import SearchEngine

# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default

# end lines ***for using method to_json in objects ***

# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = 'afgsreg86sr897b6st8b76va8er76fcs6g8d7'
# open browser dev tool to see the cookies
app.session_cookie_name = 'IRWA_SEARCH_ENGINE'

# instantiate our search engine
searchEngine = SearchEngine()

# instantiate our in memory persistence
analytics_data = AnalyticsData()

# print("current dir", os.getcwd() + "\n")
# print("__file__", __file__ + "\n")
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
# print(path + ' --> ' + filename + "\n")
# load documents corpus into memory.
file_path = path + "/dataset_tweets_WHO.txt"

# file_path = "../../tweets-data-who.json"
corpus = utils.load_documents_corpus()
global last_query_id
print("loaded corpus. first elem:", list(corpus.values())[0])
queries_tweets = {}



@app.route('/')
@app.route('/')
def search_form():
    return render_template('index.html', page_title="Welcome")

@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']

    results = searchEngine.search(search_query, corpus)
    queries = analytics_data.get_queries()
    found_count = len(results)

    if search_query not in queries.keys():
        analytics_data.set_queries(search_query, len(search_query.split()), found_count, [time.time()], 0, len(queries)+1)
    
    else:
        if len(queries[search_query][2]) >= 1:
            last_time_spent = queries[search_query][2][-1]-queries[search_query][2][0]
            analytics_data.set_last_time(search_query, last_time_spent)

        analytics_data.set_query_time(search_query, time.time())


    for i in range(len(results)): #to store for all documents, the query in where they appeared
        queries_tweets[results[i].id] = [i+1, search_query] # we store the order of ranking and the query

    last_query = analytics_data.last_query

    if  analytics_data.last_query != "":
        file = open('data/queries.csv', 'r')
        file = file.readlines()

        with open('data/queries.csv', 'a') as fd:
            if len(queries[last_query][2]) < 1:
                fd.write('\n'+str(len(file) + 1)+','+last_query+','+str(queries[last_query][0])+','+str(found_count)+','+ str(queries[last_query][3]))
            else:
                fd.write('\n'+str(len(file) + 1)+','+last_query+','+str(queries[last_query][0])+','+str(found_count)+','+ str(queries[last_query][2][-1] - queries[last_query][2][0]))

        with open('data/tweets.csv', 'a') as fd:
            for i in range(len(results)):
                fd.write('\n'+str(results[i].id)+','+str(i+1)+','+last_query)

    analytics_data.set_last_query(search_query)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count)


@app.route('/doc_details', methods=['GET'])
def doc_details():
    # getting request parameters:
    # user = request.args.get('user')
    clicked_doc_id = request.args["id"]
    analytics_data.set_clicks(clicked_doc_id)
    tweet = corpus[clicked_doc_id]

    last_query = analytics_data.get_last_query()
    analytics_data.append_query_time(last_query, time.time())
    file = pd.read_csv('data/clicks.csv')

    if clicked_doc_id in file.tweet_id.values:
        file['clicks'][file['Query ID']== clicked_doc_id] += 1
        file.to_csv('data/clicks.csv')
        file.close()
    else:
        with open('data/clicks.csv', 'a') as fd:
            fd.write('\n'+str(clicked_doc_id)+','+str(1))

    print("click in id={} - fact_clicks len: {}".format(clicked_doc_id, len(analytics_data.fact_clicks)))

    return render_template('doc_details.html', id=clicked_doc_id, tweet=tweet)


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with dashboard ###
    :return:
    """
    ### Start replace with your code ###
    last_query = analytics_data.get_last_query()
    queries = analytics_data.get_queries()
    analytics_data.append_query_time(last_query, time.time())


    #return render_template('stats.html', data_queries=data_queries, data_tweets=data_tweets, data_clicks = data_clicks)
    return render_template('stats.html', clicks_data= analytics_data.fact_clicks, queries = queries, queries_tweets=queries_tweets)
    ### End replace with your code ###

@app.route('/sentiment')
def sentiment_form():
    return render_template('sentiment.html')

@app.route('/dashboard')
def dashboard():
    last_query = analytics_data.get_last_query()
    queries = analytics_data.get_queries()
    analytics_data.append_query_time(last_query, time.time())

    keys = queries.keys()
    terms = []
    results = []
    times = []
    ids = []
    for key in keys:
        terms.append(queries[key][0])
        results.append(queries[key][1])
        if len(queries[key][2]) < 1:
            times.append(queries[key][3])
        else:
            times.append(queries[key][2][-1]-queries[key][2][0] + queries[key][3])
        ids.append(queries[key][4])

    sns.countplot(x = terms)
    plt.xlabel("number of query terms") 
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))  
    plt.savefig('static/terms.png')
    plt.cla()  

    sns.lineplot(x=ids, y =times)
    plt.xlabel("query id")
    plt.ylabel("time spent (s)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('static/times.png')
    plt.cla()

    sns.lineplot(x=ids, y =results)
    plt.xlabel("query id")
    plt.ylabel("number of results")    
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('static/n_results.png')
    plt.cla()

    return render_template('dashboard.html', queries=queries)

@app.route('/sentiment', methods=['POST'])
def sentiment_form_post():
    text = request.form['text']
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    score = ((sid.polarity_scores(str(text)))['compound'])
    return render_template('sentiment.html', score=score)


if __name__ == "__main__":
    data_queries = pd.read_csv('data/queries.csv')
    data_tweets = pd.read_csv('data/tweets.csv')
    data_clicks = pd.read_csv('data/clicks.csv')

    for idx, row in data_queries.iterrows():
        analytics_data.set_queries(row[1], row[2], row[3], [], row[4], row[0])

    for idx, row in data_clicks.iterrows():
        analytics_data.fact_clicks[row[0]] = row[1]

    for idx, row in data_tweets.iterrows():
        queries_tweets[row[0]] = [row[1], row[2]] 

    app.run(port="8088", host="0.0.0.0", threaded=False, debug=True)

    
