from collections import defaultdict
from os import terminal_size
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import json
import math
import re
import numpy as np
import collections
from numpy import linalg as la
import re

### TEXT PROCESSING ###
def remove_emoji(text):
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002700-\U000027BF"  # Dingbats
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_symbols_and_links(text):
    text = re.sub('https://.*', '', text) # to remove the links
    text = re.sub('http://.*', '', text)
    for ch in ['&',':','.',',',';','…','-','!','?','¿','amp','rt','"',"'"]:
        if ch in text:
            text = text.replace(ch, '')
    return text

def build_terms(line, removeStopWords = True):
    """
    Preprocess the article text (title + body) removing stop words, stemming,
    transforming in lowercase and return the tokens of the text.
    
    Argument:
    line -- string (text) to be preprocessed
    
    Returns:
    line - a list of tokens corresponding to the input text after the preprocessing
    """

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    ## START CODE
    line= line.lower() ## Transform in lowercase
    line= remove_emoji(line) ## remove emojis
    line= remove_symbols_and_links(line) ## remove symbols and links
    line= line.split() ## Tokenize the text to get a list of terms
    hashtags= [l.replace('#', '') for l in line if '#' in l] # we separate the hashtags
    if removeStopWords:
        line= [l for l in line if l not in stop_words and l.replace('#', '') not in hashtags and len(l)>1] ## eliminate the stopwords (HINT: use List Comprehension)
    line= [stemmer.stem(l) for l in line] ## perform stemming for the keywords
    hashtags= [stemmer.stem(l) for l in hashtags] ## perform stemming for the hashtags
    ## END CODE
    return line, hashtags

### END TEXT PROCESSING ###

### INDEXING ###

def create_index(tweets):
    """
    Implement the inverted index
    
    Argument:
    lines -- collection of Wikipedia articles
    
    Returns:
    index - the inverted index (implemented through a Python dictionary) containing terms as keys and the corresponding
    list of documents where these keys appears in (and the positions) as values.
    """
    keywords_index = defaultdict(list)
    hashtags_index = defaultdict(list)

    for tweet_id in tweets.keys():  # Remember, lines contain all documents
        tweet = tweets[tweet_id].description
        keywords, hashtags = build_terms(tweet)
         
        ## ===============================================================        
        ## create the index for the current page and store it in current_page_index (current_page_index)
        ## current_page_index ==> { ‘term1’: [current_doc, [list of positions]], ...,‘term_n’: [current_doc, [list of positions]]}

        ## Example: if the curr_doc has id 1 and his text is 
        ##"web retrieval information retrieval":

        ## current_page_index ==> { ‘web’: [1, [0]], ‘retrieval’: [1, [1,4]], ‘information’: [1, [2]]}

        ## the term ‘web’ appears in document 1 in positions 0, 
        ## the term ‘retrieval’ appears in document 1 in positions 1 and 4
        ## ===============================================================

        current_tweet_keyword_index = {}
        current_tweet_hashtag_index = {}

        for position, keyword in enumerate(keywords): # terms contains page_title + page_text. Loop over all terms
            try:
                # if the term is already in the index for the current page (current_page_index)
                # append the position to the corresponding list
                
                current_tweet_keyword_index[keyword][1].append(position)  
                
            except:
                # Add the new term as dict key and initialize the array of positions and add the position
                current_tweet_keyword_index[keyword]=[tweet_id, [position]] #'I' indicates unsigned int (int in Python)
            
        #merge the current page index with the main index
        for keyword, info in current_tweet_keyword_index.items():
            keywords_index[keyword].append(info)
        
        
        for position, hashtag in enumerate(hashtags): # terms contains page_title + page_text. Loop over all terms
            try:
                # if the term is already in the index for the current page (current_page_index)
                # append the position to the corresponding list
                
                current_tweet_hashtag_index[hashtag][1].append(position) 
                
            except:
                # Add the new term as dict key and initialize the array of positions and add the position
                current_tweet_hashtag_index[hashtag]=[tweet_id, [position]] #'I' indicates unsigned int (int in Python)
            
        #merge the current page index with the main index
        for hashtag, info in current_tweet_hashtag_index.items():
            hashtags_index[keyword].append(info)
        
                    
    return keywords_index, hashtags_index

### END INDEXING ###

### TF-IDF ###
def create_index_tfidf(num_tweets, index):
    """
    Implement the inverted index and compute tf, df and idf
    
    Argument:
    lines -- collection of Wikipedia articles
    numDocuments -- total number of documents
    
    Returns:
    index - the inverted index (implemented through a python dictionary) containing terms as keys and the corresponding 
    list of document these keys appears in (and the positions) as values.
    tf - normalized term frequency for each term in each document
    df - number of documents each term appear in
    idf - inverse document frequency of each term
    """ 
        
    tf=defaultdict(list) #term frequencies of terms in documents (documents in the same order as in the main index)
    df=defaultdict(int)         #document frequencies of terms in the corpus
    idf=defaultdict(float)
        
    #normalize term frequencies
    # Compute the denominator to normalize term frequencies (formula 2 above)
    # norm is the same for all terms of a document.
    norm=0
    for term, values in index.items(): 
        # posting is a list containing doc_id and the list of positions for current term in current document: 
        # posting ==> [currentdoc, [list of positions]] 
        # you can use it to inferr the frequency of current term.
        for tweet in values:
            norm+=len(tweet[1])**2 # numbers of appearances of a term 
    norm=math.sqrt(norm)


    #calculate the tf(dividing the term frequency by the above computed norm) and df weights
    for term, values in index.items():     
        # append the tf for current term (tf = term frequency in current doc/norm)
        for tweet in values:
            tf[term].append(np.round(len(tweet[1])/norm,4))  ## SEE formula (1) above
        #increment the document frequency of current term (number of documents containing the current term)
        df[term]= len(index[term])  # increment df for current term


    # Compute idf following the formula (3) above. HINT: use np.log
    for term in df:
        if df[term] != 0:
            idf[term] = np.round(np.log(float(num_tweets/df[term])),4)
        else:
            idf[term] = 0
            
    return tf, df, idf

### END TF-IDF ###

### EVALUATION ###
def rank_tweets(terms, tweets, index, idf, tf):
    """
    Perform the ranking of the results of a search based on the tf-idf weights
    
    Argument:
    terms -- list of query terms
    docs -- list of documents, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies
    title_index -- mapping between page id and page title
    
    Returns:
    Print the list of ranked documents
    """

    # I'm interested only on the element of the docVector corresponding to the query terms 
    # The remaining elements would became 0 when multiplied to the query_vector
    tweet_vectors = defaultdict(lambda: [0] * len(terms)) # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)


    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query. 
    # Example: collections.Counter(["hello","hello","world"]) --> Counter({'hello': 2, 'world': 1})
    #HINT: use when computing tf for query_vector

    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index:
            continue

        ## Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex]= query_terms_count[term]/query_norm * idf[term]
        
        # Generate doc_vectors for matching docs
        for tweet_index, (tweet, postings) in enumerate(index[term]):
            # Example of [doc_index, (doc, postings)]
            # 0 (26, array('I', [1, 4, 12, 15, 22, 28, 32, 43, 51, 68, 333, 337]))
            # 1 (33, array('I', [26, 33, 57, 71, 87, 104, 109]))
            # term is in doc 26 in positions 1,4, .....
            # term is in doc 33 in positions 26,33, .....

            #tf[term][0] will contain the tf of the term "term" in the doc 26            
            if tweet in tweets:
                tweet_vectors[tweet][termIndex] = tf[term][tweet_index] * idf[term]  # TODO: check if multiply for idf

    # Calculate the score of each doc 
    # compute the cosine similarity between queryVector and each tweetVector:
    # HINT: you can use the dot product because in case of normalized vectors it corresponds to the cosine similarity
    # see np.dot
    
    tweet_scores=[[np.dot(curTweetVec, query_vector), tweet] for tweet, curTweetVec in tweet_vectors.items() ]
    tweet_scores.sort(reverse=True)
    result_tweets = [x[1] for x in tweet_scores]
    #print document titles instead if document id's
    #result_docs=[ title_index[x] for x in result_docs ]
    # if len(result_tweets) == 0:
    #     print("No results found, try again")
    #     query = input()
    #     docs = search_tf_idf(query, index)
    #print ('\n'.join(result_docs), '\n')
    return result_tweets

def search_tf_idf(query, index):
    """
    output is the list of documents that contain any of the query terms. 
    So, we will get the list of documents for each query term, and take the union of them.
    """
    query = build_terms(query)[0]
    tweets = set()
    for term in query:
        try:
            # store in term_tweets the ids of the tweets that contain "term"                        
            term_tweets = []
            for posting in index[term]:
                term_tweets.append(posting[0])
            
            # tweets = tweets Union term_tweets
            tweets = tweets.union(term_tweets)
        except:
            #term is not in index
            pass
    tweets = list(tweets)
    tf_k, df_k, idf_k = create_index_tfidf(len(tweets), index)
    ranked_tweets = rank_tweets(query, tweets, index, idf_k, tf_k)
    return ranked_tweets

### END EVALUATION ###


def rankTweets(tweets, search_query):
    res = []

    keywords_index, hashtags_index = create_index(tweets)
    ranked_tweets = search_tf_idf(search_query, keywords_index)

    for tweet_id in ranked_tweets:
        title = []
        for term in build_terms(search_query, False)[0]:
            words = build_terms(tweets[tweet_id].title, False)[0]
            if term in words:
                term_pos = words.index(term)
                for pos in range(term_pos -4, term_pos+5):
                    if pos >= 0 and pos < len(words):
                        title.append(tweets[tweet_id].title.split()[pos])
        title = " ".join(title) 
        date = tweets[tweet_id].doc_date.split()
        date = " ".join([date[0], date[1], date[2], date[5], date[3]])

        tweets[tweet_id].set_title(title)

        res.append(TweetInfo(tweet_id, title, tweets[tweet_id].description, date, tweets[tweet_id].user, "doc_details?id={}&param1=1&param2=2".format(tweet_id)))

    return res



class TweetInfo:
    def __init__(self, id, title, description, doc_date, user, link):
        self.id = id
        self.title = title
        self.description = description
        self.doc_date = doc_date
        self.user = user
        self.url = link