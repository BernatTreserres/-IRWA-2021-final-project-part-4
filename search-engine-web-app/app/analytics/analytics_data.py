class AnalyticsData:
    fact_clicks = {}
    queries = {}
    last_query = ""

    def set_query_time(self, query, value):
        self.queries[query][2] = [value]

    def append_query_time(self, query, value):
        self.queries[query][2].append(value)

    def set_last_time(self, query, time):
        self.queries[query][3] = time

    def set_queries(self, query, terms, results, times, last_time, id):
        self.queries[query] = [terms, results, times, last_time, id]
    
    def get_queries(self):
        return self.queries

    def set_last_query(self, query):
        self.last_query = query

    def get_last_query(self):
        return self.last_query

    def set_clicks(self, tweet_id):
        if tweet_id in self.fact_clicks.keys():
            self.fact_clicks[tweet_id] += 1
        else:
            self.fact_clicks[tweet_id] = 1



class Click:
    def __init__(self, doc_id):
        self.doc_id = doc_id
