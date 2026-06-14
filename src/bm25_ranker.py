from rank_bm25 import BM25Okapi

class BM25Engine:
    def __init__(self, tokenized_corpus):
        """
        tokenized_corpus: a list of lists, where each sublist is a tokenized document.
        """
        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_scores(self, tokenized_query):
        """Returns the base probabilistic score for all documents against the query."""
        return self.bm25.get_scores(tokenized_query)