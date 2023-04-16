import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

class BM25(object):
    def __init__(self, b=0.99, k1=1.6, ngram_range=(1, 1)):
        self.vectorizer = TfidfVectorizer(max_df=0.90, min_df=1,
                                          use_idf=True,
                                          ngram_range=ngram_range,)

        print(ngram_range)
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])

        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)

        return (numer / denom).sum(1).A1

## Evaluation
#Micro Precision Function
def micro_prec(true_list,pred_list,k):
    #define list of top k predictions
    cor_pred = 0
    #top1 candidate is always the query itself, so start from 1
    top_k_pred = pred_list[1:k+1].copy()
    #iterate throught the top k predictions
    for doc in top_k_pred:
        #if document in true list, then increment count of relevant predictions
        if doc in true_list:
            cor_pred += 1
    #return total_relevant_predictions_in_top_k/k
    return cor_pred, k  