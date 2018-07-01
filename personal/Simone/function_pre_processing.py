    
class Preprocessing:    
    
    def bm25_row(self, X, K1=1.2, B=0.75):
        #Weighs each row of a sparse matrix by OkapiBM25 weighting
        # calculate idf per term (user)
        X = sp.coo_matrix(X)
        N = float(X.shape[0])
        idf = log(N / (1 + bincount(X.col)))
        
        # calculate length_norm per document (artist)
        row_sums = np.ravel(X.sum(axis=1))
        average_length = row_sums.mean()
        length_norm = (1.0 - B) + B * row_sums / average_length
        
        # weight matrix rows by bm25
        X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
        return X.tocsr()
    
    
    def bm25_col(self, X, K1=1.2, B=0.75):
        X = X.T
        X = self.bm25_row(X,K1,B)
        X = X.T
        return X.tocsr()
    
    def tfidf_row(self, X):
        #TFIDF each row of a sparse amtrix
        X = sp.coo_matrix(X)

        # calculate IDF
        N = float(X.shape[0])
        idf = log(N / (1 + bincount(X.col)))

        # apply TF-IDF adjustment
        X.data = sqrt(X.data) * idf[X.col]
        return X.tocsr()
    
       
    def tfidf_col(self, X):
        X = X.T
        X = self.tfidf_row(X)
        X = X.T
        return X.tocsr()