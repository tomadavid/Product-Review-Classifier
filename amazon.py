import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords


class Model:
    def __init__(self, vector_size=100, window=5, min_count=1):

        self.w2v_model = None
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.classifier = LogisticRegression()

    # preprocessing a review
    def preprocessing(self, text):

        stop_words = stopwords.words('english')
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return words
    
    # w2v train
    def train_word2vec(self, review_tokens):

        self.w2v_model = Word2Vec(sentences=review_tokens, 
                         vector_size=self.vector_size, 
                         window=self.window, 
                         min_count=self.window,  
                         workers=4)
        
    # w2v conversion
    def w2v(self, word):
        if word in self.w2v_model.vw:
            return self.w2v_model.wv[word]
        else:
            return np.zeros(100)
    
    # train classifier
    def train_classifier(self, reviews, labels):
        
        x = self.average_reviews([self.preprocessing(review) for review in reviews])

        self.classifier.fit(x, labels)

    # predict
    def predict(self, reviews):

        x = self.average_reviews([self.preprocessing(review) for review in reviews])

        predicted = []
        for review in x:
            rev = np.array(x)
            predicted += self.classifier.predict(rev.reshape(1, -1))
        
        return predicted

        
    # averages all w2v of a review to obtain a vector representing the review
    def rev2vec(self, review):

        zeros = np.zeros(100)

        average = []
        word_count = 0
        
        for word in review:
            if len(average) == 0:
                average = word
            else:
                average = np.add(average, word)
            if word != zeros:
                word_count += 1

        average = average/word_count

        return average

    # average reviews
    def average_reviews(self, reviews):

        averaged_reviews = []

        for review in reviews:
            w2v_mat = []
            for word in review:
                w2v_mat += [self.w2v(word)]
            averaged_reviews += [self.rev2vec(w2v_mat)]

        return averaged_reviews
    
