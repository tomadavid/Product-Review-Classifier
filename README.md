# Amazon Product Review Sentiment Analysis
This projects trains a model that performs sentiment analysis over product reviews and outputs its result as being "Positive" or "Negative".

The model was trained using the Amazon Reviews dataset, with a training set containing one million comments and its respective label.

# Explanation
Review comments are tokenized and all stopwords are removed.
Each token is then converted to a word2vec vector (100 dimensions).
Each review comment is represented by a vector calculated by averaging the w2v vectors of it's tokens.
We Logistic Regression as a classifier, thus assuming there exists linear division in the vector space between the classes (one side with an average negative nature and other with positive)
