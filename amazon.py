import nltk
import pandas as pd
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

# preprocessing a review
def preprocessing(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return words


data = pd.read_csv('train.csv')
data = pd.DataFrame(data)

print(data.columns)