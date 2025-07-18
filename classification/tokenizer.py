import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

stop_words = set(stopwords.words('russian'))
punctuation = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

def tokenize(text):
    tokens = nltk.word_tokenize(text.lower())
    return [lemmatizer.lemmatize(t) for t in tokens
                if t not in stop_words and t not in punctuation ]