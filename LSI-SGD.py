from string import punctuation, digits

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup

from nltk.stem import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords

from sklearn.linear_model import SGDClassifier 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import LsiModel, TfidfModel
from gensim.matutils import Sparse2Corpus, corpus2dense 

# This is necessary to stablize topic generation
SEED = 42
np.random.seed(SEED)


def preprocess(html):
    """
    Build a tree using beautiful soup, remove all code, then get the all the text.
    """
    soup = BeautifulSoup(html)
    for code in soup.findAll('pre'):
        code.extract()
    return soup.get_text()


# Create an english language stemmer
stemmer = SnowballStemmer('english')

# Create a set of stop words including english stop words and, punctuation, and digits
stopwords = stopwords.words('english') + list(punctuation) + list(digits)
stopwords = set(stopwords)

# This is the punctuation we want to strip off
bad_punct = '.,:;!?\'"'

def tokenize(text):
    """
    Takes raw text and splits it into tokens. These tokens are then cleaned, filtered and stemmed
    """
    tokens = text.split()
    tokens = map(lambda x: x.rstrip(bad_punct).lstrip(bad_punct), tokens)
    tokens = filter(lambda token: not token in stopwords, tokens)
    tokens = map(stemmer.stem, tokens)
    return tokens

# Create a lazy iterator over the file
reader = pd.read_csv('training.csv', iterator=True, chunksize=10000)

# Read each chunk, drop any nulls, and then...
chunks = []
for chunk in reader:
    # In-place doesn't copy the chunk which reduces memory consumption
    chunk.dropna(inplace=True)
    chunks.append(chunk)

# Concatenate each chunk. It is slightly more memory friendly than parsing the whole thing at once.
train = pd.concat(chunks)

# Manually garbage collect the chunks of data before we move one
del(chunks)

# Do the same thing for the testing set
reader = pd.read_csv('testing.csv', iterator=True, chunksize=10000)

chunks = []
for chunk in reader:
    chunk.dropna(inplace=True)
    chunks.append(chunk)

test = pd.concat(chunks)

del(chunks)

# Split the tags by spaces
train_labels = train['Tags'].map(lambda x: x.split())
test_labels = test['Tags'].map(lambda x: x.split())

# The label binarizer takes all the tags and turns them into a big sparse matrix
mlb = MultiLabelBinarizer()
mlb.fit(pd.concat([train_labels, test_labels]))
labels = mlb.transform(train_labels)

# Turn the tokens into a sparse matrix
vect = CountVectorizer(
    # Get text from html
    preprocessor = preprocess,
    # Turn the text into tokens
    tokenizer = tokenize,
    # Generate ngrams
    ngram_range = (1, 2),
    # Remove extremely common tokens
    max_df = 0.5,
    # Remove extremely uncommon tokens
    min_df = 0.001
)

tokens = vect.fit_transform(train['Body'])

# The corpus is simply a wrapper around a sparse array
corpus = Sparse2Corpus(tokens.T)

# Use Gensim's tfidf implementation
# It should yield the same weights as those in SGD.py
tfidf = TfidfModel(corpus)

# This wraps the corpus in a lazy transformation
corpus_tfidf = tfidf[corpus]

# Create the latent semantic analysis transformation 
lsi = LsiModel(corpus_tfidf, num_topics=100)

# This wraps the corpus in another lazy transformation
corpus_lsi = lsi[corpus_tfidf]

def lsi2scipy(docs, num):
    """
    Takes the topics generated by gensim and puts them in a dense scipy array
    """
    topics = np.empty((len(docs), 100))
    count = 0

    # Transformations are actually executed here
    for doc in docs:
        for topic, weight in doc:
            topics[count][topic - 1] = weight
        count += 1
    return topics

# Create and train the classifier
clf = OneVsRestClassifier(SGDClassifier(random_state=SEED), n_jobs=-1)
clf.fit(lsi2scipy(corpus_lsi), labels)
       
labels = mlb.transform(test_labels)

# Do the transformations on the test set
tokens = vect.transform(test['Body'])
corpus = Sparse2Corpus(tokens.T)
corpus_tfidf = tfidf[corpus]
corpus_lsi = lsi[corpus_tfidf]

# Prefice the classes
predicted = clf.predict(lsi2scipy(corpus_lsi))

# Print the final f1 score
print(f1_score(labels, predicted, average='samples'))
