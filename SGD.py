from string import punctuation, digits

import pandas as pd

from bs4 import BeautifulSoup

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

from sklearn.linear_model import SGDClassifier 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

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

# Create the term frequency inverse document frequency transformer
tfidf = TfidfTransformer()

# Make tokens
tokens = vect.fit_transform(train['Body'])
# Transform those tokens
tokens = tfidf.fit_transform(tokens)

# Train the classifier using all available cpu's
clf = OneVsRestClassifier(SGDClassifier(random_state=8365092), n_jobs=-1)
clf.fit(tokens, labels)

# Use the existing label binarizer to make the tags in the test set into
# A sparse binary matrix
labels = mlb.transform(test_labels)

# Make tokens
tokens = vect.transform(test['Body'])

# Transform the tokens using the same tfidf transformer
tokens = tfidf.transform(tokens)

# Predict the classes from the tokens
predicted = clf.predict(tokens)

# Print the f1 score using a sample by sample average
print(f1_score(labels, predicted, average='samples')) 
