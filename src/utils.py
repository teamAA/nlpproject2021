import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def tokenization(text):
    # Word Tokenization
    text = re.split('\W+', text)
    return text

def remove_stopwords(text):
    stopword = nltk.corpus.stopwords.words('english')
    text = [word for word in text if word not in stopword]
    return text

def lemmatizing(text):
    # Word Lemmatization 
    lm = nltk.WordNetLemmatizer()
    text = [lm.lemmatize(word, nltk.corpus.wordnet.VERB) for word in text]
    return text

def stemming(text):
    # Word Stemming 
    ps = nltk.PorterStemmer()
    text = [ps.stem(word) for word in text]
    return text

# Remove Whitespace
def empty_token(text):
    text = [x for x in text if len(x)>0]
    return text

def argmax_2(lst):
  lst = lst.tolist()
  return (lst.index(max(lst)))-1