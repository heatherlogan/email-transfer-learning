from topic_model_imports import *


import nltk
from nltk.corpus import stopwords
import re 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
STOP_WORDS = stopwords.words('english')
nltk.download('wordnet')

STOP_WORDS.append("i'm")
STOP_WORDS.append("re")
STOP_WORDS.append("fwd")
STOP_WORDS.append("i'm")
STOP_WORDS.append("i'm")
STOP_WORDS.append("am")
STOP_WORDS.append("pm")
STOP_WORDS.append("you're")
STOP_WORDS.append("youre")
STOP_WORDS.append("you")
STOP_WORDS.append("want")
STOP_WORDS.append("go")
STOP_WORDS.append("say")
STOP_WORDS.append("like")
nltk.download('punkt')
stemmer = PorterStemmer()


def process_sentences(sentence):

  # lowercase and regex to clean invalid characters from tokens

  s = sentence.lower()
  # letter repetition (if more than 2)
  s = re.sub(r'([a-z])\1{2,}', r'\1', s)
  # non-word repetition (if more than 1)
  s = re.sub(r'([\W+])\1{1,}', r'\1', s)
  # parenthesis
  s = re.sub(r'\(.*?\)', '. ', s)
  clean_sentence = s.strip()
  return clean_sentence


def lemmatize_stemming(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')# Tokenize and lemmatize


def process_tokens(text):

  # sentence wise pre-processing: tokenize, filter stopwords, filter short words, lemmatises

    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
          
    return result



def preprocess(emails, sample_size=None):
  """
  constructs sentences, token lists and indexes (if using a sample) of email list 
  by tokenizing sentence-wise, then token-wise. 
  params: 
    emails - email text as a list.
    sample_size = number of emails to pre_process
  """

  # if no sample size, use all data. 
  if not sample_size:
    sample_size=len(emails)

  print('Preprocessing emails ...')

  n_emails = len(emails)
  sentences = [] # preprocess sentence level
  token_lists = [] # preprocess at word level
  idxs = []  # indexes of sample size

  sample_emails = np.random.choice(n_emails, sample_size)

  for i, index in enumerate(sample_emails):
    
    sentence = process_sentences(emails[index])
    token_list = process_tokens(sentence)

    if token_list:
      idxs.append(index)
      sentences.append(sentence)
      token_lists.append(token_list)
  print('Preprocessing raw texts done.')
  return sentences, token_lists, idxs
