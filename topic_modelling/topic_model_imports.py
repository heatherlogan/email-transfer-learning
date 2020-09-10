import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle 
import re
import matplotlib.pyplot as plt
from datetime import datetime
import time
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


import gensim
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *


from collections import Counter
from wordcloud import WordCloud
import umap
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import transformers
from transformers import XLNetTokenizer, XLNetModel
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models


import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

np.random.seed(400)

