from topic_model_imports import *
from topic_model_preprocessing import preprocess



def get_topic_words(token_lists, labels, k=None):
    """
    get top words within each topic from clustering results
    """
    if k is None:
        k = len(np.unique(labels))
    topics = ['' for _ in range(k)]
    for i, c in enumerate(token_lists):
        topics[labels[i]] += (' ' + ' '.join(c))
    word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
    # get sorted word counts
    word_counts = list(map(lambda x: sorted(x, key=lambda x: x[1], reverse=True), word_counts))
    # get topics
    topics = list(map(lambda x: list(map(lambda x: x[0], x[:10])), word_counts))

    return topics


def get_coherence(model, token_lists, measure='c_v'):
    """
    Get model coherence from gensim.models.coherencemodel
    :param model: Topic_Model object
    :param token_lists: token lists of docs
    :param topics: topics as top words
    :param measure: coherence metrics
    :return: coherence score
    """
    if model.method == 'LDA':
        cm = CoherenceModel(model=model.ldamodel, texts=token_lists, corpus=model.corpus, dictionary=model.dictionary,
                            coherence=measure)
    else:
        topics = get_topic_words(token_lists, model.cluster_model.labels_)
        cm = CoherenceModel(topics=topics, texts=token_lists, corpus=model.corpus, dictionary=model.dictionary,
                            coherence=measure)
    return cm.get_coherence()


def get_silhouette(model):
    """
    Get silhouette score from model
    :param model: Topic_Model object
    :return: silhouette score
    """
    if model.method == 'LDA':
        return
    lbs = model.cluster_model.labels_
    vec = model.vec[model.method]

    # return score first, then input

    return silhouette_score(vec, lbs)



def plot_proj(embedding, lbs):
    """
    Plot reduced embeddings
    :param embedding: UMAP/TSNE/PCA embeddings
    :param lbs: labels
    """
    plt.figure(figsize=(15,10))
    n = len(embedding)
    counter = Counter(lbs)
    for i in range(len(np.unique(lbs))):
        plt.plot(embedding[:, 0][lbs == i], embedding[:, 1][lbs == i], '.', alpha=0.5,
                 label='cluster {}: {:.2f}%'.format(i, counter[i] / n * 100))
    plt.legend(loc = 'best')
    plt.grid(color ='grey', linestyle='-',linewidth = 0.25)

    plt.show()



def dimensionality_reduction(model, dim_method):

    # reduces full XLNET embeddings to 2 dimensions using UMAP/PCA/TSNE

  if dim_method=='UMAP':

      reducer = umap.UMAP()
      print('Calculating UMAP projection ...')
      vec = reducer.fit_transform(model.vec[model.method])
      print('Calculating UMAP projection done!')
    
  elif dim_method=='TSNE':
      print('Calculating TSNE projection ...')

      reducer = TSNE(n_components=2)
      X = model.vec[model.method]
      vec = reducer.fit_transform(X)
      print('Calculating TSNE projection done!')

  elif dim_method=='PCA':

      print('Calculating PCA projection ...')

      reducer = PCA(n_components=2)

      X = model.vec[model.method]

      vec = reducer.fit_transform(X)

      print('Calculating PCA projection done!')

  else:
    print('invalid dimensionality reduction method. ')
    return 

  return vec, model.cluster_model.labels_



def visualize(model, dim_reduction='UMAP'):
    """
    Visualize the result for the topic model by 2D embeddings
    :param model: Topic_Model object
    """

    if model.method == 'LDA':
        return

    vec, labels = dimensionality_reduction(model, dim_method=dim_reduction)

    plot_proj(vec, labels)





def get_wordcloud(model, token_lists, topic):
    """
    Get word cloud of each topic from fitted model
    :param model: Topic_Model object
    :param sentences: preprocessed sentences from docs
    """
    if model.method == 'LDA':
        return
    print('Getting wordcloud for topic {} ...'.format(topic))
    lbs = model.cluster_model.labels_
    tokens = ' '.join([' '.join(_) for _ in np.array(token_lists)[lbs == topic]])

    wordcloud = WordCloud(width=800, height=560,
                          background_color='white', collocations=False,
                          min_font_size=10).generate(tokens)

    # plot the WordCloud image
    plt.figure(figsize=(8, 5.6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=2)


def filter_keywords(df, keywords, limit=5):

    # filters keywords to remove 
    # params = full dataframe, cluster keywords, limit for keyword appearing in N clusters

  counts = Counter(word_tokenize(' '.join(df['keywords'].tolist())))

  # pick words only appearing in under some number of clsuters
  unique_words = [word for (word,count) in counts.items() if count<limit]
  
  keywords = word_tokenize(keywords)
  unique_keywords = [word for word in keywords if word in unique_words]
  
  return unique_keywords



def topic_results_table(model, token_lists):

    # produce a results table of cluster | cluster member percentage | keywords | filtered keywords 

  df = pd.DataFrame(columns=['cluster', 'members', 'keywords'])
  recurring_terms = ['company', 'time', 'say', 'energy', 'power', 'image', 'enron', 'year', 'img']  

  for i in range(model.k):
    lbs = model.cluster_model.labels_
    tokens = [_ for _ in np.array(token_lists)[lbs == i]]
    flat_tokens  = [item for sublist in tokens for item in sublist]

    # order by most common keywords in each list
    token_count = Counter(flat_tokens)
    keywords = [word for (word, count) in token_count.most_common(100) if word not in recurring_terms]

    # percentage of documents belonging to the cluster
    doc_members = len(tokens)/len(email_text) *  100

    # append row to dataframe
    df.loc[i] = [i, "{:.2f} %".format(doc_members), ' '.join(keywords) ]

    # limit for filtering keywords is if they appear in (Num_topics-2) clusters. 
  df['filtered_keywords'] = df.apply(lambda x: filter_keywords(df, x.keywords, limit=(model.k-2)), axis=1)
  
  return df




