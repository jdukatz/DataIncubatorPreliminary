import json
import sys

import pandas as pd
import numpy as np

import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabaz_score, silhouette_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

from gensim.models import word2vec

def preprocess_archive(file, vec_method):
    print('building vectors...')

    chyrons = pd.read_csv(file, sep='\t')

    if vec_method == 'tfidf':
        tfidf_vec = TfidfVectorizer() # max_df frequency?
        vectors = tfidf_vec.fit_transform(chyrons['text'])
    elif vec_method == 'embedding':
        sents = word2vec.Text8Corpus('text8')
        embedding_model = word2vec.Word2Vec(sents, size=50) # 50 dim embeddings
        vectors = np.empty((len(chyrons), 50))
        nlp = spacy.load('en')
        for i in range(len(chyrons)):
            chyron_vec = np.zeros(50)
            chyron_text = chyrons.iloc[i]['text']
            tokenized_chyron = nlp(chyron_text)
            for t in tokenized_chyron:
                try:
                    word_vec = embedding_model.wv.word_vec(t.lower_)
                except(KeyError):
                    word_vec = np.random.rand(50) # add noise for OOV
                chyron_vec += word_vec
            chyron_vec = chyron_vec / len(tokenized_chyron)
            vectors[i] = chyron_vec
    else:
        raise Exception('vector generation method not recognized')

    return vectors, chyrons

def kmeans_cluster(vectors):
    print('k-means clustering...')
    clusters = KMeans(n_clusters=4).fit(vectors)
    return clusters.labels_

def spectral_cluster(vectors):
    print('spectral clustering...')
    return SpectralClustering(n_clusters=4).fit_predict(vectors)

def agglom_cluster(vectors):
    print('agglomerative clustering...')
    return AgglomerativeClustering(n_clusters=4).fit_predict(vectors)

def project(vectors, method):

    if method == 'tSNE':
        print('building projections with tSNE...')
        reduction = TSNE(n_components=2)

    elif method == 'PCA':
        print('building projections with PCA...')
        reduction = PCA(n_components=2)

    elif method == 'MDS':
        print('building projections with MDS...')
        reduction = MDS(n_components=2)
        
    else:
        raise Exception('Projection method not recognized')

    return reduction.fit_transform(vectors)

def run_evaluations(chyrons, vectors, clusters):
    ch_score = calinski_harabaz_score(vectors, clusters)
    silhouette = silhouette_score(vectors, clusters)
    le = LabelEncoder()
    true_labels = le.fit_transform(chyrons['channel'])
    rand_index = adjusted_rand_score(true_labels, clusters)
    return {'ch_score': ch_score, 'silhouette': silhouette, 'rand_index': rand_index}

def evals_out(metrics, outfile):
    json_string = json.dumps(metrics)
    with open('processed_data/' + outfile, 'w') as f:
        f.write('var ' + outfile[:-3] + ' = ' + json_string)

def data_out(chyrons, projection, clusters, cluster_method, projection_method, vector_method, out_f):
    """
    Take the data from one set of chyrons and spit it out in a json
    """
    rows = []
    for i in range(projection.shape[0]):
        chyron = chyrons.iloc[i]
        row = {'channel': chyron['channel'], # string
               'time_stamp': chyron['date_time_(UTC)'], # string
               'raw_text': chyron['text'].lower(), # string
               'projected_vector': projection[i].tolist(), # 2-d vector, list form
               'cluster_id': int(clusters[i]), # int
               'cluster_method': cluster_method, # string
               'projection_method': projection_method, # string
               'vector_method': vector_method # string
               }
        rows.append(row)
    data_string = json.dumps(rows, ensure_ascii=False)
    with open('processed_data/' + out_f, 'w') as f:
        f.write('var ' + out_f[:-3] + ' = ' + data_string)

def pipeline(data_in, dataset_name):
    """
    Pipeline for processing one chyron data set
    """
    vectors, chyrons = preprocess_archive(data_in, 'embedding')
    km = kmeans_cluster(vectors)
    spec = spectral_cluster(vectors)
    agg = agglom_cluster(vectors)

    clust_methods = {'km': km, 'spec': spec, 'agg': agg}
    for name, clust in clust_methods.items():
        for proj in ['tSNE', 'PCA', 'MDS']:
            filestring = dataset_name + '_' + proj + '_' + name + '.js'
            projection = project(vectors, proj)
            print('writing data to ' + filestring + '...')
            data_out(chyrons, projection, clust, name, proj, 'embedding', filestring)
        evals_out(run_evaluations(chyrons, vectors, clust), dataset_name + '_' + name + '_eval.js')


if __name__ == '__main__':
    pipeline(sys.argv[1], sys.argv[2])
