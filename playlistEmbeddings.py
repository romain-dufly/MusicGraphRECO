
##--## Artist embeddings from graph representation ##--##

import numpy as np
import pandas as pd
import networkx as nx

import node2vec as n2v
import pecanpy as pc
import gensim.models.word2vec as w2v

## Load data

nodes = pd.read_csv('/users/eleves-a/2021/romain.dufly/MODAL/allArtists60+.csv', sep=';')
edges = pd.read_csv('/users/eleves-a/2021/romain.dufly/MODAL/playlistLinks.csv', sep=';')

print(nodes.head())
print(edges.head())

def edg_from_csv(df):
    '''Create edgelist .edg file, in the format node1_id node2_id weight'''
    with open('/users/eleves-a/2021/romain.dufly/MODAL/edgelist.edg', 'w') as f:
        for _, row in df.iterrows():
            f.write(str(row['artist_uri_x']) + ' ' + str(row['artist_uri_y']) + ' ' + str(row['count']) + '\n')
#edg_from_csv(edges)


## Node2Vec using PecanPy

def node2vec():
    g = pc.pecanpy.DenseOTF(p=1, q=1, workers=8, verbose=True)

    g.read_edg('/users/eleves-a/2021/romain.dufly/MODAL/edgelist.edg', weighted=True, directed=False, delimiter=' ')
    walks = g.simulate_walks(num_walks=32, walk_length=32)
    w2v_model = w2v.Word2Vec(walks, vector_size=16,min_count=1, window=10, workers=8)

    w2v_model.save('/users/eleves-a/2021/romain.dufly/MODAL/node2vec.model')
#node2vec()


## Load model and get embeddings
w2v_model = w2v.Word2Vec.load('/users/eleves-a/2021/romain.dufly/MODAL/node2vec.model')

def get_embeddings():
    def get_embedding(x):
        try :
            return w2v_model.wv[x]
        except :
            return np.zeros(64) # if the artist is not in the graph, return 0 vector

    nodes['embedding'] = nodes['id'].apply(get_embedding)
    nodes.to_csv('/users/eleves-a/2021/romain.dufly/MODAL/embeddings.csv', sep=';', index=False)

#get_embeddings()