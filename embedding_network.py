"""
NOT COMPLETE - NOT WORKING
TODO Clean the code and reconfigure for easy use.
"""
import networkx as nx
import numpy as np
import pandas as pd
import os
import random

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UnsupervisedSampler

import matplotlib.pyplot as plt

import tensorflow.keras as keras # DO NOT USE KERAS DIRECTLY
from sklearn import preprocessing, feature_extraction, model_selection

from tensorflow.keras import layers, optimizers, losses, metrics, Model

from stellargraph import globalvar

import matplotlib.pyplot as plt


def random_features(edgelist, d):
    X = pd.DataFrame()
    G = nx.from_pandas_edgelist(edgelist)
    nodelist = sorted(list(G.nodes()))

    for feat in range(d):
        colname = "f" + str(feat)
        X[colname] = [np.random.rand() for i in nodelist]
    
    X.index = nodelist

    return X


def getTrainReady(gname, aname, feature_d=32, insample=False, ratio=None, 
                  n_walks=1, n_length=5, batch_size=50, num_samples=[10, 5]):
    gpath = os.path.join(MAIN_DIR, gname)
    network_df = pd.read_csv(gpath, sep=" ", header=None)
    network_df.rename(columns={0:"source", 1:"target"}, inplace=True)

    apath = os.path.join(MAIN_DIR, "bio-Annot.txt")
    annot = pd.read_csv(apath, sep=" ")
    
    if insample and (ratio is not None):
        ratio = 1
        pos = annot.Clique[annot.Clique == 1]
        neg = annot.Clique[annot.Clique == 0].sample(int(pos.shape[0]*ratio))
        targets = pd.concat([pos, neg])
    else:
        targets = annot.Clique
    
    X = random_features(network_df, feature_d)
    G = sg.StellarGraph(X, network_df)

    nodes = list(G.nodes())
    number_of_walks = n_walks
    length = n_length

    unsupervised_samples = UnsupervisedSampler(
        G, nodes=nodes, length=length, number_of_walks=number_of_walks
    )

    generator = GraphSAGELinkGenerator(G, batch_size, num_samples)
    train_gen = generator.flow(unsupervised_samples)

    return network_df, G, annot, targets, train_gen, generator


#MAIN_DIR = "/content/drive/MyDrive/CMP615 - Project"
graph_name = "bio-All.txt"
annot_name = "bio-Annot.txt"
RANDOM_SEED = 9

feature_d = 32

n_walks = 1
n_length = 5

batch_size = 50
num_samples = [10, 5]