import numpy as np
import pandas as pd
from itertools import combinations
import os
import glob
import networkx as nx

from tqdm import tqdm

import argparse as arg


def setVertexIds(df):
    df["source"] = df["source"].apply(lambda x: x+1)
    df["target"] = df["target"].apply(lambda x: x+1)   
    return df


def createClique(n, k):
    clique = sorted(np.random.choice(range(0, n), k, replace=False))
    edges = combinations(clique, 2)
    # print(clique)
    return set(edges)


def readBatch(arr):
    tk = tqdm(arr, total=len(arr))
    for path in tk:
        df = pd.read_csv(path, header=None, sep=' ')
        df.rename(columns={0:"source", 1:"target"}, inplace=True)
        df = setVertexIds(df)

        G = nx.from_pandas_edgelist(df)

        G.remove_edges_from(list(nx.selfloop_edges(G)))

        if len(list(nx.selfloop_edges(G))) > 0:
            print("Error!")
            return
        
        req_edges = createClique(n, cq)
        added = (req_edges.difference(set(G.edges())))
        G.add_edges_from(added)

        df = nx.to_pandas_edgelist(G)
        df.to_csv(path, index=False, header=False, sep=' ')


def generateGraphs(params):
    graphname = params['graph']
    n = int(params['n'])
    numit = int(params['numGen'])
    graphType = params['type']

    if graphType == 'GNP':
        p = params['d']
        np.random.seed(4639)
        # generate all randomness at once
        pairs = np.array([t for t in combinations(np.arange(n), 2)])
        ps = np.random.rand(pairs.shape[0], numit) <= p
        for it in np.arange(numit):
            # keep the edges that are sampled
            pairsKeep = pairs[ps[:, it] == 1]
            outname = str(n) + '_' + graphname + '_' + graphType + '_' + str(it) + '.txt'
            outname = os.path.join(RANDOM_DIR, outname)
            np.savetxt(outname, pairsKeep, fmt=('%d', '%d'), delimiter=' ', comments='')


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument('--outdir', '-o', required=True)
    parser.add_argument('--node', '-n', default=128, type=int)
    parser.add_argument('--prob', '-p', default=0.5, type=float)
    parser.add_argument('--number', '-k', default=100, type=int)
    parser.add_argument('--clique', '-c', default=12, type=int)
    parser.add_argument('--type', '-t', default='GNP')

    args = parser.parse_args()
    n = args.node
    p = args.prob
    k = args.number
    cq = args.clique
    t = args.type

    RANDOM_DIR = args.outdir
    RANDOM_DIR = RANDOM_DIR + "_" + str(n)
    if not os.path.exists(RANDOM_DIR):
        os.mkdir(RANDOM_DIR)

    print("Progress started for %s with parameters:\n\tNode: %d\n\tProbability: %f\n\tIteration: %d" % (t, n, p, k))
    params = {'graph': 'output', 'type': t, 'n': n, 'd': p, 'numGen': k}
    generateGraphs(params)

    files = glob.glob(os.path.join(RANDOM_DIR, "*.txt"))
    readBatch(files)