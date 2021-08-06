import os
import pandas as pd
import numpy as np
import glob

import argparse as arg


def inGraphNodes(df):
    idx = 1
    pastGraph = None
    labels = []
    for _, row in df.iterrows():
        if pastGraph == None:
            pastGraph = row["Graph"]
        
        if (pastGraph != row["Graph"]):
            idx = 1
            pastGraph = row["Graph"]
        labels.append(idx)
        idx += 1
    
    df["InNodes"] = labels
    return df


def get_cliques(fpath):
    cliques = []
    with open(fpath, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split()[2:]
            line = [int(el) for el in line]
            cliques.append(line)
    
    return cliques

if __name__ == "__main__":
    parser = arg.ArgumentParser()

    parser.add_argument('--outdir', '-o', default="random_samples_annot")
    parser.add_argument('--attr', '-a', required=True)
    parser.add_argument('--node', '-n', default=128, type=int, required=True)
    args = parser.parse_args()

    n = args.node
    DIR = os.path.join("cliquer", str(n) + "_cliquer")

    attr = args.attr

    df = pd.read_csv(os.path.join(attr, str(n) + "_attr.txt"), sep=' ', header=None)
    df.rename(columns={0:"Node", 1:"Graph"}, inplace=True)

    df = inGraphNodes(df)

    clique_files = glob.glob(os.path.join(DIR, "*.out"))
    df["Clique"] = [np.nan for i in range(df.shape[0])]

    for f in clique_files:
        name = f.split('\\')[-1].split(".")[0]
        cliques = get_cliques(f)
        inCliqueNodes = set().union(*cliques)

        for node in inCliqueNodes:
            idx = df.loc[(df.Graph == name) & (df.InNodes == node), "Clique"] = 1

    df = df.replace(np.nan, 0)

    OUTDIR = args.outdir
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)
    
    df.to_csv(os.path.join(OUTDIR, str(n) + "_annot.txt"), sep=' ', index=False)
