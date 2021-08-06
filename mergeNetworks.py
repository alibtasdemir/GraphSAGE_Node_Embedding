import igraph as ig
import pandas as pd
import os
import glob
from tqdm import tqdm

import argparse as arg


def read_graphs_ig(network_list):
    tk = tqdm(network_list, total=len(network_list))
    G_list = []

    for n in tk:
        name = n.split('\\')[-1].split('.')[0]
        tk.set_postfix({'file': name})
        df = pd.read_csv(n, header=None, sep=' ')
        tuples = [tuple(x) for x in df.values]
        G = ig.Graph.TupleList(tuples, directed = False)
        G.vs["gname"] = [name for i in range(G.vcount())]
        G.simplify()
        G_list.append(G)

    return G_list


def main(n, OUTDIR):
    DIR = "random_samples_" + str(n)

    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)
    
    networks = glob.glob(os.path.join(DIR, '*'))

    # Working but not complete.
    G_list = read_graphs_ig(networks)
    G_all = ig.disjoint_union(G_list)
    del(G_list)

    print(ig.summary(G_all))

    G_all.write_edgelist(os.path.join(OUTDIR, str(n) + ".txt"))
    A = [(name, attr) for name, attr in enumerate(G_all.vs["gname"])]

    with open(os.path.join(OUTDIR, str(n) + "_attr.txt"), "w") as f:
        out = ""
        for line in A:
            out += str(line[0]) + " " + line[1] + "\n"
        out = out[:-1]
        f.write(out)

    return 1


if __name__ == "__main__":
    parser = arg.ArgumentParser()

    parser.add_argument('--outdir', '-o', default="random_samples_merged")
    parser.add_argument('--node', '-n', default=128, type=int, required=True)
    args = parser.parse_args()

    n = args.node
    OUTDIR = args.outdir

    main(n, OUTDIR)