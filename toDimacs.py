import networkx as nx
import pandas as pd
import numpy as np
import glob
import os
import sys
from tqdm import tqdm

import argparse as arg


def test_graphs(nlist):
    tk = tqdm(nlist, total=len(nlist))
    res = False
    for n in tk:
        name = n.split('/')[-1].split('.')[0]
        tk.set_postfix({'file': name})
        df = pd.read_csv(n, header=None, sep=' ')
        res = res or ((df[0].min() == 0) or (df[1].min() == 0))
    return not res


def prepend_line(file_name, line):
    """ Insert given string as a new line at the beginning of a file """
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Write given line to the dummy file
        write_obj.write(line + '\n')
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)


def clear_data(df):
    return pd.DataFrame(np.sort(df.values, axis=1), columns=df.columns).drop_duplicates()


def convert_dimacs(df, filename, outdir):
    out = ""
    df = clear_data(df)
    G = nx.from_pandas_edgelist(df, source=0, target=1)
    
    V_n, E_n = G.number_of_nodes(), G.number_of_edges()

    df["prefix"] = ["e" for i in range(df.shape[0])]

    filepath = os.path.join(outdir, filename) + ".txt"
    
    df.to_csv(filepath, sep=' ', header=None, index=False, columns=["prefix", 0, 1])
    top_line = "p edge %d %d" % (V_n, E_n)
    prepend_line(filepath, top_line)


def read_files(network_list, outdir):
    tk = tqdm(network_list, total=len(network_list))
    for n in tk:
        name = n.split('\\')[-1].split('.')[0]
        tk.set_postfix({'file': name})
        df = pd.read_csv(n, header=None, sep=' ')
        convert_dimacs(df, name, outdir)

if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument('--outdir', '-o')
    parser.add_argument('--dir', '-i', required=True)

    args = parser.parse_args()

    DIR = args.dir
    OUTDIR = DIR + "_DIMACS"
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)

    networks = glob.glob(os.path.join(DIR, "*"))

    print("Checking graphs...")
    if not test_graphs(networks):
        sys.exit()
    print("Converting...")
    read_files(networks, OUTDIR)
    print("Done!")
