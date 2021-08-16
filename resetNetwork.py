import pandas as pd
import os
import glob
from tqdm import tqdm

import argparse as arg


def reset(path, start=1):
    df = pd.read_csv(path, sep=" ", header=None)
    df.rename(columns={0:"source", 1:"target"}, inplace=True)
    u_s = set(df.source.unique())
    u_t = set(df.target.unique())
    nodes = list(u_s.union(u_t))
    node_map = dict()   

    id = start
    for node in nodes:
        node_map[node] = id
        id += 1
    
    df['source'] = df['source'].map(node_map)
    df['target'] = df['target'].map(node_map)

    return df

if __name__ == "__main__":
    parser = arg.ArgumentParser()

    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o')
    args = parser.parse_args()

    input_folder = args.input
    if args.output is not None:
        output_folder = args.output
    else:
        output_folder = input_folder + "_reset"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    graphs = glob.glob(os.path.join(input_folder, '*.edges'))

    tk = tqdm(graphs, total=len(graphs))
    for gpath in tk:
        name = gpath.split('\\')[-1]
        tk.set_postfix({'file': name})
        network_df = reset(gpath)
        network_df.to_csv(os.path.join(output_folder, name), sep=' ', header=False, index=False)