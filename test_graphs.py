from tqdm import tqdm
import pandas as pd
import glob
import os
import argparse as arg

def test_graphs(nlist):
    errors=[]
    tk = tqdm(nlist, total=len(nlist))
    res = False
    for n in tk:
        name = n.split('/')[-1].split('.')[0]
        tk.set_postfix({'file': name})
        df = pd.read_csv(n, header=None, sep=' ')
        res = res or ((df[0].min() == 0) or (df[1].min() == 0))
        if res:
            errors.append(name.split("\\")[-1])
    return errors


def test_graphs_reverse(nlist):
    errors=[]
    tk = tqdm(nlist, total=len(nlist))
    res = True
    for n in tk:
        name = n.split('/')[-1].split('.')[0]
        tk.set_postfix({'file': name})
        df = pd.read_csv(n, header=None, sep=' ')
        res = res and ((df[0].min() == 1) or (df[1].min() == 1))
        if not res:
            errors.append(name.split("\\")[-1])
    return errors


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument('--dir', '-d', required=True)
    parser.add_argument('--reverse', '-r', default=False, type=bool)

    args = parser.parse_args()

    DIR = args.dir
    rev = args.reverse

    networks = glob.glob(os.path.join(DIR, "*"))

    print("Checking graphs...")
    if rev:
        errs = test_graphs_reverse(networks)
    else:
        errs = test_graphs(networks)
    
    print(errs)