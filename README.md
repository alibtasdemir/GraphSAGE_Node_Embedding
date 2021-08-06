# GraphSAGE Node Embedding

Code implementation of [Learning Heuristics for the Maximum Clique Enumeration Problem Using Node Embeddings](https://web.cs.hacettepe.edu.tr/~ozkahya/pub/Paper1045_SIU_2021.pdf).

## Requirements

We use [networkx](https://networkx.org/) for graph processing and [TQDM](https://tqdm.github.io/) for better outputs.

```bash
pip install networkx
pip install tqdm
```

## Usage

### Random Graph Generation
To generate random graphs use **generate_random.py**:
```bash
python generate_random.py -o OUTPUT_DIRECTORY -n NODES -p PROB -k SAMPLES -c CLIQUE  
```
There are 5 parameters for the random graph generation.

*OUTPUT DIRECTORY* is the directory that the script saves the generated random graphs (Required).

The number of *NODES* (default 128), and the number of *SAMPLES* (default 100) will be generated given (Optional).

*PROB* is the probability value (between 0 and 1) that represents the probability of making an edge between two nodes in the random graph (Optional) (default 0.5). 

*CLIQUE* is the size of the maximum clique that will be planted in the random graph (Optional) (default 12).

Sample run;
```bash
python generate_random.py -o random_sample -n 256 -k 300 -c 13  
```
This run will create 300 samples of random graphs that have 256 nodes in the directory "random_sample_256". Each graph is guaranteed to have a maximum clique with the size 13.

### Testing Graphs' Format

To test the generated graphs for the format, there is a script named "test_graphs.py". It basically checks if the generated graphs' nodes ids are starting from 0 or 1. (The desired format is starting from 1).

### Edge List to DIMACS

Another utility script is the conversion of edge list networks to DIMACS format. We use cliquer to tag nodes if they are in a maximum clique or not. Cliquer needs graphs in DIMACS format. So the file "toDimacs.py" converts the edge list files to DIMACS format.


## License
[MIT](https://choosealicense.com/licenses/mit/)