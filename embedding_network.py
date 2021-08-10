"""
NOT COMPLETE - NOT WORKING
TODO Clean the code and reconfigure for easy use.
"""
import networkx as nx
import numpy as np
import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UnsupervisedSampler

import tensorflow.keras as keras # DO NOT USE KERAS DIRECTLY

import argparse as arg


def random_features(edgelist, d):
    X = pd.DataFrame()
    G = nx.from_pandas_edgelist(edgelist)
    nodelist = sorted(list(G.nodes()))

    for feat in range(d):
        colname = "f" + str(feat)
        X[colname] = [np.random.rand() for i in nodelist]
    
    X.index = nodelist

    return X


def preprocess(gpath, apath, feature_d=32, insample=False, ratio=None, 
                  n_walks=1, n_length=5, batch_size=50, num_samples=[10, 5]):

    network_df = pd.read_csv(gpath, sep=" ", header=None)
    network_df.rename(columns={0:"source", 1:"target"}, inplace=True)

    annot = pd.read_csv(apath, sep=" ")
    
    if insample and (ratio is not None):
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


def create_model(pretrained=None, tensorboard=False):
    layer_sizes = [50, 50]
    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2"
    )
    # Build the model and expose input and output sockets of graphsage, for node pair inputs:
    x_inp, x_out = graphsage.in_out_tensors()
    
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)

    model = keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.Precision()],
    )

    if pretrained is not None:
        print("Using pretrained model weights at ", pretrained)
        restore_path = pretrained
        model.load_weights(restore_path)
        return x_inp, x_out, model
    else:
        return x_inp, x_out, model


def train_GraphSage(train_data, epochs, tensorboard=False):
    # Checkpoint Directory
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "model/cp-{epoch:02d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if tensorboard:
        # Setup Callback
        cp_callback = [
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, 
                verbose=1, 
                save_weights_only=True
            ),
            keras.callbacks.TensorBoard(
                log_dir='tensorboard_logs',
                histogram_freq=1
            )
            ]
    else:
        cp_callback = [
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, 
                verbose=1, 
                save_weights_only=True
            )
            ]

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    #TRAIN
    history = model.fit(
        train_data,
        epochs=epochs,
        verbose=1,
        use_multiprocessing=False,
        workers=4,
        shuffle=True,
        callbacks=cp_callback
    )

    return model, history


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument('--graphpath', '-g', required=True)
    parser.add_argument('--annotpath', '-a', required=True)
    parser.add_argument('--epochs', '-e', default=5, type=int)
    parser.add_argument('--dfeature', '-f', default=32, type=int)
    parser.add_argument('--batch', '-b', default=50, type=int)
    parser.add_argument('--pretrained', '-p', default=None)
    parser.add_argument('--tensorboard', '-t', default=False, type=bool)
    parser.add_argument('--save', '-s', default=False, type=bool)

    args = parser.parse_args()
    graph_path = args.graphpath
    annot_path = args.annotpath
    epochs = args.epochs
    feature_d = args.dfeature
    batch_size = args.batch
    pretrained_path = args.pretrained
    tb = args.tensorboard
    save = args.save

    n_walks = 1
    n_length = 5
    num_samples = [10, 5]

    #PRETRAINING
    network_df, G, annot, targets, train_gen, generator = preprocess(
            graph_path, annot_path, feature_d=feature_d, n_walks=n_walks, 
            n_length=n_length, batch_size=batch_size, num_samples=num_samples,
            insample=False, ratio=3
        )
    
    # Restore or train GraphSage
    x_inp, x_out, model = create_model(pretrained=pretrained_path)
    if pretrained_path is None:
        model, history = train_GraphSage(train_gen, epochs, tensorboard=tb)
    
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    node_ids = targets.index
    node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(node_ids)
    node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)
    if save:
        print(node_embeddings.shape)
        df = pd.DataFrame(node_embeddings, columns=[i for i in range(node_embeddings.shape[1])])
        df["Clique"] = targets.values
        df.to_csv("embeddings_v2.csv", index=None)
    else:
        print(node_embeddings.shape)