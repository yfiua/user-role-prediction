from __future__ import division

import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from transform_funcs import *

def read_graph(dataset):
    # dataset (network)
    df = pd.read_csv('data/' + dataset, sep='\t', header=None)

    nodes = np.unique(df[[0, 1]].values);
    max_node_num = max(nodes) + 1
    num_nodes = len(nodes)

    G = igraph.Graph(directed=True)
    G.add_vertices(max_node_num)
    G.add_edges(df[[0, 1]].values)

    G = G.subgraph(nodes)

    return G

def main():
    # setting
    matplotlib.rcParams.update({'font.size': 22})

    # read data
    G = read_graph('de-wiki-talk')
    degree_de = G.degree()

    G = read_graph('fr-wiki-talk')
    degree_fr = G.degree()

    bins = np.unique(degree_fr)
    hist, _ = np.histogram(degree_fr, np.append(bins, max(bins)+1))
    cumsum = np.cumsum(hist[::-1])[::-1] / len(degree_fr)     # reversed cumsum
    plt.step(bins, cumsum, where='post')

    bins = np.unique(degree_de)
    hist, _ = np.histogram(degree_de, np.append(bins, max(bins)+1))
    cumsum = np.cumsum(hist[::-1])[::-1] / len(degree_de)    # reversed cumsum
    plt.step(bins, cumsum, where='post')

    plt.xlabel('Degree')
    plt.ylabel('P')

    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('plots/degree-original.eps', format='eps')
    plt.clf()

    # transformation
    degree_de = degree_transform(degree_de)
    degree_fr = degree_transform(degree_fr)

    bins = np.unique(degree_fr)
    hist, _ = np.histogram(degree_fr, np.append(bins, max(bins)+1))
    cumsum = np.cumsum(hist[::-1])[::-1] / len(degree_fr)     # reversed cumsum
    plt.step(bins, cumsum, where='post')

    bins = np.unique(degree_de)
    hist, _ = np.histogram(degree_de, np.append(bins, max(bins)+1))
    cumsum = np.cumsum(hist[::-1])[::-1] / len(degree_de)    # reversed cumsum
    plt.step(bins, cumsum, where='post')

    plt.xlabel('Degree')
    plt.ylabel('P')

    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('plots/degree-transformed.eps', format='eps')
    plt.clf()

if __name__ == '__main__':
    # init
    main()

