from __future__ import division

import sys
import igraph
import powerlaw
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

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

# degree transformation
def degree_transform(degree, **kwargs):
    # pre-processing
    degree = np.array(degree)

    # fitting power-law distribution
    fit = powerlaw.Fit(degree, discrete=True, xmin=(1,6))

    alpha = fit.alpha
    x_min = fit.xmin

    n = len(degree)
    total = len(degree[degree >= x_min])
    c = (alpha - 1) * total / n

    T = {}
    for d in np.unique(degree):
        if (d <= x_min):
            T[d] = d
        else:
            T[d] = np.power(d/x_min, alpha-1) * x_min

    degree = np.array([ T[d] for d in degree ])
    return degree

def plot(degree_1, degree_2, output_file):
    fig, ax = plt.subplots()

    plot_degree(degree_1, 'x', 'Wiki-talk-de')
    plot_degree(degree_2, '.', 'Wiki-talk-fr')

    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.xlim(xmax=1e6)
    plt.ylim(ymin=1e-6)

    ax.legend(loc='upper right', numpoints=4, fontsize=20)

    plt.tight_layout()
    plt.savefig(output_file, format='eps')
    plt.clf()

def plot_degree(degree, marker='x', label='wiki-talk'):
    bins = np.unique(degree)
    hist, _ = np.histogram(degree, np.append(bins, max(bins)+1))
    hist = hist / len(degree)
    plt.loglog(bins, hist, marker, label=label)

def main():
    # setting
    matplotlib.rcParams.update({'font.size': 18})

    # read data
    G = read_graph('de-wiki-talk')
    degree_1 = G.degree()

    G = read_graph('fr-wiki-talk')
    degree_2 = G.degree()

    plot(degree_1, degree_2, 'plots/degree-original.eps')

    # transformation
    degree_1 = degree_transform(degree_1)
    degree_2 = degree_transform(degree_2)

    plot(degree_1, degree_2, 'plots/degree-transformed.eps')

if __name__ == '__main__':
    # init
    main()

