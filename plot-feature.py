from __future__ import division

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def main():
    # params
    datasets = [ 'Wiki-talk-de', 'Wiki-talk-fr', 'Boards.ie', 'Software-AG' ]
    features = [ 'degree', 'clustering_coefficient', 'pagerank' ] #, 'eccentricity',  'strength' , 'diversity' ]  'personalized_pagerank',
    R = 10

    res = np.load('res/eval-feature-aggr.npy')

    # plot
    matplotlib.rcParams.update({'font.size': 22})
    markers = [ 'o', 'v', '^', 's' ]

    for f in range(len(features)):
        fig, ax = plt.subplots()
        for i in range(len(datasets)):
            line = res[i,f,:]
            plt.plot(range(1, R+1), line, label=datasets[i], marker=markers[i], markersize=10)

        ax.legend(fontsize=22, loc=4)
        ax.margins(y=0.02)

        plt.xlabel(r'Number of rounds $r$')
        plt.ylabel(r'Maximum absolute value of $\rho$')

        plt.tight_layout()
        plt.savefig('plots/feature-' + features[f] + '.eps', format='eps')
        plt.clf()

if __name__ == '__main__':
    # init
    main()

