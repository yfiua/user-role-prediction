from __future__ import division

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def main():
    # data
    data = {}
    data[0] = [ 0.9, 0.95, 0.92, 0.92, 0.738, 0.491 ]
    data[1] = [ 0.9, 0.90, 0.94, 0.89, 0.706, 0.564 ]
    data[2] = [ 1.0, 0.95, 0.90, 0.93, 0.690, 0.561 ]
    data[3] = [ 0.0, 0.05, 0.12, 0.29, 0.320, 0.340 ]

    roles = [ 'Administrator', 'Moderator', 'Subscriber', 'Banned' ]
    X = range(6)
    K = [ 10, 20, 50, 100, 500, 1000 ]

    # plot
    matplotlib.rcParams.update({'font.size': 20})
    markers = [ 'o', 'v', '^', 's' ]

    for i in range(len(roles)):
        fig, ax = plt.subplots()

        plt.plot(X, data[i], label=roles[i], marker=markers[i], markersize=10)

        # manipulate
        Y = ax.get_yticks()
        plt.xticks(X, K)
        plt.yticks(Y, ['{:3.0f}%'.format(y * 100) for y in Y])

        plt.xlabel(r'Number of identified users $k$')
        plt.ylabel(r'Percentage of trusted users in $k$')

        loc_legend = ( 4 if i==3 else 3 )
        ax.legend(fontsize=20, loc=loc_legend)

        plt.tight_layout()
        plt.savefig('plots/application-' + roles[i].lower() + '.eps', format='eps')
        plt.clf()

    # plot positive roles
    fig, ax = plt.subplots()
    for i in range(len(roles)-1):
        plt.plot(X, data[i], label=roles[i], marker=markers[i], markersize=10)

    # manipulate
    Y = ax.get_yticks()
    plt.xticks(X, K)
    plt.yticks(Y, ['{:3.0f}%'.format(y * 100) for y in Y])

    plt.xlabel(r'Number of identified users $k$')
    plt.ylabel(r'Percentage of trusted users in $k$')

    ax.legend(fontsize=20, loc=3)

    plt.tight_layout()
    plt.savefig('plots/application-positive.eps', format='eps')
    plt.clf()

if __name__ == '__main__':
    # init
    main()

