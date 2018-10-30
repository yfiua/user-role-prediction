import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_degree_loglog(degrees):
    bins = np.unique(degrees)
    hist, _ = np.histogram(degrees, np.append(bins, max(bins)+1))

    # plot
    plt.loglog(bins, hist, '.')
    plt.xlabel('Degree(d)')
    plt.ylabel('Frequency')

    plt.savefig('degree-distribution.eps', format='eps', dpi=1000)

def plot_cum_degree_loglog(degrees):
    bins = np.unique(degrees)
    hist, _ = np.histogram(degrees, np.append(bins, max(bins)+1))

    hist = np.cumsum(hist[::-1])[::-1] / len(degrees)
    # plot
    plt.step(bins, hist)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree(d)')
    plt.ylabel('P(X>d)')

    plt.savefig('cumsum-degree-distribution.eps', format='eps', dpi=1000)
