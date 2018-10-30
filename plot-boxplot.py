import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def stats(data, funcname):
    print funcname
    print 'mean   =', np.mean(data)
    print 'median =', np.median(data)
    print 'stdev  =', np.std(data)
    print 'min    =', np.min(data)

task = sys.argv[1]
#fit_langs = {'ar', 'ca', 'de', 'el', 'en', 'eo', 'es', 'fr', 'gl', 'it', 'ja', 'nl', 'pl', 'pt', 'ru', 'sk', 'sr', 'sv', 'vi', 'zh'}
#data2 = np.squeeze(filter(lambda d:d[0] in fit_langs, data))[:,2].astype('float')

df = pd.read_csv('res/' + task + '-traditional.csv', header=None)
df = df.dropna()
data1 = df[1].values
stats(data1, 'traditional')

df = pd.read_csv('res/' + task + '-transductive-none.csv', header=None)
df = df.dropna()
df = df[df[0] != df[1]]
data = df.values
data2 = np.squeeze(filter(lambda d:d[0], data))[:,2].astype('float')
stats(data2, 'none')

df = pd.read_csv('res/' + task + '-transductive-quantile.csv', header=None)
df = df.dropna()
df = df[df[0] != df[1]]
data = df.values
data3 = np.squeeze(filter(lambda d:d[0], data))[:,2].astype('float')
stats(data3, 'quantile')

df = pd.read_csv('res/' + task + '-transductive-degree.csv', header=None)
df = df.dropna()
df = df[df[0] != df[1]]
data = df.values
data4 = np.squeeze(filter(lambda d:d[0], data))[:,2].astype('float')
stats(data4, 'degree')

df = pd.read_csv('res/' + task + '-transductive-pagerank.csv', header=None)
df = df.dropna()
df = df[df[0] != df[1]]
data = df.values
data5 = np.squeeze(filter(lambda d:d[0], data))[:,2].astype('float')
stats(data5, 'pagerank')

df = pd.read_csv('res/' + task + '-transductive-all.csv', header=None)
df = df.dropna()
df = df[df[0] != df[1]]
data = df.values
data6 = np.squeeze(filter(lambda d:d[0], data))[:,2].astype('float')
stats(data6, 'all')

# plotting
matplotlib.rcParams.update({'font.size': 18})

fig, ax = plt.subplots()
plt.boxplot([data1, data2, data3, data4, data5, data6], showfliers=False, showmeans=True)

plt.ylabel('ROC-AUC')
plt.xticks([1, 2, 3, 4, 5, 6], ['No-Trans.', 'None', 'Quant.', 'Degr.', 'P.R.', 'All'], rotation=0)
ax.margins(y=0.02)

plt.tight_layout()
plt.savefig('plots/boxplot-' + task + '.eps', format='eps')
