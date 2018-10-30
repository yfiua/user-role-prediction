import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

task = sys.argv[1]
func = sys.argv[2]
sort_by = sys.argv[3]
#fit_langs = {'ar', 'ca', 'de', 'el', 'en', 'eo', 'es', 'fr', 'gl', 'it', 'ja', 'nl', 'pl', 'pt', 'ru', 'sk', 'sr', 'sv', 'vi', 'zh'}

df = pd.read_csv('res/' + task + '-transductive-' + func + '.csv', header=None)
df = df.dropna()
data = df.values

rows = np.unique(data[:,0])
cols = np.unique(data[:,1])

# sort
df_meta = pd.read_csv('res/meta-info.csv', skipinitialspace=True)
df_meta = df_meta[[(lang.strip() in rows) for lang in df_meta['lang']]]

order = np.argsort(df_meta[sort_by])
rows = rows[order]
cols = cols[order]

# prepare data for heatmap
df = pd.DataFrame(index=rows, columns=cols)
df = df.fillna(0.0)

for d in data:
    df[d[0]][d[1]] = d[2]

data = df.values
fig, ax = plt.subplots(figsize=(16, 9))
heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

# want a more natural, table-like display
#ax.invert_yaxis()
#ax.xaxis.tick_top()

plt.xlabel('Source Dataset')
plt.ylabel('Target Dataset')

plt.xlim([0, data.shape[0]])
plt.ylim([0, data.shape[1]])

ax.set_xticklabels(rows, minor=False)
ax.set_yticklabels(cols, minor=False)

# colorbar
cbar = plt.colorbar(heatmap)
cbar.set_label('ROC-AUC', rotation=270, labelpad=20)

#plt.show()
plt.savefig('plots/heatmap-' + task + '-' + func + '-' + sort_by + '.eps', format='eps')
