from __future__ import division

import sys
import igraph
import numpy as np
import pandas as pd

# main
def main(argv):
    # languages
    langs = [ 'ar', 'bn', 'br', 'ca', 'cy', 'de', 'el', 'en', 'eo', 'es', 'eu', 'fr', 'gl', 'ht', 'it', 'ja', 'lv', 'nds', 'nl', 'oc', 'pl', 'pt', 'ru', 'sk', 'sr', 'sv', 'vi', 'zh' ]

    # print title
    print 'lang, # normal users, # bots, # admins,% normal users, % bots, % admins, n, m, global_clusco, avg_degree, assortativity'

    for lang in langs:
        # read network
        df = pd.read_csv('data/' + lang + '-wiki-talk', sep='\t', header=None)

        nodes = np.unique(df[[0, 1]].values);
        max_node_num = max(nodes) + 1
        num_nodes = len(nodes)

        G = igraph.Graph(directed=True)
        G.add_vertices(max_node_num)
        G.add_edges(df[[0, 1]].values)

        G = G.subgraph(nodes)

        # read roles
        df_role = pd.read_csv('data/' + lang + '-user-group', sep='\t', header=None)
        roles = df_role[[0,1]].values

        Y = [0] * max_node_num
        for r in roles:
            Y[r[0]] = r[1]

        Y = np.array([Y[i] for i in nodes])
        global_clusco = G.transitivity_undirected()
        avg_degree = np.mean(G.degree())
        assortativity = G.assortativity_degree()

        print lang, ',', len(Y[Y == 0]), ',', len(Y[Y == 1]), ',', len(Y[Y == 2]), ',', len(Y[Y == 0]) / len(Y), ',', len(Y[Y == 1]) / len(Y), ',', len(Y[Y == 2]) / len(Y), ',', G.vcount(), ',', len(G.es), ',', global_clusco, ',', avg_degree, ',', assortativity

if __name__ == '__main__':
    main(sys.argv[1:])

