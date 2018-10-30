from __future__ import division

import sys
import igraph
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from transform_funcs import *
from scipy.sparse import coo_matrix

def graph_to_sparse_matrix(G):
    n = G.vcount()
    xs, ys = map(np.array, zip(*G.get_edgelist()))
    if not G.is_directed():
        xs, ys = np.hstack((xs, ys)).T, np.hstack((ys, xs)).T
    else:
        xs, ys = xs.T, ys.T
    return coo_matrix((np.ones(xs.shape), (xs, ys)), shape=(n, n), dtype=np.int16)

def get_feature(G, f):
    return _transform_func_degree(getattr(G, f)()) if callable(getattr(G, f)) else _transform_func(getattr(G, f))

# aggregate by the mean value of feature of neighbours
def mean_neighbour(A, d, feature):
    return A.dot(feature) / d

def get_feature_matrix(G, features, rounds=5):
    # local clustering coefficient
    G_sim = G.as_directed().simplify(multiple=False)                 # remove loops

    lcc = np.array(G_sim.transitivity_local_undirected(mode='zero'))
    lcc[lcc < 0] = 0                                                 # implementation of igraph is really shitty
    if (sys.argv[1] == 'lcc'):                                       # normalized lcc
        lcc = lcc_transform(lcc, G_sim.degree())
    G.clustering_coefficient = lcc

    # compute PageRank
    G_sim = G.copy()
    G_sim = G_sim.simplify(multiple=False)                           # remove loops

    alpha = 0.15
    pagerank = np.array(G_sim.pagerank(damping=1-alpha))
    if (sys.argv[1] in [ 'pagerank', 'all' ]):                       # normalized PageRank
        n = G_sim.vcount()
        pr_ub = np.percentile(pagerank, 100-300/n, interpolation='lower')
        pagerank[pagerank > pr_ub] = pr_ub
        min_max_scaler = MinMaxScaler()
        pagerank = min_max_scaler.fit_transform(pagerank)
        #dangling_nodes = (np.array(G_sim.outdegree()) == 0)
        #r_low = (alpha + (1-alpha) * sum(pagerank[dangling_nodes])) / G_sim.vcount()
        #pagerank = pagerank / r_low
    G.pr = pagerank

    feature_matrix = [ get_feature(G, f) for f in features ]
    X = np.array(feature_matrix).T

    # adjacency matrix (simplified)
    A = graph_to_sparse_matrix(G.as_undirected().simplify())
    d = np.squeeze(np.array(A.sum(axis=1))).astype(np.int)
    d[d == 0] = 1

    for i in range(rounds):
        feature_matrix = [ mean_neighbour(A, d, f) for f in feature_matrix ]
        X = np.concatenate((X, np.array(feature_matrix).T), axis=1)

    #X = np.hstack((X, np.array([pagerank]).T))
    return X

def read_data(lang, features):
    # dataset (network)
    df = pd.read_csv('data/' + lang + '-wiki-talk', sep='\t', header=None)

    nodes = np.unique(df[[0, 1]].values);
    max_node_num = max(nodes) + 1
    num_nodes = len(nodes)

    G = igraph.Graph(directed=True)
    G.add_vertices(max_node_num)
    G.add_edges(df[[0, 1]].values)

    G = G.subgraph(nodes)

    # features
    X = get_feature_matrix(G, features)

    # dataset (roles)
    df_role = pd.read_csv('data/' + lang + '-user-group', sep='\t', header=None)
    roles = df_role[[0,1]].values

    y = [0] * max_node_num
    for r in roles:
        y[r[0]] = r[1]

    y = np.array([y[i] for i in nodes])

    return np.squeeze(X), y

# main
def main():
    # params
    n_trees = 64
    features = [ 'clustering_coefficient' , 'degree' , 'indegree' , 'outdegree', 'pr' ] #, 'eccentricity',  'strength' , 'diversity' ]   #'pagerank', 'personalized_pagerank',
    langs = [ 'ar', 'bn', 'br', 'ca', 'cy', 'de', 'el' , 'en', 'eo', 'es', 'eu', 'fr', 'gl', 'ht', 'it', 'ja', 'lv', 'nds', 'nl', 'oc', 'pl', 'pt', 'ru', 'sk', 'sr', 'sv', 'vi', 'zh' ]
#   langs = [ 'br', 'cy', 'ar', 'lv', 'zh' ]

    # preparation
    X = {}
    y = {}
    for lang in langs:
        print 'reading dataset', lang
        X[lang], y[lang] = read_data(lang, features)

    # classification

    ## bot classifier
    f_transductive = open('res/bot' + _suffix, 'w')
    if (_perform_traditional):
        f_traditional = open('res/bot-traditional.csv', 'w')

    for lang_source in langs:
        y_source = (y[lang_source] == 1)

        if (_perform_traditional):
            ## traditional learning
            X_fit, X_eval, y_fit, y_eval= train_test_split(X[lang_source], y_source, test_size=0.368, random_state=42)

            if (len(np.unique(y_fit)) == 1) or (len(np.unique(y_eval)) == 1):
                # ROC not defined
                auc = np.nan
            else:
                # classifier
                clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
                clf.fit(X_fit, y_fit)

                y_predict = clf.predict_proba(X_eval)[:,1]
                auc = roc_auc_score(y_eval, y_predict)

            f_traditional.write(lang_source + ',' + str(auc) + '\n')

        ## classifier
        clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        clf.fit(X[lang_source], y_source)

        ## evaluation
        for lang_target in langs:
            y_target = (y[lang_target] == 1)
            if (len(np.unique(y_source)) == 1) or (len(np.unique(y_target)) == 1):   # ROC not defined
                auc = np.nan
            else:
                y_predict = clf.predict_proba(X[lang_target])[:,1]
                auc = roc_auc_score(y_target, y_predict)

            f_transductive.write(lang_source + ',' + lang_target + ',' + str(auc) + '\n')

    f_transductive.close()
    if (_perform_traditional):
        f_traditional.close()

    ## admin classifier
    f_transductive = open('res/admin' + _suffix, 'w')
    if (_perform_traditional):
        f_traditional = open('res/admin-traditional.csv', 'w')

    for lang_source in langs:
        y_source = (y[lang_source] == 2)

        if (_perform_traditional):
            ## traditional learning
            X_fit, X_eval, y_fit, y_eval= train_test_split(X[lang_source], y_source, test_size=0.368, random_state=42)

            if (len(np.unique(y_fit)) == 1) or (len(np.unique(y_eval)) == 1):
                # ROC not defined
                auc = np.nan
            else:
                # classifier
                clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
                clf.fit(X_fit, y_fit)

                y_predict = clf.predict_proba(X_eval)[:,1]
                auc = roc_auc_score(y_eval, y_predict)

            f_traditional.write(lang_source + ',' + str(auc) + '\n')

        # transductive learning
        ## classifier
        clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        clf.fit(X[lang_source], y_source)

        ## evaluation
        for lang_target in langs:
            y_target = (y[lang_target] == 2)
            if (len(np.unique(y_source)) == 1) or (len(np.unique(y_target)) == 1):   # ROC not defined
                auc = np.nan
            else:
                y_predict = clf.predict_proba(X[lang_target])[:,1]
                auc = roc_auc_score(y_target, y_predict)

            f_transductive.write(lang_source + ',' + lang_target + ',' + str(auc) + '\n')

    f_transductive.close()
    if (_perform_traditional):
        f_traditional.close()

if __name__ == '__main__':
    # init
    _suffix = '-transductive-' + sys.argv[1] + '.csv'
    _perform_traditional = False

    if (sys.argv[1] in [ 'degree', 'all' ]):
        _transform_func_degree = degree_transform
        _transform_func = no_transform
    else:
        _transform_func_degree = no_transform
        _transform_func = no_transform
        _perform_traditional = False

    main()

