import pandas as pd
from graphviz import Digraph

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz-10.0.1-win64/bin'


# Define functions to visualize transition matrices as graphs
def edgelist_to_digraph(df_edgelist):
    ''' Convert an edge list into a weighted directed graph
    '''
    g = Digraph(format='pdf')
    g.attr(rankdir='LR', size='30')
    g.attr('node', shape='circle')
    nodelist = []
    for _, row in df_edgelist.iterrows():
        node1, node2, weight = [str(item) for item in row]
        if node1 not in nodelist:
            g.node(node1, **{'width': '1', 'height': '1'})
            nodelist.append(node1)
        if node2 not in nodelist:
            g.node(node2, **{'width': '1', 'height': '1'})
            nodelist.append(node2)
        g.edge(node1, node2, label=weight)
    return g


def visualize_matrix(matrix, fname):
    edgelist = []
    for key in matrix.keys():
        src = str(key[0])
        src.rstrip("0")
        dst = str(key[1])
        dst.rstrip("0")
        weight = round(100*matrix[key], 2)
        weight = str(weight) + '%'
        edgelist.append([src, dst, weight])
    df = pd.DataFrame(edgelist, columns=['src', 'dst', 'weight'])
    g = edgelist_to_digraph(df)
    g.render(fname, view=False)
    os.remove(fname)
