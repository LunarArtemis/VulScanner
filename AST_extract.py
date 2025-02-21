
# Extract AST from C source code using clang
import clang.cindex
import sys
import json
import os
import dask.dataframe as dd # for parallel computing 
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Tuple
clang.cindex.Config.set_library_path(r"C:\Program Files\LLVM\bin")

def save_ast(node):
    node.children = list(node.get_children())

    for child in node.children:
        counter = save_ast(child)
        
def numbering_ast_nodes(node, counter=1):
    node.identifier = counter
    counter += 1

    node.children = list(node.get_children())
    for child in node.children:
        counter = numbering_ast_nodes(child, counter)

    return counter

def generate_edgelist(ast_root):
    edges = []

    def walk_tree_and_add_edges(node):
        for child in node.children:
            edges.append([node.identifier, child.identifier])
            walk_tree_and_add_edges(child)

    walk_tree_and_add_edges(ast_root)

    return edges

def generate_features(ast_root):
    
    features = {}

    def walk_tree_and_set_features(node):
        out_degree = len(node.children)
        #in_degree = 1
        #degree = out_degree + in_degree
        degree = out_degree

        features[node.identifier] = degree

        for child in node.children:
            walk_tree_and_set_features(child)

    walk_tree_and_set_features(ast_root)

    return features

def get_source_file(datapoints):
    if len(datapoints) == 1:
        return datapoints.iloc[0]

def clang_process(testcase, **kwargs):
    
    parse_list = [
        (datapoint.filename, datapoint.code)
        for datapoint in testcase.itertuples()
    ]

    source_file= get_source_file(testcase)

    # Parsing the source code and extracting AST using clang
    index = clang.cindex.Index.create()
    translation_unit = index.parse(
        path=source_file.filename,
        unsaved_files=parse_list,
    )
    ast_root = translation_unit.cursor

    save_ast(ast_root)
    numbering_ast_nodes(ast_root)

    edgelist = generate_edgelist(ast_root)

    features = generate_features(ast_root)

    graph_representation = {
        "edges": edgelist,
        "features": features,
    }

    # delete clang objects
    del translation_unit
    del ast_root
    del index

    # Writing to sample.json
    # with open("sample.json", "w") as outfile:
    #     json.dump(graph2vec_representation,outfile)
    return json.dumps(graph_representation)

def process_dataset(csv_location, output_location, num_partitions=20):
    print("Preprocess source code files and extracting AST's")

    data = pd.read_csv(csv_location)
    data = dd.from_pandas(data, npartitions=num_partitions)

    graphs = data.groupby(['testID']).apply(
        clang_process,
        axis='columns',
        meta=('processed_for_graph2vec', 'unicode'),
    )

    # Compute the Dask DataFrame to get a pandas Series
    computed_graphs = graphs.compute()

    graph2vec_input_dir = output_location + "/graph2vec_input/"
    os.makedirs(graph2vec_input_dir, exist_ok=True)

    # Use items() instead of iteritems() on the computed pandas Series
    for index, row in computed_graphs.items():
        print("Current Iteration: " + str(index))
        with open(graph2vec_input_dir + str(index) + ".json", 'w') as f:
            f.write(row)

    print("-> Done.")
    return graph2vec_input_dir

csv_location = 'Datasets/Normalized_CWE-469.csv.gz'
output_location = 'Datasets/'
process_dataset(csv_location, output_location, num_partitions=20)