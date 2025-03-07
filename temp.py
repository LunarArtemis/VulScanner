# Extract AST from C source code using clang
import clang.cindex
import sys
import json
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Tuple

# Configure libclang path
if os.name == 'nt':  # Windows
    print('Windows')
    clang.cindex.Config.set_library_file('D:/Project/LLVM/bin/libclang.dll')
    print(clang.cindex.Config.library_path)
elif os.name == 'posix':  # Linux/Mac
    print('Linux/Mac')
    clang.cindex.Config.set_library_file('/Library/Developer/CommandLineTools/usr/lib/libclang.dylib')
    # clang.cindex.Config.set_library_path('/Library/Developer/CommandLineTools/usr/lib/')
    print(clang.cindex.Config.library_path)

# Verify if libclang is loaded
print(clang.cindex.Config.loaded)  # Should print `True`

def save_ast(node):
    """ Recursively save the AST in a dictionary format """
    node.children = list(node.get_children())

    for child in node.children:
        save_ast(child)
        
def numbering_ast_nodes(node, counter=1):
    """ Recursively number the AST nodes """
    node.identifier = counter
    counter += 1

    node.children = list(node.get_children())
    for child in node.children:
        counter = numbering_ast_nodes(child, counter)

    return counter

def generate_edgelist(ast_root):
    """ Generate an edge list from the AST """
    edges = []

    def walk_tree_and_add_edges(node):
        for child in node.children:
            edges.append([node.identifier, child.identifier])
            walk_tree_and_add_edges(child)

    walk_tree_and_add_edges(ast_root)

    return edges

def generate_features(ast_root):
    """ Generate features for each node in the AST """
    features = {}

    def walk_tree_and_set_features(node):
        out_degree = len(node.children)
        degree = out_degree

        features[node.identifier] = degree

        for child in node.children:
            walk_tree_and_set_features(child)

    walk_tree_and_set_features(ast_root)

    return features

def get_source_file(datapoints):
    """ Get the source file from the list of datapoints """
    if len(datapoints) == 1:
        return datapoints.iloc[0]

def clang_process(testcase, **kwargs):
    """Parses source code with Clang and extracts AST-based graph representation."""
    parse_list = [
        (testcase.filename, testcase.code)
    ]

    # source_file = get_source_file(testcase)

    # Parsing the source code and extracting AST using clang
    index = clang.cindex.Index.create()
    translation_unit = index.parse(
        path=testcase.filename,
        unsaved_files=parse_list,
    )
    ast_root = translation_unit.cursor

    save_ast(ast_root)
    numbering_ast_nodes(ast_root)

    graphs_embedding = generate_edgelist(ast_root)
    nodes_embedding = generate_features(ast_root)

    y = torch.tensor([testcase.vuln], dtype=torch.int64)

    # delete clang objects
    del translation_unit
    del ast_root
    del index

    return Data(x=nodes_embedding, edge_index=graphs_embedding, y=y)

class GenDatasets(Dataset):
    def __init__(self, csv_path, root, transform=None, pre_transform=None):
        """
        Args:
            csv_path (str): Path to the CSV dataset.
            root (str): Root directory where processed data will be stored.
            transform (callable, optional): Optional transform to be applied on a sample.
            pre_transform (callable, optional): Optional pre-transform before processing.
        """
        self.csv_path = csv_path
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        super(GenDatasets, self).__init__()
        
        self.processed_dir = os.path.join(root, 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)
        self.data = pd.read_csv(self.csv_path)
    
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.data))]

    def download(self):
        pass  # No downloading required

    def process(self):
        for index, vuln in tqdm(self.data.iterrows(), total=len(self.data)):
            data = clang_process(vuln)
            torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))
    
    def len(self):
        return len(self.data)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
    
dataset = GenDatasets(csv_path="Datasets/Normalized_CWE-469.csv", root="./data")
dataset.process()