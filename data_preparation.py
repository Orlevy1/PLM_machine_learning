import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict


# Read the TSV file from mmseqs2 and create a list of edges
edges = []
with open('clusterRes_cluster_fixed.tsv', 'r') as file:
    for line in file:
        vertex1, vertex2 = map(int, line.strip().split('\t'))
        edges.append((vertex1, vertex2))

# Determine the number of vertices
num_vertices = max(max(edge) for edge in edges) + 1

# Create CSR matrix data
row_indices = [edge[0] for edge in edges]
col_indices = [edge[1] for edge in edges]
data = np.ones(len(edges), dtype=int)

# Create a full binary adjacency matrix with 0 where there is no edge
adjacency_matrix_dense = np.zeros((num_vertices, num_vertices), dtype=int)

# Fill the matrix with 1 where there is an edge
adjacency_matrix_dense[row_indices, col_indices] = 1
adjacency_matrix_dense[col_indices, row_indices] = 1

# Compute connected components
n_components, labels = connected_components(adjacency_matrix_dense, directed=False)

# Extract the verticies from the connected components
comp_list = [[] for _ in range(n_components)]
df = pd.read_csv('unique_protein_sequences.csv')
with open('unique_protein_sequences.csv', 'r') as file:
    for i in range(len(labels)):
        comp_list[labels[i]].append(i)

# Save only seq_ids from our data (len 400>)
for i in range(len(comp_list)):
    for vert in comp_list[i]:
        if vert not in df['seq_id'].values:
            comp_list[i].remove(vert)

# Remove empty components
for comp in comp_list:
    if not comp:
        comp_list.remove(comp)

# Save as a dictionary - the key is the connected component and the value is the verticies
labels_dict = {}
for i in range(len(comp_list)):
    for vert in comp_list[i]:
        labels_dict[vert] = i
my_dataframe = pd.DataFrame(list(labels_dict.items()), columns=['Keys', 'Values'])

# Create a defaultdict to store lists of keys for each value
value_key_map = defaultdict(list)

# Iterate over each row in the DataFrame
for index, row in my_dataframe.iterrows():
    value_key_map[row['Values']].append(row['Keys'])

# Convert the defaultdict to a list - now we have a list of lists
# Each inside list is a connected component with seq_ids of the vert in the same component
result_list = list(value_key_map.values())

# Shuffle the order of connected components
np.random.seed(42)
np.random.shuffle(labels)

# Split into train, test, and validation sets
train_size = int(0.8 * len(result_list))
test_size = int(0.19 * len(result_list))
validation_size = len(result_list) - train_size - test_size

train_set, remaining_set = train_test_split(result_list, test_size=test_size + validation_size, random_state=42)
test_set, validation_set = train_test_split(remaining_set, test_size=validation_size, random_state=42)

# Extract the proteins to lists according to train test val split
train_set_proteins = []
for comp in train_set:
    for protein in comp:
        train_set_proteins.append(protein)

test_set_proteins = []
for comp in test_set:
    for protein in comp:
        test_set_proteins.append(protein)

validation_set_proteins = []
for comp in validation_set:
    for protein in comp:
        validation_set_proteins.append(protein)

# Create text files for PLM code
with open('train_set_proteins.txt', 'w') as file:
    for item in train_set_proteins:
        file.write(str(item) + '\n')
with open('test_set_proteins.txt', 'w') as file:
    for item in test_set_proteins:
        file.write(str(item) + '\n')
with open('validation_set_proteins.txt', 'w') as file:
    for item in validation_set_proteins:
        file.write(str(item) + '\n')
