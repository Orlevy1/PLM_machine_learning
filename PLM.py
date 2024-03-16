# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns

# Read the text files containing the IDs of sequences for the train, test, and validation sets
train_ids = pd.read_csv('train_set_proteins.txt', header=None, names=['ID'])
test_ids = pd.read_csv('test_set_proteins.txt', header=None, names=['ID'])
validation_ids = pd.read_csv('validation_set_proteins.txt', header=None, names=['ID'])

# Read the CSV file containing the features (feature map) for all sequences
feature_map = pd.read_csv('sequence_representations.csv', header=None, index_col=0)

# Implementing the code from homework2 last year for the other model feature map
# We run in only once to save time, no need to run it again

'''
list_amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def aa_content_feature_map(sequence):
    counts = np.zeros(len(list_amino_acids))

    L = len(sequence)
    for s in sequence:
        try:
            counts[list_amino_acids.index(s)] += 1
        except:  # If unknown amino acid, will trigger exception.
            continue  # Ignore

    frequencies = counts / counts.sum()
    return frequencies

'''

# Preprocess the data to create train, test, and validation sets based on the IDs
train_data = feature_map[feature_map.index.isin(train_ids['ID'])]
test_data = feature_map[feature_map.index.isin(test_ids['ID'])]
validation_data = feature_map[feature_map.index.isin(validation_ids['ID'])]

# Read the CSV file containing the target variable
unique_protein_sequences = pd.read_csv('unique_protein_sequences.csv')

# Manipulation on the data for the second feature map

'''
unique_protein_sequences['frequencies'] = unique_protein_sequences['protein_sequence'].apply(aa_content_feature_map)

for i, aa in enumerate(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']):
    unique_protein_sequences[aa+'_frequency'] = unique_protein_sequences['frequencies'].apply(lambda x: x[i])

unique_protein_sequences = unique_protein_sequences.drop(columns=['frequencies'])
unique_protein_sequences.to_csv('new_features.csv', index=False)
'''

# Second model feature map preprocess and split
feature_map_comper = pd.read_csv('new_features.csv', header=None, index_col=0)

train_data_copmper = feature_map_comper[feature_map_comper.index.isin(train_ids['ID'])]
test_data_comper = feature_map_comper[feature_map_comper.index.isin(test_ids['ID'])]
validation_data_comper = feature_map_comper[feature_map_comper.index.isin(validation_ids['ID'])]

# Extract the indices present in train_data
train_indices = train_data.index
test_indices = test_data.index

# Initialize an empty list to store the selected sequences
selected_tm_train = []

# Iterate over the indices of unique_protein_sequences
for seq_id in unique_protein_sequences['seq_id']:
    # Check if the index is in train_indices
    seq_id_int = int(seq_id)
    if seq_id_int in train_indices:
        index = unique_protein_sequences.index[unique_protein_sequences['seq_id'] == seq_id].tolist()[0]
        selected_tm_train.append(unique_protein_sequences.loc[index][4])

selected_tm_test = []
for seq_id in unique_protein_sequences['seq_id']:
    # Check if the index is in train_indices
    seq_id_int = int(seq_id)
    if seq_id_int in test_indices:
        index = unique_protein_sequences.index[unique_protein_sequences['seq_id'] == seq_id].tolist()[0]
        selected_tm_test.append(unique_protein_sequences.loc[index][4])


# Train the model using scikit-learn

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 30, random_state = 0)
regressor.fit(train_data, selected_tm_train)

# Predict thermostatic temperature for the test data
y_true = selected_tm_test
y_pred = regressor.predict(test_data)
predictions_list = y_pred.tolist()

# Calculate the differences between corresponding elements
differences = [abs(y_true[i] - predictions_list[i]) for i in range(len(y_true))]
# Calculate the average of the differences
average_difference = np.mean(differences)
print(average_difference)

# Calculate Pearson correlation coefficient and p-value
pearson_corr, p_value = pearsonr(y_true, y_pred)
print("Pearson correlation coefficient:", pearson_corr)

regressor_comp = RandomForestRegressor(n_estimators = 30, random_state = 0)
regressor_comp.fit(train_data_copmper, selected_tm_train)

# Predict tm for the test data of the second model
y_true_comp = selected_tm_test
y_pred_comp = regressor_comp.predict(test_data_comper)
predictions_list_comp = y_pred_comp.tolist()

# Calculate the differences between corresponding elements
differences_comp = [abs(y_true_comp[i] - predictions_list_comp[i]) for i in range(len(y_true_comp))]
# Calculate the average of the differences
average_difference_comp = np.mean(differences_comp)
print(average_difference_comp)

# Calculate Pearson correlation coefficient and p-value
pearson_corr_comp, p_value_comp = pearsonr(y_true_comp, y_pred_comp)

print("Pearson correlation coefficient for second model:", pearson_corr_comp)



# Visualization part

thermo_germs =[185,581,668,968,1314,1780,1789,1804,2109,2469,2581,2591,2652,2768,3307,3789,4224,4784,4946,4974,5010,5047,5133,5936,5999]
homosapians =[5,6,83,150,179,183,227,664,728,758,927,937,1258,786,841,879]

tm_thermo_germs =[]
tm_homo=[]
pred_germs = []
pred_germs_comp=[]
pred_homo =[]
pred_homo_comp=[]

# Get the predicted values from the list and also the real values for both models

for tm in thermo_germs:
    index = test_indices.get_loc(tm)
    tm_thermo_germs.append(selected_tm_test[index])

for tm in homosapians:
    index = test_indices.get_loc(tm)
    tm_homo.append(selected_tm_test[index])

for tm in thermo_germs:
    index = test_indices.get_loc(tm)
    pred_germs.append(predictions_list[index])

for tm in homosapians:
    index = test_indices.get_loc(tm)
    pred_homo.append(predictions_list[index])
    
    
for tm in thermo_germs:
    index = test_indices.get_loc(tm)
    pred_germs_comp.append(predictions_list_comp[index])

for tm in homosapians:
    index = test_indices.get_loc(tm)
    pred_homo_comp.append(predictions_list_comp[index])
    
   


# Create histogram plots for pred_germs and pred_homo separately
plt.figure(figsize=(10, 6))

sns.histplot(pred_germs, color='blue', kde=True, label='pred_germs')
sns.histplot(pred_homo, color='red', kde=True, label='pred_homo')

# Set plot title and labels
plt.title('Distribution of Predicted Values')
plt.xlabel('Predicted Value')
plt.ylabel('Frequency')

# Show legend
plt.legend()

# Show plot
plt.show()


# Create histogram plots for tm_thermo_germs and tm_homo separately
plt.figure(figsize=(10, 6))

sns.histplot(tm_thermo_germs, color='blue', kde=True, label='tm_thermo_germs')
sns.histplot(tm_homo, color='red', kde=True, label='tm_homo')

# Set plot title and labels
plt.title('Distribution of Real Values')
plt.xlabel('Real Value')
plt.ylabel('Frequency')

# Show legend
plt.legend()

# Show plot
plt.show()


# Create histogram plots for pred_germs_comp and pred_homo_comp separately
plt.figure(figsize=(10, 6))

sns.histplot(pred_germs_comp, color='blue', kde=True, label='pred_germs_comp')
sns.histplot(pred_homo_comp, color='red', kde=True, label='pred_homo_comp')

# Set plot title and labels
plt.title('Distribution of Predicted Values - comperision')
plt.xlabel('Predicted Value')
plt.ylabel('Frequency')

# Show legend
plt.legend()

# Show plot
plt.show()

