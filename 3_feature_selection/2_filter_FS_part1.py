### Created by Myongin Oh
### Last updated on Aug 23, 2025

### STEP 0. Import libraries
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
import matplotlib.pyplot as plt
import gc
import seaborn as sns
import multiprocessing
import fisher_score_mod as fsm

### STEP 1. Load input data
dfReduced_standard = pd.read_csv('output_dist_res_with_class_post_var.csv')

### STEP 2. Calculate Fisher Scores
X = dfReduced_standard.drop('class', axis=1)
# Fisher's score
score = fsm.fisher_score(X.to_numpy(), dfReduced_standard['class'].to_numpy(), mode='score')
#print(score)
dfFScore = pd.DataFrame(score)
dfFScore.to_csv('fisher_scores_dfReduced.csv', encoding='utf-8', index=True, header=None)

idx = fisher_score.fisher_score(X.to_numpy(), dfReduced_standard['class'].to_numpy(), mode='index')   
#print(idx)
dfFSidx = pd.DataFrame(idx)
dfFSidx.to_csv('fisher_indices_dfReduced.csv', encoding='utf-8', index=True, header=None)

# Read the CSV file containing Fisher scores
df_scores = pd.read_csv('fisher_scores_dfReduced.csv')
df_scores.columns = ['ind', 'fs']

# Read the CSV file containing indices (if needed for reference)
df_indices = pd.read_csv('fisher_indices_dfReduced.csv')

# Sort the DataFrame by the 'fs' column in descending order
df_scores_sorted = df_scores.sort_values(by='fs', ascending=False)

# Save the sorted DataFrame to a new CSV file
df_scores_sorted.to_csv('fisher_scores_dfReduced_descending.csv', index=False)