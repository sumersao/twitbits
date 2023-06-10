# %%
import random
import pandas as pd
import json
import os

# %%
'''
Load dataset .json files (baseline, best, cohere). 
'''

baseline_path = '../results/primera_baseline_complete.json'
best_cluster_path = '../../kmeans_bert_concat_complete.json' # replace with path of best performing clustering/summarizing method
cohere_path = '../results/shortenedtext_cohere_summaries.json'

with open(baseline_path) as file:
    baseline = json.load(file)

with open(best_cluster_path) as file:
    best_cluster = json.load(file)

with open(cohere_path) as file:
    cohere = json.load(file)

# %%
'''
Grabbing random index of generated summaries for testing purposes.
'''
'''
print('baseline summary: ')
print(baseline[8]['generated_summary'])
print()

print('generated summary: ')
print(best_cluster[8]['generated_summaries'])
print()

print('cohere summary: ')
print(cohere[8]['generated_summary'])
print()

print('ground truth summary: ')
print(cohere[8]['gt_summary'])
'''

# %%
'''
From list of random 200 indices, pick random n = 20 to do qualitative analysis on.
'''

df_indices = pd.read_csv('../cohere_sample_indices.csv')
sample_indices = df_indices['Indices'].values.tolist()

num_eval = 20
sample_indices = random.sample(range(len(sample_indices)), num_eval)
print(sample_indices)

# %%
'''
Get generated summaries of randomly selected samples from each dataset. Organize into dictionary.
'''
assert len(sample_indices) == 20, "Sample indices is not of length 20."

generated_summaries = []

for index in sample_indices:
    generated_summary = {}
    
    baseline_summary = baseline[index]['generated_summary']
    best_summary = best_cluster[index]['generated_summaries']
    cohere_summary = cohere[index]['generated_summary']
    gold_summary = cohere[index]['gt_summary']

    generated_summary['gold'] = cohere[index]['gt_summary']
    generated_summary['baseline'] = baseline[index]['generated_summary']
    generated_summary['best'] = best_cluster[index]['generated_summaries']
    generated_summary['cohere'] = cohere[index]['generated_summary']

    generated_summaries.append(generated_summary)


#generated_summaries[3]



# %%
'''
Qualitative experiment system. 
For each sample, prints gold summary, then baseline/best/cohere summaries in random order. User is then able to rank the 4 summaries blindly.
When user types 'continue', the labels are displayed in order for data collection. 
Whe user types 'next', we continue to the next sample. Goes through a total of n = 20 samples for experiment completion.
'''

print('Starting qualitative experiment.')
print()

for generated_summary in generated_summaries:

    # First clear terminal screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # Then print ground truth label
    print('Ground Truth Summary: ') 
    print(generated_summary['gold'])
    print()

    # Then print baseline/best/cohere summaries in random order
    order = random.sample(['baseline', 'best', 'cohere'], 3)
    
    for item in order:
        print('Summary: ')
        print(generated_summary[item])
        print()
    
    # Pause until 'continue' is entered (to get labeled order of current sample)
    user_input = input("Type 'continue' to proceed: ")
    while user_input.lower() != 'continue':
        user_input = input("Type 'continue' to proceed: ")

    # Print the order of the generated summaries
    print("Order:", ", ".join(order))

    # Pause until 'continue' is entered (to proceed to next sample)
    user_input = input("Type 'next' to proceed: ")
    while user_input.lower() != 'next':
        user_input = input("Type 'next' to proceed: ")

print('Qualitative experiment complete.')


