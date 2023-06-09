# %%
import random
import time
import pandas as pd
import json
import cohere
from tqdm import tqdm

# %%
'''
Load dataset .json file. 
'''
dataset_path = '../augmented_test.json'

with open(dataset_path) as file:
    dataset = json.load(file)

# %%
'''
Choose random sample from dataset. Save the sample indices to 'cohere_sample_indices.csv'.  
'''
'''
sample_size = 200 # for testing, change to ~200 for actual experiment

sample_indices = random.sample(range(len(dataset)), sample_size)
df_indices = pd.DataFrame(sample_indices, columns=['Indices'])
df_indices.to_csv('cohere_sample_indices.csv', index=False) # save indices for later summarization comparison
'''

# %%
'''
Read in sample indices from 'cohere_sample_indices.csv', get random sample from dataset.
Combine documents within each datapoint to feed into cohere api. 
'''
df_indices = pd.read_csv('../cohere_sample_indices.csv')
sample_indices = df_indices['Indices'].values.tolist()
#sample_indices = sample_indices[0:5] # take out when running actual experiment

sample_dataset_text = [dataset[i]['documents'] for i in sample_indices]

input_size = 4096
short_sample_dataset_text = []
for datapoint in sample_dataset_text:
    document_size = int(0.5 * input_size / len(datapoint)) # each doc's token length is 1/(# documents) input_size, assume each word is 2 tokens
    first_words = [doc[:document_size] for doc in datapoint]
    short_sample_dataset_text.append(first_words)

#concat_sample_dataset = [' '.join(i) for i in sample_dataset_text] # for full documents
concat_short_sample_dataset = [' '.join(i) for i in short_sample_dataset_text]

#original_text = concat_sample_dataset # for full documents
shortened_text = concat_short_sample_dataset

# %%
'''
Use cohere api to generate summaries, and save to output file 'cohere_summaries.txt'.
'''

co = cohere.Client('PcE6kHvoamLYwNGqcGHcnkpF23LV4WkCO4CSH1mB')

'''
with open('originaltext_cohere_summaries.txt', 'w') as file:
    for i in range(len(original_text)):
        cohere_summary = co.summarize(text=original_text[i],)
        summary = cohere_summary.summary
        file.write(summary + '\n')

        time.sleep(12)
'''

assert len(shortened_text) == len(sample_indices), "shortened_text should length 200"

shortened_summaries = []
for i in tqdm(range(len(shortened_text))):
    result = {}
    
    try:
        cohere_summary = co.summarize(text=shortened_text[i], length='long', format='paragraph')
        summary = cohere_summary.summary
        result['generated_summary'] = cohere_summary.summary
    except Exception as e:
        print(f"An error occurred for index {i}: {str(e)}")
        result['generated_summary'] = ""  # Append an empty value in case of an error
    
    dataset_i = sample_indices[i]
    result['gt_summary'] = dataset[dataset_i]['summary']
    shortened_summaries.append(result)
    
    time.sleep(12) # only 5 calls per minute with free cohere key
with open("results/shortenedtext_cohere_summaries.json", "w") as file:
    json.dump(shortened_summaries, file)
    


