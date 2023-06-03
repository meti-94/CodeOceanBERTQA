import os
from data import *
print('\n\n', '#'*20, "Table 2".upper(), '#'*20)
df = pd.read_csv(r'../data/Reverb1.1/reverb_wikipedia_tuples-1.1.txt', sep='\t', header=None)
reverb_columns_name = ['ExID', 'arg1', 'rel', 'arg2', 'narg1', 'nrel', 'narg2', 'csents', 'conf', 'urls']
df.columns = reverb_columns_name
df = df.dropna()
df = df.drop_duplicates()

print(f'#Triples\t\t:\t{len(df)}')
print(f'#Relations\t\t:\t{len(df["rel"].unique())}')
print(f'#Entity 1\t\t:\t{len(df["arg1"].unique())}')
print(f'#Entity 2\t\t:\t{len(df["arg2"].unique())}')
print(f'Total Unique Entities\t:\t{len(set(df["arg1"].unique().tolist()+df["arg2"].unique().tolist()))}')
vocab = df["arg1"].unique().tolist()+df["arg2"].unique().tolist()+df["rel"].unique().tolist()
vocab = list(map(lambda x:x.split(), vocab))
vocab = [item for sublist in vocab for item in sublist]
print(f'Vocabulary Size\t\t:\t{len(set(vocab))}')