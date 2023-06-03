import os
from data import *
print('\n\n', '#'*20, "Table 3".upper(), '#'*20)
train_df = pd.read_excel('../data/Intermediate/train.xlsx'); valid_df = pd.read_excel('../data/Intermediate/valid.xlsx'); test_df = pd.read_excel('../data/Intermediate/test.xlsx')


def get_unique_ent_rel(dataframe):
    arg1 = [eval(item)[0] for item in dataframe['triple'].to_list()]
    arg2 = [eval(item)[2] for item in dataframe['triple'].to_list()]
    rel = [eval(item)[1] for item in dataframe['triple'].to_list()]
    print(f'Number of Questions\t:\t{len(dataframe)}')
    print(f'Entity 1\t:\t{len(set(arg1))}')
    print(f'Entity 2\t:\t{len(set(arg2))}')
    print(f'Relations\t:\t{len(set(rel))}')
    print(f'Total Unique Entities\t:\t{len(set(arg1+arg2))}')
    tokenizer = lambda string:string.strip().lower().split()
    tokenized_questions = dataframe['Question'].astype(str).apply(tokenizer).to_list()
    flatten_tokenized_questions = [item for sublist in tokenized_questions for item in sublist]
    print(f'Unique Words\t:\t{len(set(flatten_tokenized_questions))}')

print("*** Training ***")
get_unique_ent_rel(train_df)
print("\n*** Validation ***")
get_unique_ent_rel(valid_df)
print("\n*** Test ***")
get_unique_ent_rel(test_df)