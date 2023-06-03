#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install transformers -q')
get_ipython().system('pip install pandas -q')
get_ipython().system('pip install scikit-learn -q')
get_ipython().system('pip install openpyxl -q')
get_ipython().system('pip install tabulate -q')
get_ipython().system('pip install PatternLite -q')
import nltk
nltk.download('omw-1.4')


# In[12]:


import os
from data import *


# In[13]:


# check if Bertified data exists
if not os.path.isfile('../data/Bertified/entities.npy'):
    reverb_lines = read_reverb('../data/Reverb1.1/reverb_wikipedia_tuples-1.1.txt')
    questions = pd.read_excel('../data/ReverbSQA/Final_Sheet_990824.xlsx', sheet_name=1, engine='openpyxl')
    index = get_tuple_frequency(reverb_lines, questions)
    index[index['Frequency']<10].to_excel('../data/ProcessedQuestions/normalized_questions.xlsx')
    combine_with_reverb(questions_path='../data/ProcessedQuestions/normalized_questions.xlsx', 
                    reverb_path='../data/Reverb1.1/reverb_wikipedia_tuples-1.1.txt')
    create_bertified_dataset()


# In[19]:


with open('/root/capsule/results/output.txt', 'w') as f:
    f.write('Start')


# In[31]:


get_ipython().run_cell_magic('capture', 'cap --no-stderr', '# table 1 output\n\nprint(\'\\n\\n\', \'#\'*20, "Table 1".upper(), \'#\'*20)\nprint(pd.read_excel(\'../data/Intermediate/train.xlsx\').sample(6)[[\'triple\', \'Question\']])\nwith open(\'/root/capsule/results/output.txt\', \'a\') as f:\n    f.write(cap.stdout)\n\n')


# In[32]:


get_ipython().run_cell_magic('capture', 'cap --no-stderr', '# table 2 output\n\nprint(\'\\n\\n\', \'#\'*20, "Table 2".upper(), \'#\'*20)\ndf = pd.read_csv(r\'../data/Reverb1.1/reverb_wikipedia_tuples-1.1.txt\', sep=\'\\t\', header=None)\nreverb_columns_name = [\'ExID\', \'arg1\', \'rel\', \'arg2\', \'narg1\', \'nrel\', \'narg2\', \'csents\', \'conf\', \'urls\']\ndf.columns = reverb_columns_name\ndf = df.dropna()\ndf = df.drop_duplicates()\n\nprint(f\'#Triples\\t\\t:\\t{len(df)}\')\nprint(f\'#Relations\\t\\t:\\t{len(df["rel"].unique())}\')\nprint(f\'#Entity 1\\t\\t:\\t{len(df["arg1"].unique())}\')\nprint(f\'#Entity 2\\t\\t:\\t{len(df["arg2"].unique())}\')\nprint(f\'Total Unique Entities\\t:\\t{len(set(df["arg1"].unique().tolist()+df["arg2"].unique().tolist()))}\')\nvocab = df["arg1"].unique().tolist()+df["arg2"].unique().tolist()+df["rel"].unique().tolist()\nvocab = list(map(lambda x:x.split(), vocab))\nvocab = [item for sublist in vocab for item in sublist]\nprint(f\'Vocabulary Size\\t\\t:\\t{len(set(vocab))}\')\nwith open(\'/root/capsule/results/output.txt\', \'a\') as f:\n    f.write(cap.stdout)\n')


# In[33]:


get_ipython().run_cell_magic('capture', 'cap --no-stderr', '# table 3 output\n\nprint(\'\\n\\n\', \'#\'*20, "Table 3".upper(), \'#\'*20)\ntrain_df = pd.read_excel(\'../data/Intermediate/train.xlsx\'); valid_df = pd.read_excel(\'../data/Intermediate/valid.xlsx\'); test_df = pd.read_excel(\'../data/Intermediate/test.xlsx\')\n\n\ndef get_unique_ent_rel(dataframe):\n    arg1 = [eval(item)[0] for item in dataframe[\'triple\'].to_list()]\n    arg2 = [eval(item)[2] for item in dataframe[\'triple\'].to_list()]\n    rel = [eval(item)[1] for item in dataframe[\'triple\'].to_list()]\n    print(f\'Number of Questions\\t:\\t{len(dataframe)}\')\n    print(f\'Entity 1\\t:\\t{len(set(arg1))}\')\n    print(f\'Entity 2\\t:\\t{len(set(arg2))}\')\n    print(f\'Relations\\t:\\t{len(set(rel))}\')\n    print(f\'Total Unique Entities\\t:\\t{len(set(arg1+arg2))}\')\n    tokenizer = lambda string:string.strip().lower().split()\n    tokenized_questions = dataframe[\'Question\'].astype(str).apply(tokenizer).to_list()\n    flatten_tokenized_questions = [item for sublist in tokenized_questions for item in sublist]\n    print(f\'Unique Words\\t:\\t{len(set(flatten_tokenized_questions))}\')\n\nprint("*** Training ***")\nget_unique_ent_rel(train_df)\nprint("\\n*** Validation ***")\nget_unique_ent_rel(valid_df)\nprint("\\n*** Test ***")\nget_unique_ent_rel(test_df)\nwith open(\'/root/capsule/results/output.txt\', \'a\') as f:\n    f.write(cap.stdout)\n')


# In[37]:


get_ipython().run_cell_magic('capture', 'cap --no-stderr', '\nprint(\'\\n\\n\', \'#\'*20, "Table 5 GRU".upper(), \'#\'*20)\n# table 5 output {GRU\'s row}\n!python3 ./BuboQA/entities/train.py  --entity_detection_mode GRU \\\n                                    --fix_embed --data_dir ../data/SimpleQuestionNotationEntity \\\n                                    --batch_size 256 \\\n                                    --vector_cache ../data/Cache/sq_glove300d.pt \\\n\nprint(\'\\n\\n\', \'#\'*20, "Table 5 LSTM".upper(), \'#\'*20)\n# table 5 output {GRU\'s row}\n!python3 ./BuboQA/entities/train.py  --entity_detection_mode LSTM \\\n                                    --fix_embed --data_dir ../data/SimpleQuestionNotationEntity \\\n                                    --batch_size 256 \\\n                                    --vector_cache ../data/Cache/sq_glove300d.pt \\\n\nprint(\'\\n\\n\', \'#\'*20, "Table 6 LSTM".upper(), \'#\'*20)\n# table 6 output {LSTM\'s row}\n!python3 ./BuboQA/relations/train.py  --relation_prediction_mode LSTM \\\n                                     --fix_embed --data_dir ../data/SimpleQuestionNotationRelation \\\n                                     --batch_size 256 \\\n                                     --vector_cache ../data/Cache/sq_glove300d.pt\n\nprint(\'\\n\\n\', \'#\'*20, "Table 6 CNN".upper(), \'#\'*20)\n# table 6 output {CNN\'s row}\n!python3 ./BuboQA/relations/train.py  --relation_prediction_mode CNN \\\n                                     --fix_embed --data_dir ../data/SimpleQuestionNotationRelation \\\n                                     --batch_size 256 \\\n                                     --vector_cache ../data/Cache/sq_glove300d.pt\n\nwith open(\'/root/capsule/results/output.txt\', \'a\') as f:\n    f.write(cap.stdout)\n')


# In[46]:


get_ipython().run_cell_magic('capture', 'cap --no-stderr', '# table 7 & 8 & 9 & 10 output\n\n%cd /root/capsule/code/src\nprint(\'\\n\\n\', \'#\'*20, "Table 7 & 8 Test".upper(), \'#\'*20)\n!python3 train.py False NodeEdgeDetector rsq test\nprint(\'\\n\\n\', \'#\'*20, "Table 7 & 8 Valid".upper(), \'#\'*20)\n!python3 train.py False NodeEdgeDetector rsq valid\nprint(\'\\n\\n\', \'#\'*20, "Table 9 & 10 Test".upper(), \'#\'*20)\n!python3 train.py False NodeEdgeDetector sq test\nprint(\'\\n\\n\', \'#\'*20, "Table 9 & 10 Valid".upper(), \'#\'*20)\n!python3 train.py False NodeEdgeDetector sq valid\nwith open(\'/root/capsule/results/output.txt\', \'a\') as f:\n    f.write(cap.stdout)\n')


# In[12]:


get_ipython().run_cell_magic('capture', 'cap --no-stderr', '# table 11 output\n\n%cd /root/capsule/code/src\nprint(\'\\n\\n\', \'#\'*20, "Table 11 BERT-LSTM-CRF".upper(), \'#\'*20)\n!python3 train.py False BertLSTMCRF rsq test\nprint(\'\\n\\n\', \'#\'*20, "Table 11 BERT-CNN".upper(), \'#\'*20)\n!python3 train.py False BertCNN rsq test\nprint(\'\\n\\n\', \'#\'*20, "Table 11 Multi-Depth ".upper(), \'#\'*20)\n!python3 train.py False MultiDepthNodeEdgeDetector rsq test\nprint(\'\\n\\n\', \'#\'*20, "Table 11 Fine_tune BERT".upper(), \'#\'*20)\n!python3 train.py False NodeEdgeDetector rsq test\nwith open(\'/root/capsule/results/output.txt\', \'a\') as f:\n    f.write(cap.stdout)\n')


# In[3]:


# Model for creating candidates
# %cd /root/capsule/code/src
# !mkdir ../../data/Models
# !python3 train.py True NodeEdgeDetector rsq test


# In[ ]:


### Reproducing Candidates
# %cd /root/capsule/code/src
# !python3 evaluation.py tfidf test

# !python3 evaluation.py tfidf valid


# In[1]:


get_ipython().run_cell_magic('capture', 'cap --no-stderr', '# table 12 output\n\n%cd /root/capsule/code\nprint(\'\\n\\n\', \'#\'*20, "Table 12 (Test, Valid)".upper(), \'#\'*20)\n\nfrom src.utils import get_hit\nimport pandas as pd\ntest_df = pd.read_excel(\'../data/Intermediate/test.xlsx\')\nactual = test_df[\'Reverb_no\'].to_list()\nsystem_results = pd.read_excel(\'../data/Candidates/test_results.xlsx\')[\'sys\'].apply(lambda item:eval(item))\nprint(get_hit(actual, system_results))\ntest_df = pd.read_excel(\'../data/Intermediate/valid.xlsx\')\nactual = test_df[\'Reverb_no\'].to_list()\nsystem_results = pd.read_excel(\'../data/Candidates/valid_results.xlsx\')[\'sys\'].apply(lambda item:eval(item))\nprint(get_hit(actual, system_results))\n\nwith open(\'/root/capsule/results/output.txt\', \'a\') as f:\n    f.write(cap.stdout)\n')


# In[3]:


get_ipython().run_cell_magic('capture', 'cap --no-stderr', '# table 13 output\n\n%cd /root/capsule/code\nprint(\'\\n\\n\', \'#\'*20, "Table 13 (Test, Valid)".upper(), \'#\'*20)\n\nimport pandas as pd\nfrom collections import Counter\ntest = pd.read_excel(\'../data/Candidates/test_results.xlsx\', engine =\'openpyxl\')\nvalid = pd.read_excel(\'../data/Candidates/valid_results.xlsx\', engine =\'openpyxl\')\nrr = lambda row: 1/10000 if int(row[\'Reverb_no\']) not in [item[0] for item in eval(row[\'sys\'])] else 1/(1+[item[0] for item in eval(row[\'sys\'])].index(row[\'Reverb_no\']))\ntest[\'RR\'] = test.apply(rr, axis=1)\nvalid[\'RR\'] = valid.apply(rr, axis=1)\nprint(test[\'RR\'].mean(), valid[\'RR\'].mean())\nwith open(\'/root/capsule/results/output.txt\', \'a\') as f:\n    f.write(cap.stdout)\n')


# In[21]:


get_ipython().run_cell_magic('capture', 'cap --no-stderr', 'print(\'\\n\\n\', \'#\'*20, "Table 14 (Relation, Entity)".upper(), \'#\'*20)\n\ndef claculate(df):\n    node_precision, edge_precision = [], []\n    for index, row in df.iterrows():\n        temp = max([fuzz.ratio(item, row[\'node\']) for item in row[\'triple\']])\n        node_precision.append(temp) \n        temp = max([fuzz.ratio(item, row[\'edge\']) for item in row[\'triple\']])\n        edge_precision.append(temp) \n    print(sum(node_precision)/len(node_precision))\n    print(sum(edge_precision)/len(edge_precision))\n\n\nfrom fuzzywuzzy import fuzz\nimport pandas as pd\nreference = pd.read_excel(\'../data/Intermediate/test.xlsx\')\nreference.Question = reference.Question.apply(lambda x:str(x).lower().strip())\ntest = pd.read_excel(\'../data/Candidates/test_results.xlsx\', engine =\'openpyxl\')\nanalysis = pd.merge(test, reference, how=\'inner\', on=\'Question\')\nanalysis[\'candidates_index\'] = analysis[\'sys\'].apply(lambda item:[i[0] for i in eval(item)])\ngh = lambda row:row[\'candidates_index\'].index(row[\'Reverb_no_x\']) if (row[\'Reverb_no_x\'] in row[\'candidates_index\']) else 100\nanalysis[\'hit\'] = analysis.apply(gh, axis=1)\nanalysis = analysis[[\'Question\', \'node\', \'edge\', \'normalized_triple_y\', \'Reverb_no_x\', \'hit\']]\nanalysis[\'triple\'] = analysis.normalized_triple_y.apply(lambda x:list(str(item).lower() for item in eval(x)))\n\nprint(\'Hit@1 Similarity (Test)\')\nselected = analysis[(analysis[\'hit\']<1)]\nclaculate(selected)\nprint(\'Hit@3 Similarity (Test)\')\nselected = analysis[(analysis[\'hit\']<3)]\nclaculate(selected)\nprint(\'Hit@5 Similarity (Test)\')\nselected = analysis[(analysis[\'hit\']<5)]\nclaculate(selected)\nprint(\'Hit@10 Similarity (Test)\')\nselected = analysis[(analysis[\'hit\']<10)]\nclaculate(selected)\nprint(\'Hit@100 Similarity (Test)\')\nselected = analysis[(analysis[\'hit\']<100)]\nclaculate(selected)\n\nreference = pd.read_excel(\'../data/Intermediate/valid.xlsx\')\nreference.Question = reference.Question.apply(lambda x:str(x).lower().strip())\ntest = pd.read_excel(\'../data/Candidates/valid_results.xlsx\', engine =\'openpyxl\')\nanalysis = pd.merge(test, reference, how=\'inner\', on=\'Question\')\nanalysis[\'candidates_index\'] = analysis[\'sys\'].apply(lambda item:[i[0] for i in eval(item)])\ngh = lambda row:row[\'candidates_index\'].index(row[\'Reverb_no_x\']) if (row[\'Reverb_no_x\'] in row[\'candidates_index\']) else 100\nanalysis[\'hit\'] = analysis.apply(gh, axis=1)\nanalysis = analysis[[\'Question\', \'node\', \'edge\', \'normalized_triple_y\', \'Reverb_no_x\', \'hit\']]\nanalysis[\'triple\'] = analysis.normalized_triple_y.apply(lambda x:list(str(item).lower() for item in eval(x)))\nprint()\nprint(\'Hit@1 Similarity (Valid)\')\nselected = analysis[(analysis[\'hit\']<1)]\nclaculate(selected)\nprint(\'Hit@3 Similarity (Valid)\')\nselected = analysis[(analysis[\'hit\']<3)]\nclaculate(selected)\nprint(\'Hit@5 Similarity (Valid)\')\nselected = analysis[(analysis[\'hit\']<5)]\nclaculate(selected)\nprint(\'Hit@10 Similarity (Valid)\')\nselected = analysis[(analysis[\'hit\']<10)]\nclaculate(selected)\nprint(\'Hit@100 Similarity (Valid)\')\nselected = analysis[(analysis[\'hit\']<100)]\nclaculate(selected)\n\nwith open(\'/root/capsule/results/output.txt\', \'a\') as f:\n    f.write(cap.stdout)\n')


# In[ ]:


get_ipython().system('pip uninstall -y numpy -q')
get_ipython().system('pip install numpy==1.23 -q')
get_ipython().system('pip install python-terrier -q')
get_ipython().system('pip install --upgrade git+https://github.com/terrierteam/pyterrier_colbert.git -q')
get_ipython().system('pip install faiss-gpu==1.6.3 -q')
import faiss
assert faiss.get_num_gpus() > 0


# In[ ]:


import pyterrier as pt
pt.init()
checkpoint="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
get_ipython().system('rm -rf /content/colbertindex')
import pyterrier_colbert.indexing
indexer = pyterrier_colbert.indexing.ColBERTIndexer(checkpoint, "/content", "colbertindex", chunksize=3)

