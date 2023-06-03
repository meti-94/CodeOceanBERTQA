print('\n\n', '#'*20, "Table 14 (Relation, Entity)".upper(), '#'*20)

def claculate(df):
    node_precision, edge_precision = [], []
    for index, row in df.iterrows():
        temp = max([fuzz.ratio(item, row['node']) for item in row['triple']])
        node_precision.append(temp) 
        temp = max([fuzz.ratio(item, row['edge']) for item in row['triple']])
        edge_precision.append(temp) 
    print(sum(node_precision)/len(node_precision))
    print(sum(edge_precision)/len(edge_precision))


from fuzzywuzzy import fuzz
import pandas as pd
reference = pd.read_excel('../data/Intermediate/test.xlsx')
reference.Question = reference.Question.apply(lambda x:str(x).lower().strip())
test = pd.read_excel('../data/Candidates/test_results.xlsx', engine ='openpyxl')
analysis = pd.merge(test, reference, how='inner', on='Question')
analysis['candidates_index'] = analysis['sys'].apply(lambda item:[i[0] for i in eval(item)])
gh = lambda row:row['candidates_index'].index(row['Reverb_no_x']) if (row['Reverb_no_x'] in row['candidates_index']) else 100
analysis['hit'] = analysis.apply(gh, axis=1)
analysis = analysis[['Question', 'node', 'edge', 'normalized_triple_y', 'Reverb_no_x', 'hit']]
analysis['triple'] = analysis.normalized_triple_y.apply(lambda x:list(str(item).lower() for item in eval(x)))

print('Hit@1 Similarity (Test)')
selected = analysis[(analysis['hit']<1)]
claculate(selected)
print('Hit@3 Similarity (Test)')
selected = analysis[(analysis['hit']<3)]
claculate(selected)
print('Hit@5 Similarity (Test)')
selected = analysis[(analysis['hit']<5)]
claculate(selected)
print('Hit@10 Similarity (Test)')
selected = analysis[(analysis['hit']<10)]
claculate(selected)
print('Hit@100 Similarity (Test)')
selected = analysis[(analysis['hit']<100)]
claculate(selected)

reference = pd.read_excel('../data/Intermediate/valid.xlsx')
reference.Question = reference.Question.apply(lambda x:str(x).lower().strip())
test = pd.read_excel('../data/Candidates/valid_results.xlsx', engine ='openpyxl')
analysis = pd.merge(test, reference, how='inner', on='Question')
analysis['candidates_index'] = analysis['sys'].apply(lambda item:[i[0] for i in eval(item)])
gh = lambda row:row['candidates_index'].index(row['Reverb_no_x']) if (row['Reverb_no_x'] in row['candidates_index']) else 100
analysis['hit'] = analysis.apply(gh, axis=1)
analysis = analysis[['Question', 'node', 'edge', 'normalized_triple_y', 'Reverb_no_x', 'hit']]
analysis['triple'] = analysis.normalized_triple_y.apply(lambda x:list(str(item).lower() for item in eval(x)))
print()
print('Hit@1 Similarity (Valid)')
selected = analysis[(analysis['hit']<1)]
claculate(selected)
print('Hit@3 Similarity (Valid)')
selected = analysis[(analysis['hit']<3)]
claculate(selected)
print('Hit@5 Similarity (Valid)')
selected = analysis[(analysis['hit']<5)]
claculate(selected)
print('Hit@10 Similarity (Valid)')
selected = analysis[(analysis['hit']<10)]
claculate(selected)
print('Hit@100 Similarity (Valid)')
selected = analysis[(analysis['hit']<100)]
claculate(selected)