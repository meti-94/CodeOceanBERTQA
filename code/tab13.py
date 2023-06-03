print('\n\n', '#'*20, "Table 13 (Test, Valid)".upper(), '#'*20)

import pandas as pd
from collections import Counter
test = pd.read_excel('../data/Candidates/test_results.xlsx', engine ='openpyxl')
valid = pd.read_excel('../data/Candidates/valid_results.xlsx', engine ='openpyxl')
rr = lambda row: 1/10000 if int(row['Reverb_no']) not in [item[0] for item in eval(row['sys'])] else 1/(1+[item[0] for item in eval(row['sys'])].index(row['Reverb_no']))
test['RR'] = test.apply(rr, axis=1)
valid['RR'] = valid.apply(rr, axis=1)
print(test['RR'].mean(), valid['RR'].mean())