print('\n\n', '#'*20, "Table 12 (Test, Valid)".upper(), '#'*20)

from src.utils import get_hit
import pandas as pd
test_df = pd.read_excel('../data/Intermediate/test.xlsx')
actual = test_df['Reverb_no'].to_list()
system_results = pd.read_excel('../data/Candidates/test_results.xlsx')['sys'].apply(lambda item:eval(item))
print(get_hit(actual, system_results))
test_df = pd.read_excel('../data/Intermediate/valid.xlsx')
actual = test_df['Reverb_no'].to_list()
system_results = pd.read_excel('../data/Candidates/valid_results.xlsx')['sys'].apply(lambda item:eval(item))
print(get_hit(actual, system_results))