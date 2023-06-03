import os
from data import *
# check if Bertified data exists
if not os.path.isfile('../data/Bertified/entities.npy'):
    reverb_lines = read_reverb('../data/Reverb1.1/reverb_wikipedia_tuples-1.1.txt')
    questions = pd.read_excel('../data/ReverbSQA/Final_Sheet_990824.xlsx', sheet_name=1, engine='openpyxl')
    index = get_tuple_frequency(reverb_lines, questions)
    index[index['Frequency']<10].to_excel('../data/ProcessedQuestions/normalized_questions.xlsx')
    combine_with_reverb(questions_path='../data/ProcessedQuestions/normalized_questions.xlsx', 
                    reverb_path='../data/Reverb1.1/reverb_wikipedia_tuples-1.1.txt')
    create_bertified_dataset()
print('\n\n', '#'*20, "Table 1".upper(), '#'*20)
print(pd.read_excel('../data/Intermediate/train.xlsx').sample(6)[['triple', 'Question']])