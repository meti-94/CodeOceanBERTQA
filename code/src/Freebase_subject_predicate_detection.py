from train import *
from graph import *
import pandas as pd
from utils import get_hit
'''
	A simple script to evaluate model on UNSEEN data
'''
if __name__=='__main__':
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	bert = BertModel.from_pretrained("bert-base-uncased")
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	node_edge_detector = NodeEdgeDetector(bert, tokenizer, dropout=torch.tensor(0.5))
	optimizer = AdamW
	kw = {'lr':0.0002, 'weight_decay':0.1}
	tl = TrainingLoop(node_edge_detector, optimizer, True, **kw)
	loss = mse_loss
	tl.load()

	# RKBG = ReverbKnowledgeBase()
	wordstoberttokens_array, berttokenstoids_array, input_token_ids_array, nodes_borders_array, edges_spans_array, node_array, edge_array = [], [], [], [], [], [], []
	questions_array = []
	test_df = pd.read_excel('test.xlsx')
	# actual = test_df['Reverb_no'].to_list()
	# system_results, candidates_array, actual_answer_array = [], [], []
	for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
		wordstoberttokens, berttokenstoids, input_token_ids, nodes_borders, edges_spans, node, edge = tl.readable_predict_article(
                                                device, _input=row['Question'], print_result=False)
		wordstoberttokens_array.append(wordstoberttokens)
		berttokenstoids_array.append(berttokenstoids)
		input_token_ids_array.append(input_token_ids)
		nodes_borders_array.append(nodes_borders)
		edges_spans_array.append(edges_spans)
    
		node = ' '.join(node); edge = ' '.join(edge)
		node = node.replace(' ##', ''); edge = edge.replace(' ##', '')

		node_array.append(node)
		edge_array.append(edge)
		questions_array.append(row['Question'].lower().split())
		# temp = RKBG.tfidf_query(node=node, edge=edge)
		# candidates_array.append(temp[:min(len(temp), 25)])
		# actual_answer_array.append(row['Reverb_no'])
	output_data = {
    # 'bert_tokenizer_output':wordstoberttokens_array,
    # 'bert_token_ids':berttokenstoids_array,
    # 'input_token_ids':input_token_ids_array,
    # 'nodes_borders':nodes_borders_array,
    # 'edges_spans':edges_spans_array,
    'node':node_array,
    'edges':edge_array,
    # 'questions':questions_array,
    # 'candidates':candidates_array, 
    # 'actual_answer':actual_answer_array
  	}
	pd.DataFrame(output_data).to_excel('Freebase_subject_predicate.xlsx')
		

		
