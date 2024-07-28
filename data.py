import torch
import logging
import math
from torch.utils.data import DataLoader, TensorDataset

#logger = logging.getLogger(__file__)
#logging.basicConfig(level=logging.INFO) 






def create_dataloader_train(text_tokens, pad_token_id, args): 
	chunks, lm_labels = [], []
	input_length, block_size = len(text_tokens) , args.block_size
	
	# padding and chunking to block_size
	added_pad_tokens = block_size - (input_length % block_size) if (input_length % block_size != 0) else 0
	padded_input = text_tokens + added_pad_tokens * [pad_token_id]

	for i in range(0, len(padded_input), block_size):
		tokens = padded_input[i: i + block_size]
		labels = [(t if t != pad_token_id else -100) for t in tokens]
		chunks.append(tokens)
		lm_labels.append(labels)

	chunks, lm_labels = torch.tensor(chunks), torch.tensor(lm_labels)
	dataloader = DataLoader(TensorDataset(chunks, lm_labels), args.train_batch_size)
	return dataloader
	



def create_dataloader_test(data, tokenizer, args): 
	samples, lm_labels = [], []
	tokenized_samples = data.essay.apply(lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))).values
	
	# To make the calculations easier, we process samples one-by-one, but if they are longer than the block_size, we pad-and-chunk them and put all chuncks in one batch 
	# This can be made into batched processing with simple modification in loss function (e.g. changing the reduction method)  
	max_sample_length = max([len(s) for s in tokenized_samples])
	batch_block = (math.ceil(max_sample_length/args.block_size))*args.block_size
	batch_size = batch_block//args.block_size

	for test_item in tokenized_samples:
		padded_sample = test_item + (batch_block - len(test_item)) * [tokenizer.pad_token_id]
		for i in range(0, batch_block, args.block_size):
			sample_chunk = padded_sample[i : i + args.block_size]
			labels = [(t if t != tokenizer.pad_token_id else -100) for t in sample_chunk]
			samples.append(sample_chunk)
			lm_labels.append(labels)
	
	sample_ids = torch.tensor(samples)
	lm_labels = torch.tensor(lm_labels)

	dataloader = DataLoader(TensorDataset(sample_ids, lm_labels), batch_size)
	return dataloader
	
	

	
# dataloaders for cross-validation setting	
def create_dataloader_folds(args, tokenizer, lang, splits, texts, labels):
	print(f'Creating Folds for {lang} ...')
	folds, labels = [], []
	for train_index, test_index in splits:
		train_texts, test_texts = texts.loc[train_index,:], texts.loc[test_index,:]
		train_labels, test_labels = labels.loc[train_index,:], labels.loc[test_index,:]
		train_texts = " ".join(train_texts[train_labels.label==lang].essay.values)
		tokenized_train_texts = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train_texts))

		train_loader =  create_dataloader_train(tokenized_train_texts, tokenizer.pad_token_id, args)
		test_loader = create_dataloader_test(test_texts, tokenizer, args)
		folds.append([train_loader, test_loader])
		labels.append(test_labels.label.values)
	return folds, labels



# dataloaders for train-test setting (as a 1-fold dataset)	
def create_dataloader(args, tokenizer, lang, train_data, test_data):
	print(f'Creating data for {lang} ...')
	train_texts = " ".join(train_data[train_data.label==lang].essay.values)
	tokenized_train_texts = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train_texts))

	train_loader =  create_dataloader_train(tokenized_train_texts, tokenizer.pad_token_id, args)
	test_loader = create_dataloader_test(test_data, tokenizer, args)
	test_labels = test_data.label.values

	return [[train_loader, test_loader]], [test_labels]
