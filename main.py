#import logging
import json
import torch
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from data import create_dataloader_folds, create_dataloader


#logging.basicConfig(level=logging.INFO)

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)



def evaluate(losses, truth):	
	langs = list(losses.keys())
	folds = list(losses[langs[0]].keys())
	epochs = list(losses[langs[0]][folds[0]].keys())
	errors_fold_epoch={fold:{epoch:{lang:[] for lang in langs} for epoch in epochs} for fold in folds}

	for lang in losses:
		for fold in losses[lang]:
			for epoch in losses[lang][fold]:
				errors_fold_epoch[fold][epoch][lang]=losses[lang][fold][epoch]

	results = {fold:{} for fold in folds}
	confusion = {fold:{} for fold in folds}

	for fold in folds:
		tr = truth[fold]
		for epoch in epochs:
			df = pd.DataFrame(errors_fold_epoch[fold][epoch])
			label = df[langs].idxmin(axis=1)
			acc = (tr==label).sum()/len(tr)
			f1 = f1_score(tr,label, average='macro')
			results[fold][epoch]=[acc,f1]
			confusion[fold][epoch] = {'pr': label.values.tolist(), 'tr':tr}

	average = {f'epoch_{epoch}':{'acc':{'mean':np.mean([results[i][epoch][0] for i in folds]), 'std':np.std([results[i][epoch][0] for i in folds])}, 
					  'f1':{'mean':np.mean([results[i][epoch][1] for i in folds]) , 'std':np.std([results[i][epoch][0] for i in folds])}} for epoch in epochs}
	results.update({'average':average})
	
	return results, confusion





def train_test(args, lang, folds):
	set_seed(args)
	device = args.device
	loss_folds = {}

	for n,fold in enumerate(folds):
		train_loader, test_loader = fold
		t_total = len(train_loader) // args.accumulate_grad * args.epoch
		model = AutoModelForCausalLM.from_pretrained(args.model_name)
		opt = AdamW(model.parameters(), lr = args.lr)
		scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
		model.to(device)
		model.train()
		opt.zero_grad()
		eval_loss = {}

		for epoch in range(1, args.epoch+1):
			print(f'{lang} model, Fold {n+1} of {len(folds)}, Epoch: {epoch} of {args.epoch}')
			# training loop
			for i, batch in enumerate(tqdm(train_loader)):
				input_ids, labels = batch[0].to(device), batch[1].to(device)
				loss = model(input_ids, labels =labels).loss / args.accumulate_grad
				loss.backward()
				if (i+1) % args.accumulate_grad == 0:
					torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
					opt.step()
					scheduler.step()
					opt.zero_grad()

			# evaluation loop
			if epoch in range(args.epoch_eval, args.epoch + 1):
				print('Evaluating ...')
				model.eval()
				loss_epoch = []
				for i, batch in enumerate(tqdm(test_loader)):
					with torch.no_grad():
						input_ids, labels = batch[0].to(device), batch[1].to(device)
						loss = model(input_ids, labels =labels).loss
						loss_epoch.append(loss.cpu().item())
				eval_loss[epoch] = loss_epoch
		loss_folds[n]=eval_loss
				
	return loss_folds

				
				
def main(args):
	set_seed(args)
	tokenizer = AutoTokenizer.from_pretrained(args.model_name)
	tokenizer.pad_token = '<pad>'
	
	# This assumes files are in the .csv format with columns: 'essay' and 'label' 
	train_data = pd.read_csv(args.train_data_path)
	test_data = pd.read_csv(args.test_data_path) if args.test_data_path != '' else None
	langs = train_data['label'].unique()

	if args.eval_setting == 'cv':
		data = pd.concat([train_data, test_data]) if test_data else train_data
		texts, labels = data[['essay']], data[['label']]
		skf = StratifiedKFold(n_splits = args.num_folds)
		splits= skf.split(texts, labels)

	outputs = {}
	for lang in langs:
		if args.eval_setting == 'cv':
			folds, test_labels = create_dataloader_folds(args, tokenizer, lang, splits, texts, labels) 
		else:
			assert test_data is not None , "Test data cannot be empty in the train-test setting." 
			folds, test_labels = create_dataloader(args, tokenizer, lang, train_data, test_data)
		
		outputs[lang] = train_test(args, lang, folds)
	
	truth_folds = {i:test_labels[i].tolist() for i in range(len(test_labels))}
	
	json.dump(outputs, open('results/losses.json','w'))
	json.dump(truth_folds, open('results/truth.json','w'))

	eval_results, confusion_results = evaluate(outputs, truth_folds)


	json.dump(eval_results, open(args.eval_path,'w'))
	json.dump(confusion_results, open(args.conf_path,'w'))
	

				
		



if __name__ == "__main__": 
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', type=str, default= 'gpt2-medium')  # or gpt2
	parser.add_argument('--train_data_path', type=str, default= 'data/TOEFL-train.csv')
	parser.add_argument('--test_data_path', type=str, default= 'data/TOEFL-test.csv')
	parser.add_argument('--eval_path', type=str, default= 'results/eval.json')
	parser.add_argument('--conf_path', type=str, default= 'results/confusion.json')

	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--block_size', type=int, default=512)
	parser.add_argument('--train_batch_size', default= 1, type=int) # adjust for gpt2
	parser.add_argument('--accumulate_grad', type=int, default=16) # adjust for gpt2
	parser.add_argument('--epoch', type=int, default=3)
	parser.add_argument('--epoch_eval', type=int, default=2)  # epoch to start evaluating the model
	parser.add_argument('--max_grad_norm', type=float, default=1.)
	parser.add_argument('--lr', type=float, default=1.5e-4) 
	parser.add_argument('--warmup_steps', type=int, default=0)

	parser.add_argument('--eval_setting', choices=['cv', 'train-test'], default= 'train-test')
	parser.add_argument('--num_folds', type=int, default=10)
	parser.add_argument('--device', default= torch.device('cuda'))
	args = parser.parse_args()
	main(args)


	
	
	