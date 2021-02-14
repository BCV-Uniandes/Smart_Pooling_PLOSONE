import os
import h2o 
import json
import glob
import argparse
import itertools
import numpy as np
import pandas as pd
import os.path as osp
from h2o.automl import H2OAutoML 

# Custom Imports
from create_descriptors import create_df

from utils.misc import (save_model, change_dates_to_ints,
						generate_train_test, mkdirs)

from utils.get_features_gapallgeo import get_features
from utils.train_utils import train_models, retrain_best_model
from utils.eval_utils import get_from_val,evaluate_thr_pool

def main(args, prev):

	args.test_prevalence = prev 
	filename = args.output_dir
	splits=2

	print('+++++++++++++++++++++++++PROCESSING DATA++++++++++++++++++++++++')

	df,prev = create_df(args)
	
	df = change_dates_to_ints(df)

	train, test = generate_train_test(df,args.filterdate)
	train, val, all_train, test = get_features(train, test)

	print('Uploading data to h2o...')
	train = h2o.H2OFrame(train, column_names=list(train.columns.astype(str)))
	val = h2o.H2OFrame(val, column_names=list(val.columns.astype(str)))
	all_train = h2o.H2OFrame(all_train, column_names=list(all_train.columns.astype(str)))
	test = h2o.H2OFrame(test, column_names=list(test.columns.astype(str)))
	mkdirs(args.output_dir)
	if args.eval:
		best_model = h2o.load_model(args.path_to_best)
		val_model = h2o.load_model(args.path_to_val)

		print('+++++++++++++++++++++++++EVALUATION++++++++++++++++++++++++')

		threshold,poolsize = get_from_val(val_model,val,df)
		prevalence, smart_pooling, dorfman = evaluate_thr_pool(best_model, test,df,threshold,args.poolsize)

		return prevalence, smart_pooling, dorfman
	else:
		# Create output dirs

		
		experiment_name = args.output_dir.replace('/','_')+'_experiment'
		model_train_path = f'{args.output_dir}/train_model'
		model_all_train_path = f'{args.output_dir}/all_train_model'

		print('+++++++++++++++++++++++++TRAINING++++++++++++++++++++++++')
		models = train_models(train, val, experiment_name)

		print('+++++++++++++++++++++++++LEADERBOARD++++++++++++++++++++++++')
		lb = models.leaderboard
		print(lb.as_data_frame())
		save_model(models.leader, model_train_path)

		print('+++++++++++++++++++++++++RETRAINING++++++++++++++++++++++++')

		best_model = retrain_best_model(all_train, models.leader, save_model=True,
										save_path=model_all_train_path)


