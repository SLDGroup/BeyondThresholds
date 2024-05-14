import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

from utils.setup_funcs import *
from datasets.dsads.dsads import *
from training.train import *
from training.models import *


def load_cached_data(logging_prefix, train_users, seed, is_ens, rand_order=None, bp=None):
	if is_ens == True:
		# load ensemble data
		ens_logname = f"{logging_prefix}/ens_{train_users}_seed{seed}"
		ens_cache_path = os.path.join(PROJECT_ROOT,"saved_data/cached_data/"+ens_logname) + ".pt"
		ens_cached_data = torch.load(ens_cache_path)
		ens_confidence_values = ens_cached_data[0,:]
		ens_predictions = ens_cached_data[1,:]
		ens_labels = ens_cached_data[2,:]

		# shuffle the order of the ensemble data
		rand_order = torch.randperm(ens_labels.shape[0])
		ens_confidence_values = ens_confidence_values[rand_order]
		ens_predictions = ens_predictions[rand_order]
		ens_labels = ens_labels[rand_order]

		return ens_confidence_values, ens_predictions, ens_labels, rand_order
	else:
		# load sensor data
		bp_logname = f"{logging_prefix}/{bp}_{train_users}_seed{seed}"
		bp_cache_path = os.path.join(PROJECT_ROOT,"saved_data/cached_data/"+bp_logname) + ".pt"
		bp_cached_data = torch.load(bp_cache_path)
		bp_confidence_values = bp_cached_data[0,:]
		bp_predictions = bp_cached_data[1,:]
		bp_labels = bp_cached_data[2,:]

		# shuffle order of sensor data
		bp_confidence_values = bp_confidence_values[rand_order]
		bp_predictions = bp_predictions[rand_order]
		bp_labels = bp_labels[rand_order]

		return bp_confidence_values, bp_predictions, bp_labels


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Sensor Selection')
	parser.add_argument('--dataset', type=str, help='which dataset to use')
	parser.add_argument('--policy', type=str, help='which policy to run (random,threshold,optimal,optimal_U,optimal_1,optimal_5)')
	parser.add_argument('--seed', type=int, help='which random seed to use for policy evaluation')
	args = parser.parse_args()

	# ================================= constants =================================

	cache_path = os.path.join(PROJECT_ROOT,f"saved_data/cached_data/{args.dataset}")

	logging_prefix = args.dataset

	n_bins = 10
	bw = 1/n_bins

	seed = args.seed

	ensemble_fractions = np.arange(0,1+bw,bw)

	if args.dataset == 'dsads':
		body_parts = ['torso','right_arm','left_arm','right_leg','left_leg']
		users = [1,2,3,4,5,6,7,8]
	elif args.dataset == 'opportunity':
		body_parts = ["BACK","RUA","RLA","LUA","LLA","L-SHOE","R-SHOE"]
		users = [1,2,3,4]
	elif args.dataset == 'rwhar':
		body_parts = ['chest','forearm','head','shin','thigh','upperarm','waist']
		users = [1,3,4,5,7,8,9,10,11,12,13,14,15]

	k = len(body_parts)
	utilization_budgets = (ensemble_fractions*(k-1)+1)/k

	results_dict = {user:{ub:{'preds':[],'labels':[],'selection':[]} for ub in utilization_budgets} for user in users}
	f1_scores = np.zeros((len(users),len(utilization_budgets)))

	# iterate over all scenarios
	init_seeds(seed)
	logger = init_logger(f"{logging_prefix}/sensel_seed{seed}")
	logger.info(f"Seed: {seed}")

	for user_i,user in enumerate(users):
		logger.info(f"\tUser: {user}")
		train_users = users[:user_i] + users[user_i+1:]
		test_users = [user]

		# ================================= load cached data =================================

		ens_confidence_values, ens_predictions, ens_labels, rand_order = load_cached_data(logging_prefix, train_users, seed, True)
		
		# use first 10% for getting confidence distribution
		calib_amt = int(0.1*len(rand_order))

		# ensure each sensor used an even number of times
		num_cycles = (len(rand_order)-calib_amt) // len(body_parts)

		# give extra to calib
		calib_amt = len(rand_order)-num_cycles*len(body_parts)

		# data and table for each sensor
		sensor_conf_vals = {i:None for i in range(len(body_parts))}
		sensor_preds = {i:None for i in range(len(body_parts))}
		sensor_labels = {i:None for i in range(len(body_parts))}

		# now load the data from each sensor
		for bp_i,bp in enumerate(body_parts):
			logger.info(f"\t\tLoading Body Part: {bp}")
			
			# load cached sensor data
			bp_confidence_values, bp_predictions, bp_labels = load_cached_data(logging_prefix, train_users, seed, False, rand_order, bp)
			sensor_conf_vals[bp_i] = bp_confidence_values[calib_amt:]
			sensor_preds[bp_i] = bp_predictions[calib_amt:]
			sensor_labels[bp_i] = bp_labels[calib_amt:]

		ens_confidence_values = ens_confidence_values[calib_amt:]
		ens_predictions = ens_predictions[calib_amt:]
		ens_labels = ens_labels[calib_amt:]

		# ================================= execute policies =================================
		# shuffle the order the senors get queried
		sensor_sequence = np.arange(0, len(body_parts))
		sensor_order = np.array([sensor_sequence for cycle in range(num_cycles)]).flatten()
		
		for ui,(ub,ef) in enumerate(zip(utilization_budgets,ensemble_fractions)):
			results_dict[user][ub]['preds'] = np.zeros(len(ens_labels))
			results_dict[user][ub]['labels'] = np.zeros(len(ens_labels))
			sensor_util_counts = np.zeros(len(body_parts))

			logger.info(f"\t\t\tutilization budget: {ub}, {ef}")

			# load the policy and bin marks
			sensor_policy = {i: np.zeros(n_bins) for i in range(len(body_parts))}
			sensor_bin_marks = {i: None for i in range(len(body_parts))}

			if args.policy == 'threshold':
				sensor_policy = {i: 0.0 for i in range(len(body_parts))}
			for sensor_i, sensor in enumerate(body_parts):
				if args.policy != 'random':
					fn = f"{args.dataset}_{args.seed}_{args.policy}_user{user}_budget{ub}_{ef}_sensor{sensor_i}"
					fn = os.path.join(cache_path,"policy",fn)
					sensor_policy[sensor_i] = np.load(f"{fn}.npy")

				fn = f"{args.dataset}_{args.seed}_user{user}_sensor{sensor_i}_bin_marks"
				fn = os.path.join(cache_path,"bin_marks",fn)
				sensor_bin_marks[sensor_i] = np.load(f"{fn}.npy")
			
			for idx,sensor in enumerate(sensor_order):
				# get the current sensors predictions and the ensemble prediction
				ens_conf,ens_pred = ens_confidence_values[idx].item(), ens_predictions[idx].item()
				sensor_conf, sensor_pred = sensor_conf_vals[sensor][idx].item(), sensor_preds[sensor][idx].item()
				label = ens_labels[idx]

				# make sure we don't exceed utilization budget of any sensor
				exceeded = False
				next_util = (sensor_util_counts + 1)/(idx+1)
				if (next_util > ub).any():
					exceeded = True

				if args.policy == 'random':
					rand_val = torch.rand(1)
					# random policy
					if (rand_val > ef) or exceeded:
						rand_pred = sensor_pred
						rand_sel = sensor
						sensor_util_counts[sensor] += 1
					else:
						rand_pred = ens_pred
						rand_sel = -1
						sensor_util_counts += 1
					
					results_dict[user][ub]['preds'][idx] = rand_pred
					results_dict[user][ub]['labels'][idx] = label

				elif args.policy == 'threshold':
					conf_thresh = sensor_policy[sensor]

					if exceeded or sensor_conf >= conf_thresh:
						thresh_pred = sensor_pred
						thresh_sel = sensor	
						sensor_util_counts[sensor] += 1
					else:
						thresh_pred = ens_pred
						thresh_sel = -1
						sensor_util_counts += 1
					
					results_dict[user][ub]['preds'][idx] = thresh_pred
					results_dict[user][ub]['labels'][idx] = label

				elif 'optimal' in args.policy: # we need to load these bin marks
					
					bin_loc = (sensor_conf >= sensor_bin_marks[sensor]).nonzero()[0][-1]
					
					if exceeded or sensor_policy[sensor][bin_loc] != 1:
						opt_pred = sensor_pred
						opt_sel = sensor
						sensor_util_counts[sensor] += 1
					else:
						opt_pred = ens_pred
						opt_sel = -1
						sensor_util_counts += 1
					
					results_dict[user][ub]['preds'][idx] = opt_pred
					results_dict[user][ub]['labels'][idx] = label

			f1 = f1_score(results_dict[user][ub]['labels'],results_dict[user][ub]['preds'],average='macro')
			logger.info(f"\t\t\t\t{args.policy} f1: {f1}")

			f1_scores[user_i][ui] = f1

# save results
fn = os.path.join(cache_path,f"selection/{args.policy}_{args.seed}")
Path(os.path.dirname(fn)).mkdir(parents=True, exist_ok=True)
np.save(fn, f1_scores)