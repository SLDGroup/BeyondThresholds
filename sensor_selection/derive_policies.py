import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
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
	

def get_conf_dist(sensor_confidence_scores, sensor_predictions, ens_predictions, labels, n_bins,
				  sensor_conf_dist, sensor_rel, ens_cond_rel, policy):

	# create uniform bins for the confidence distribution (equal number of preds per bin)
	sensor_confidence_scores_sorted_idxs = np.argsort(sensor_confidence_scores)
	sensor_ordered_confidence_scores = sensor_confidence_scores[sensor_confidence_scores_sorted_idxs]
	bin_amt = len(sensor_ordered_confidence_scores) // n_bins
	bin_boundaries = sensor_ordered_confidence_scores[::bin_amt]
	# edge case when split is exact, need to add the end point
	if len(bin_boundaries) < (n_bins+1):
		bin_boundaries = torch.cat([bin_boundaries,torch.tensor([1.0])])
	bin_boundaries[0] = 0.0
	bin_boundaries[-1] = 1.0

	bin_lowers = bin_boundaries[:-1]
	bin_uppers = bin_boundaries[1:]
	bin_marks = torch.cat([bin_lowers,bin_uppers[-1].unsqueeze(0)])

	for bin_idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
		# mask to determine which confidence scores go in this bin
		in_bin = sensor_confidence_scores.gt(bin_lower.item()) * sensor_confidence_scores.le(bin_upper.item())

		# from the mask determine what percent of the confidence scores are in this bin
		prop_in_bin = in_bin.float().mean()

		# what percent of predictions are in this bin
		sensor_conf_dist[bin_idx] = len(sensor_confidence_scores[in_bin])/len(sensor_confidence_scores)

		# if the bin is not empty, get per bin accuracy for reliability diagrams
		if prop_in_bin.item() > 0:
			ens_results = (ens_predictions==labels)[in_bin]
			if policy == 'optimal':
				sensor_rel[bin_idx] = (sensor_predictions==labels)[in_bin].float().mean()
				ens_cond_rel[bin_idx] = ens_results.float().mean() # prob of ens being correct given pred in this sensor bin
			else:
				agreement = (sensor_predictions==ens_predictions)[in_bin].float().mean() # how much they agree
				sensor_rel[bin_idx] = agreement

				if policy == 'optimal_1':
					ens_cond_rel[bin_idx] = ens_results[:len(in_bin)//10].float().mean() # 1/10th of 10% is 1%
				elif policy == 'optimal_5':
					ens_cond_rel[bin_idx] = ens_results[:len(in_bin)//2].float().mean() # 1/2 of 10% is 5%
				elif policy == 'optimal_U':
					ens_cond_rel[bin_idx] = 1 # we assume ensemble always right
				
				sensor_rel[bin_idx] = agreement*ens_cond_rel[bin_idx]
	
	return sensor_conf_dist, bin_lowers, bin_uppers, bin_marks, sensor_rel, ens_cond_rel


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Sensor Selection')
	parser.add_argument('--dataset', type=str, help='which dataset to use')
	parser.add_argument('--seed', type=int, help='which random seed to use for policy evaluation')
	parser.add_argument('--policy', type=str, help='which policy to run (random,threshold,optimal,optimal_U,optimal_1,optimal_5)')
	
	args = parser.parse_args()

	# ================================= constants =================================
 
	cache_path = os.path.join(PROJECT_ROOT,f"saved_data/cached_data/{args.dataset}")

	logging_prefix = args.dataset

	n_bins = 10
	bw = 1/n_bins

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

	seed = args.seed
	init_seeds(seed)
	logger = init_logger(f"{logging_prefix}/sensel_seed{seed}")
	logger.info(f"Seed: {seed}")

	for user_i,user in enumerate(users):
		logger.info(f"\tUser: {user}")
		train_users = users[:user_i] + users[user_i+1:]
		test_users = [user]

		# ================================= load cached data =================================

		ens_confidence_values, ens_predictions, ens_labels, rand_order = load_cached_data(logging_prefix, train_users, seed, True)

		# data and table for each sensor
		sensor_conf_vals = {i:None for i in range(len(body_parts))}
		sensor_preds = {i:None for i in range(len(body_parts))}
		sensor_labels = {i:None for i in range(len(body_parts))}

		sensor_bin_boundaries = {i: torch.linspace(0, 1, n_bins+1) for i in range(len(body_parts))}
		sensor_bin_lowers = {i: sensor_bin_boundaries[i][:-1] for i in range(len(body_parts))}
		sensor_bin_uppers = {i: sensor_bin_boundaries[i][1:] for i in range(len(body_parts))}
		sensor_bin_marks = {i: torch.cat([sensor_bin_lowers[i],sensor_bin_uppers[i][-1].unsqueeze(0)]) for i in range(len(body_parts))}

		sensor_rel_diagrams = {i: np.zeros(n_bins) for i in range(len(body_parts))} # acc per bin
		sensor_conf_dists = {i: np.zeros(n_bins) for i in range(len(body_parts))} # conf dist
		ens_cond_rel_diagrams = {i: np.zeros(n_bins) for i in range(len(body_parts))} # acc of ens per sensor bin

		sensor_policy = {i: np.zeros(n_bins) for i in range(len(body_parts))}
		if args.policy == 'threshold':
			sensor_policy = {i: 0.0 for i in range(len(body_parts))}

		# now load the data from each sensor
		for bp_i,bp in enumerate(body_parts):
			logger.info(f"\t\tLoading Body Part: {bp}")
			
			# load cached sensor data
			bp_confidence_values, bp_predictions, bp_labels = load_cached_data(logging_prefix, train_users, seed, False, rand_order, bp)
			sensor_conf_vals[bp_i] = bp_confidence_values
			sensor_preds[bp_i] = bp_predictions
			sensor_labels[bp_i] = bp_labels

		# ================================= get sensor conf dists and rel diagrams =================================
		
		# use first 10% for getting confidence distribution
		calib_amt = int(0.1*len(rand_order))

		# ensure each sensor used an even number of times
		num_cycles = (len(rand_order)-calib_amt) // len(body_parts)

		# give extra to calib
		calib_amt = len(rand_order)-num_cycles*len(body_parts)

		# collect statistics for each sensor
		for sensor_i, sensor in enumerate(body_parts):
			sensor_conf_dists[sensor_i], sensor_bin_lowers[sensor_i], sensor_bin_uppers[sensor_i], \
			sensor_bin_marks[sensor_i], sensor_rel_diagrams[sensor_i], ens_cond_rel_diagrams[sensor_i] = \
			get_conf_dist(sensor_conf_vals[sensor_i][:calib_amt], sensor_preds[sensor_i][:calib_amt],
							ens_predictions[:calib_amt], ens_labels[:calib_amt], n_bins,
							sensor_conf_dists[sensor_i], sensor_rel_diagrams[sensor_i], ens_cond_rel_diagrams[sensor_i],
							args.policy)
			fn = f"{args.dataset}_{args.seed}_user{user}_sensor{sensor_i}_bin_marks"
			fn = os.path.join(cache_path,"bin_marks",fn)
			Path(os.path.dirname(fn)).mkdir(parents=True, exist_ok=True)
			np.save(fn, sensor_bin_marks[sensor_i])

		# ================================= derive policies =================================
		for ui,(ub,ef) in enumerate(zip(utilization_budgets,ensemble_fractions)):
			logger.info(f"\t\t\tutilization budget: {ub}, {ef}")

			# each sensor has a different policy
			for sensor_i, sensor in enumerate(body_parts):
				bin_budget = int(ef/(1/n_bins))
				max_exp_acc = 0.0
				best_policy = 0.0
				s_confs_sorted_idxs = np.argsort(sensor_conf_vals[sensor_i][:calib_amt])
				ordered_confs = sensor_conf_vals[sensor_i][:calib_amt][s_confs_sorted_idxs]
				ordered_s_preds = sensor_preds[sensor_i][:calib_amt][s_confs_sorted_idxs]
				ordered_ens_preds = ens_predictions[:calib_amt][s_confs_sorted_idxs]
				ordered_labels = ens_labels[:calib_amt][s_confs_sorted_idxs]

				if args.policy == 'threshold':
					# for a threshold policy, we can only allocate a contiguous segment to the ensemble
					# so the max threshold is the bin mark of the specified bin budget
					max_thresh = sensor_bin_marks[sensor_i][bin_budget]
					search_thresh = np.linspace(0,max_thresh,50)
					for th in search_thresh:
						try:
							# for every threshold, we allocate the predictions with confidence scores that are lower
							# than the threshold to the ensemble and the rest to the sensor
							# given this segmentation we estimate the expected accuracy
							alpha_e = (ordered_confs<=th).float().mean()
							thresh_idx = (ordered_confs<=th).nonzero().view(-1)[-1]
							s_contrib = (ordered_s_preds[thresh_idx:] == ordered_labels[thresh_idx:]).float().mean()
							ens_contrib = (ordered_ens_preds[:thresh_idx] == ordered_labels[:thresh_idx]).float().mean()
							exp_acc_ = s_contrib*(1-alpha_e)+ens_contrib*alpha_e
							if exp_acc_ > max_exp_acc:
								max_exp_acc = exp_acc_
								best_policy = th
						except:
							continue
						
					sensor_policy[sensor_i] = best_policy
				else:
					# get the difference in the reliability diagrams
					diff = ens_cond_rel_diagrams[sensor_i] - sensor_rel_diagrams[sensor_i]
					diff_s_order = np.argsort(diff)[::-1]
					diff_s = diff[diff_s_order] # largest to smallest
					opt_count = 0
					# allocate the top positive elements to the ensemble
					for ds,ds_o in zip(diff_s,diff_s_order):
						if opt_count < bin_budget and ds > 0:
							sensor_policy[sensor_i][ds_o] = 1
							opt_count += 1

				logger.info(f"\t\t\t{sensor}: best policy ({args.policy}): {sensor_policy[sensor_i]}")

				fn = f"{args.dataset}_{args.seed}_{args.policy}_user{user}_budget{ub}_{ef}_sensor{sensor_i}"
				fn = os.path.join(cache_path,"policy",fn)
				Path(os.path.dirname(fn)).mkdir(parents=True, exist_ok=True)
				np.save(fn, sensor_policy[sensor_i])


			