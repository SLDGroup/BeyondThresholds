import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils.setup_funcs import *
from datasets.dsads.dsads import *
from training.train import *
from training.models import *

# ================================= constants =================================

body_parts = ['torso','right_arm','left_arm','right_leg','left_leg']
users = [1,2,3,4,5,6,7,8]
seeds = [123,456,789]
logging_prefix = "dsads"

n_bins = 10
bw = 1/n_bins

ensemble_fractions = np.arange(0,1+bw,bw)
k = len(body_parts)
utilization_budgets = (ensemble_fractions*(k-1)+1)/k

policies = ['random','threshold','optimal','optimal_U','optimal_1','optimal_5']
results_dict = {seed:{user:{policy:{ub:{'preds':[],'labels':[],'selection':[]} for ub in utilization_budgets} for policy in policies} for user in users} for seed in seeds}

accs_array2 = np.zeros((len(seeds),len(users),len(policies),len(utilization_budgets)))

# ================================= sensor selection =================================

# iterate over all scenarios
for seed_i,seed in enumerate(seeds):
	init_seeds(seed)
	logger = init_logger(f"{logging_prefix}/sensel_seed{seed}")
	logger.info(f"Seed: {seed}")

	for user_i,user in enumerate(users):
		logger.info(f"\tUser: {user}")
		train_users = users[:user_i] + users[user_i+1:]
		test_users = [user]

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

		# data and table for each sensor
		sensor_conf_vals = {i:None for i in range(len(body_parts))}
		sensor_preds = {i:None for i in range(len(body_parts))}
		sensor_labels = {i:None for i in range(len(body_parts))}

		sensor_bin_boundaries = {i: torch.linspace(0, 1, n_bins+1) for i in range(len(body_parts))}
		sensor_bin_lowers = {i: sensor_bin_boundaries[i][:-1] for i in range(len(body_parts))}
		sensor_bin_uppers = {i: sensor_bin_boundaries[i][1:] for i in range(len(body_parts))}
		sensor_bin_marks = {i: torch.cat([sensor_bin_lowers[i],sensor_bin_uppers[i][-1].unsqueeze(0)]) for i in range(len(body_parts))}

		sensor_avg_acc_bins = {i: np.zeros(n_bins) for i in range(len(body_parts))} # acc per bin
		sensor_avg_conf_bins = {i: np.zeros(n_bins) for i in range(len(body_parts))} # conf dist
		sensor_ens_avg_acc_bins = {i: np.zeros(n_bins) for i in range(len(body_parts))} # acc of ens per sensor bin
		sensor_avg_acc_bins_U = {i: np.zeros(n_bins) for i in range(len(body_parts))} # acc per bin
		sensor_ens_avg_acc_bins_U = {i: np.zeros(n_bins) for i in range(len(body_parts))} # acc of ens per sensor bin
		sensor_avg_acc_bins_1 = {i: np.zeros(n_bins) for i in range(len(body_parts))} # acc per bin
		sensor_avg_acc_bins_5 = {i: np.zeros(n_bins) for i in range(len(body_parts))} # acc per bin
		sensor_ens_avg_acc_bins_1 = {i: np.zeros(n_bins) for i in range(len(body_parts))} # acc of ens per sensor bin
		sensor_ens_avg_acc_bins_5 = {i: np.zeros(n_bins) for i in range(len(body_parts))} # acc of ens per sensor bin

		sensor_thresh_policy = {i: np.zeros(n_bins) for i in range(len(body_parts))}
		sensor_thresh_policy_c = {i: 0.0 for i in range(len(body_parts))}
		sensor_opt_policy = {i: np.zeros(n_bins) for i in range(len(body_parts))}
		sensor_opt_policy_1 = {i: np.zeros(n_bins) for i in range(len(body_parts))}
		sensor_opt_policy_5 = {i: np.zeros(n_bins) for i in range(len(body_parts))}
		sensor_thresh_policy_U = {i: np.zeros(n_bins) for i in range(len(body_parts))}
		sensor_opt_policy_U = {i: np.zeros(n_bins) for i in range(len(body_parts))}

		# now load the data from each sensor
		for bp_i,bp in enumerate(body_parts):
			logger.info(f"\t\tLoading Body Part: {bp}")
			bp_logname = f"{logging_prefix}/{bp}_{train_users}_seed{seed}"
			bp_cache_path = os.path.join(PROJECT_ROOT,"saved_data/cached_data/"+bp_logname) + ".pt"
			bp_cached_data = torch.load(bp_cache_path)
			bp_confidence_values = bp_cached_data[0,:]
			bp_predictions = bp_cached_data[1,:]
			bp_labels = bp_cached_data[2,:]

			# first we need to shuffle the data and labels to make the order iid
			bp_confidence_values = bp_confidence_values[rand_order]
			bp_predictions = bp_predictions[rand_order]
			bp_labels = bp_labels[rand_order]

			sensor_conf_vals[bp_i] = bp_confidence_values
			sensor_preds[bp_i] = bp_predictions
			sensor_labels[bp_i] = bp_labels

		# use first 10% for getting conf hist
		calib_amt = int(0.1*len(rand_order))

		# ensure each sensor used an even number of times
		num_cycles = (len(rand_order)-calib_amt) // len(body_parts)

		# give extra to calib
		calib_amt = len(rand_order)-num_cycles*len(body_parts)

		# shuffle the order the senors get queried
		sensor_sequence = np.arange(0, len(body_parts))
		sensor_order = np.array([np.random.permutation(sensor_sequence) for cycle in range(num_cycles)]).flatten()

		# first go through the calibration stage (conf dist)
		# collect table for each sensor
		for sensor in range(len(body_parts)):
			i = 0

			# uniform bins
			s_confs_sorted_idxs = np.argsort(sensor_conf_vals[sensor][:calib_amt])
			ordered_confs = sensor_conf_vals[sensor][:calib_amt][s_confs_sorted_idxs]
			ordered_s_preds = bp_predictions[:calib_amt][s_confs_sorted_idxs]
			ordered_ens_preds = ens_predictions[:calib_amt][s_confs_sorted_idxs]
			ordered_labels = ens_labels[:calib_amt][s_confs_sorted_idxs]
			bin_amt = len(ordered_confs) // n_bins
			bin_boundaries = ordered_confs[::bin_amt]
			bin_boundaries[0] = 0.0
			bin_boundaries[-1] = 1.0

			sensor_bin_boundaries[sensor] = bin_boundaries
			sensor_bin_lowers[sensor] = sensor_bin_boundaries[sensor][:-1]
			sensor_bin_uppers[sensor] = sensor_bin_boundaries[sensor][1:]
			sensor_bin_marks[sensor] = torch.cat([sensor_bin_lowers[sensor],sensor_bin_uppers[sensor][-1].unsqueeze(0)])

			for bin_lower, bin_upper in zip(sensor_bin_lowers[sensor], sensor_bin_uppers[sensor]):
				# mask to determine which confidence scores go in this bin
				in_bin = sensor_conf_vals[sensor][:calib_amt].gt(bin_lower.item()) * sensor_conf_vals[sensor][:calib_amt].le(bin_upper.item())

				# from the mask determine what percent of the confidence scores are in this bin
				prop_in_bin = in_bin.float().mean()

				# what percent of predictions are in this bin
				sensor_avg_conf_bins[sensor][i] = len(sensor_conf_vals[sensor][:calib_amt][in_bin])/len(sensor_conf_vals[sensor][:calib_amt])

				# if the bin is not empty
				if prop_in_bin.item() > 0:
					sensor_accuracy_in_bin = (sensor_preds[sensor][:calib_amt]==sensor_labels[sensor][:calib_amt])[in_bin].float().mean()
					sensor_accuracy_in_bin_U = (sensor_preds[sensor][:calib_amt]==ens_predictions[:calib_amt])[in_bin].float().mean()
					ens_accuracy_in_bin = (ens_predictions[:calib_amt]==sensor_labels[sensor][:calib_amt])[in_bin].float().mean()
					in_bin_1 = in_bin[:len(in_bin)//10]
					in_bin_5 = in_bin[:len(in_bin)//2]
					ens_accuracy_in_bin_1 = (ens_predictions[:calib_amt]==sensor_labels[sensor][:calib_amt])[:len(in_bin)//10][in_bin_1].float().mean()
					ens_accuracy_in_bin_5 = (ens_predictions[:calib_amt]==sensor_labels[sensor][:calib_amt])[:len(in_bin)//2][in_bin_5].float().mean()

					# get the average confidence score of items in this bin
					# sensor_avg_confidence_in_bin = sensor_conf_vals[sensor][in_bin].mean()
					sensor_avg_acc_bins[sensor][i] = sensor_accuracy_in_bin
					sensor_ens_avg_acc_bins[sensor][i] = ens_accuracy_in_bin # normal

					sensor_avg_acc_bins_U[sensor][i] = sensor_accuracy_in_bin_U
					sensor_ens_avg_acc_bins_U[sensor][i] = 1 # unsupervised

					sensor_avg_acc_bins_1[sensor][i] = sensor_accuracy_in_bin_U*ens_accuracy_in_bin_1
					sensor_ens_avg_acc_bins_1[sensor][i] = ens_accuracy_in_bin_1 # 1 percent
					
					sensor_avg_acc_bins_5[sensor][i] = sensor_accuracy_in_bin_U*ens_accuracy_in_bin_5
					sensor_ens_avg_acc_bins_5[sensor][i] = ens_accuracy_in_bin_5 # 5 percent
				i+=1

		# now we execute the policies using the tables
		# execute policy for each budget
		rand_accs = []
		rand_utils = []
		threshold_accs = []
		threshold_utils = []
		optimal_accs = []
		optimal_utils = []
		optimal_accs_U = []
		optimal_utils_U = []
		optimal_accs_1 = []
		optimal_utils_1 = []
		optimal_accs_5 = []
		optimal_utils_5 = []
		for ui,(ub,ef) in enumerate(zip(utilization_budgets,ensemble_fractions)):
			logger.info(f"\t\t\tutilization budget: {ub}, {ef}")

			# threshold policy, just do a brute force search
			# optimal policy: diff
			for sensor in range(len(body_parts)):
				bin_budget = int(ef/(1/n_bins))
				max_exp_acc = 0.0
				best_policy = 0.0
				s_confs_sorted_idxs = np.argsort(sensor_conf_vals[sensor][:calib_amt])
				ordered_confs = sensor_conf_vals[sensor][:calib_amt][s_confs_sorted_idxs]
				ordered_s_preds = sensor_preds[sensor][:calib_amt][s_confs_sorted_idxs]
				ordered_ens_preds = ens_predictions[:calib_amt][s_confs_sorted_idxs]
				ordered_labels = ens_labels[:calib_amt][s_confs_sorted_idxs]
				max_thresh = sensor_bin_marks[sensor][bin_budget]
				search_thresh = np.linspace(0,max_thresh,50)
				for th in search_thresh:
					try:
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
					
				sensor_thresh_policy_c[sensor] = best_policy

				diff = sensor_ens_avg_acc_bins[sensor] - sensor_avg_acc_bins[sensor]
				diff_s_order = np.argsort(diff)[::-1]
				diff_s = diff[diff_s_order] # largest to smallest
				opt_count = 0
				for ds,ds_o in zip(diff_s,diff_s_order):
					if opt_count < bin_budget and ds > 0:
						sensor_opt_policy[sensor][ds_o] = 1
						opt_count += 1

				diff_1 = sensor_ens_avg_acc_bins_1[sensor] - sensor_avg_acc_bins_1[sensor]
				diff_s_order_1 = np.argsort(diff_1)[::-1]
				diff_s_1 = diff_1[diff_s_order_1] # largest to smallest
				opt_count_1 = 0
				for ds_1,ds_o_1 in zip(diff_s_1,diff_s_order_1):
					if opt_count_1 < bin_budget and ds_1 > 0:
						sensor_opt_policy_1[sensor][ds_o_1] = 1
						opt_count_1 += 1

				diff_5 = sensor_ens_avg_acc_bins_5[sensor] - sensor_avg_acc_bins_5[sensor]
				diff_s_order_5 = np.argsort(diff_5)[::-1]
				diff_s_5 = diff_5[diff_s_order_5] # largest to smallest
				opt_count_5 = 0
				for ds_5,ds_o_5 in zip(diff_s_5,diff_s_order_5):
					if opt_count_5 < bin_budget and ds_5 > 0:
						sensor_opt_policy_5[sensor][ds_o_5] = 1
						opt_count_5 += 1

				diff_U = sensor_ens_avg_acc_bins_U[sensor] - sensor_avg_acc_bins_U[sensor]
				diff_s_order_U = np.argsort(diff_U)[::-1]
				diff_s_U = diff_U[diff_s_order_U] # largest to smallest
				opt_count_U = 0
				for ds,ds_o in zip(diff_s_U,diff_s_order_U):
					if opt_count_U < bin_budget and ds > 0:
						sensor_opt_policy_U[sensor][ds_o] = 1
						opt_count_U += 1

				logger.info(f"\t\t\ts{sensor}: best threshold policy: {best_policy}")
				logger.info(f"\t\t\ts{sensor}: best optimal policy: {sensor_opt_policy[sensor]}")
				logger.info(f"\t\t\ts{sensor}: best optimal_U policy: {sensor_opt_policy_U[sensor]}")
				logger.info(f"\t\t\ts{sensor}: best optimal_1 policy: {sensor_opt_policy_1[sensor]}")
				logger.info(f"\t\t\ts{sensor}: best optimal_5 policy: {sensor_opt_policy_5[sensor]}")


			'''------------------------------------------------------------------'''
			st = time.time()
			count = 0
			sample_idx = 0
			c = 0
			selections = np.zeros(len(sensor_order))
			for idx,sensor in enumerate(sensor_order):
				# get the current sensors predictions and the ensemble prediction
				ens_conf,ens_pred = ens_confidence_values[calib_amt:][idx], ens_predictions[calib_amt:][idx]
				sensor_conf, sensor_pred = sensor_conf_vals[sensor][calib_amt:][idx], sensor_preds[sensor][calib_amt:][idx]
				label = ens_labels[calib_amt:][idx]

				# make sure we don't exceed utilization budget of any sensor
				rand_exceeded = False
				rand_selections = np.array(results_dict[seed][user]['random'][ub]['selection'])
				thresh_exceeded = False
				thresh_selections = np.array(results_dict[seed][user]['threshold'][ub]['selection'])
				opt_exceeded = False
				opt_selections = np.array(results_dict[seed][user]['optimal'][ub]['selection'])
				opt_exceeded_U = False
				opt_selections_U = np.array(results_dict[seed][user]['optimal_U'][ub]['selection'])
				opt_exceeded_1 = False
				opt_selections_1 = np.array(results_dict[seed][user]['optimal_1'][ub]['selection'])
				opt_exceeded_5 = False
				opt_selections_5 = np.array(results_dict[seed][user]['optimal_5'][ub]['selection'])
				utils = []
				for other_sensor in range(len(body_parts)): 
					rand_curr_util = (sum(rand_selections == -1) + sum(rand_selections == other_sensor))
					rand_next_util = (rand_curr_util + 1)/(len(rand_selections) + 1)
					if rand_next_util > ub:
						rand_exceeded = True

					thresh_curr_util = (sum(thresh_selections == -1) + sum(thresh_selections == other_sensor))
					thresh_next_util = (thresh_curr_util + 1)/(len(thresh_selections) + 1)
					if thresh_next_util > ub:
						thresh_exceeded = True

					opt_curr_util = (sum(opt_selections == -1) + sum(opt_selections == other_sensor))
					opt_next_util = (opt_curr_util + 1)/(len(opt_selections) + 1)
					if opt_next_util > ub:
						opt_exceeded = True

					opt_curr_util_U = (sum(opt_selections_U == -1) + sum(opt_selections_U == other_sensor))
					opt_next_util_U = (opt_curr_util_U + 1)/(len(opt_selections_U) + 1)
					if opt_next_util_U > ub:
						opt_exceeded_U = True

					opt_curr_util_1 = (sum(opt_selections_1 == -1) + sum(opt_selections_1 == other_sensor))
					opt_next_util_1 = (opt_curr_util_1 + 1)/(len(opt_selections_1) + 1)
					if opt_next_util_1 > ub:
						opt_exceeded_1 = True

					opt_curr_util_5 = (sum(opt_selections_5 == -1) + sum(opt_selections_5 == other_sensor))
					opt_next_util_5 = (opt_curr_util_5 + 1)/(len(opt_selections_5) + 1)
					if opt_next_util_5 > ub:
						opt_exceeded_5 = True

				rand_val = torch.rand(1)
				# random policy
				if (rand_val > ef) or rand_exceeded:
					rand_pred = sensor_pred.item()
					rand_sel = sensor
						
				else:
					rand_pred = ens_pred.item()
					rand_sel = -1
				
				results_dict[seed][user]['random'][ub]['preds'].append(rand_pred)
				results_dict[seed][user]['random'][ub]['labels'].append(label)
				results_dict[seed][user]['random'][ub]['selection'].append(rand_sel)

				conf_thresh = sensor_thresh_policy_c[sensor]

				if thresh_exceeded or sensor_conf >= conf_thresh:
					thresh_pred = sensor_pred.item()
					thresh_sel = sensor
						
				else:
					thresh_pred = ens_pred.item()
					thresh_sel = -1
				
				results_dict[seed][user]['threshold'][ub]['preds'].append(thresh_pred)
				results_dict[seed][user]['threshold'][ub]['labels'].append(label)
				results_dict[seed][user]['threshold'][ub]['selection'].append(thresh_sel)

				# optimal policy
				# need to determine if in the bin
				bin_loc = (sensor_conf >= sensor_bin_marks[sensor]).nonzero().flatten()[-1]
				
				if opt_exceeded or sensor_opt_policy[sensor][bin_loc] != 1:
					opt_pred = sensor_pred.item()
					opt_sel = sensor
						
				else:
					opt_pred = ens_pred.item()
					opt_sel = -1
				
				results_dict[seed][user]['optimal'][ub]['preds'].append(opt_pred)
				results_dict[seed][user]['optimal'][ub]['labels'].append(label)
				results_dict[seed][user]['optimal'][ub]['selection'].append(opt_sel)

				
				if opt_exceeded_U or sensor_opt_policy_U[sensor][bin_loc] != 1:
					opt_pred_U = sensor_pred.item()
					opt_sel_U = sensor
						
				else:
					opt_pred_U = ens_pred.item()
					opt_sel_U = -1
				
				results_dict[seed][user]['optimal_U'][ub]['preds'].append(opt_pred_U)
				results_dict[seed][user]['optimal_U'][ub]['labels'].append(label)
				results_dict[seed][user]['optimal_U'][ub]['selection'].append(opt_sel_U)

				if opt_exceeded_1 or sensor_opt_policy_1[sensor][bin_loc] != 1:
					opt_pred_1 = sensor_pred.item()
					opt_sel_1 = sensor
						
				else:
					opt_pred_1 = ens_pred.item()
					opt_sel_1 = -1
				
				results_dict[seed][user]['optimal_1'][ub]['preds'].append(opt_pred_1)
				results_dict[seed][user]['optimal_1'][ub]['labels'].append(label)
				results_dict[seed][user]['optimal_1'][ub]['selection'].append(opt_sel_1)

				if opt_exceeded_5 or sensor_opt_policy_5[sensor][bin_loc] != 1:
					opt_pred_5 = sensor_pred.item()
					opt_sel_5 = sensor
						
				else:
					opt_pred_5 = ens_pred.item()
					opt_sel_5 = -1
				
				results_dict[seed][user]['optimal_5'][ub]['preds'].append(opt_pred_5)
				results_dict[seed][user]['optimal_5'][ub]['labels'].append(label)
				results_dict[seed][user]['optimal_5'][ub]['selection'].append(opt_sel_5)
				count += 1
			logger.info(f"exec time: {time.time()-st}, {count}")
			'''------------------------------------------------------------------'''


			# 		sample_idx += 1
			rand_pred = np.array(results_dict[seed][user]['random'][ub]['preds'])
			rand_lab = np.array(results_dict[seed][user]['random'][ub]['labels'])
			rand_sel = np.array(results_dict[seed][user]['random'][ub]['selection'])

			thresh_pred = np.array(results_dict[seed][user]['threshold'][ub]['preds'])
			thresh_lab = np.array(results_dict[seed][user]['threshold'][ub]['labels'])
			thresh_sel = np.array(results_dict[seed][user]['threshold'][ub]['selection'])

			opt_pred = np.array(results_dict[seed][user]['optimal'][ub]['preds'])
			opt_lab = np.array(results_dict[seed][user]['optimal'][ub]['labels'])
			opt_sel = np.array(results_dict[seed][user]['optimal'][ub]['selection'])

			opt_pred_U = np.array(results_dict[seed][user]['optimal_U'][ub]['preds'])
			opt_lab_U = np.array(results_dict[seed][user]['optimal_U'][ub]['labels'])
			opt_sel_U = np.array(results_dict[seed][user]['optimal_U'][ub]['selection'])

			opt_pred_1 = np.array(results_dict[seed][user]['optimal_1'][ub]['preds'])
			opt_lab_1 = np.array(results_dict[seed][user]['optimal_1'][ub]['labels'])
			opt_sel_1 = np.array(results_dict[seed][user]['optimal_1'][ub]['selection'])

			opt_pred_5 = np.array(results_dict[seed][user]['optimal_5'][ub]['preds'])
			opt_lab_5 = np.array(results_dict[seed][user]['optimal_5'][ub]['labels'])
			opt_sel_5 = np.array(results_dict[seed][user]['optimal_5'][ub]['selection'])

			logger.info(f"\t\t\t\trand_f1: {f1_score(rand_lab,rand_pred,average='macro')}")
			rand_accs.append(f1_score(rand_lab,rand_pred,average='macro'))

			logger.info(f"\t\t\t\tthresh_f1: {f1_score(thresh_lab,thresh_pred,average='macro')}")
			threshold_accs.append(f1_score(thresh_lab,thresh_pred,average='macro'))

			logger.info(f"\t\t\t\topt_f1: {f1_score(opt_lab,opt_pred,average='macro')}")
			optimal_accs.append(f1_score(opt_lab,opt_pred,average='macro'))
			logger.info(f"\t\t\t\topt_f1_U: {f1_score(opt_lab_U,opt_pred_U,average='macro')}")
			optimal_accs_U.append(f1_score(opt_lab_U,opt_pred_U,average='macro'))

			logger.info(f"\t\t\t\topt_f1_1: {f1_score(opt_lab_1,opt_pred_1,average='macro')}")
			optimal_accs_1.append(f1_score(opt_lab_1,opt_pred_1,average='macro'))

			logger.info(f"\t\t\t\topt_f1_5: {f1_score(opt_lab_5,opt_pred_5,average='macro')}")
			optimal_accs_5.append(f1_score(opt_lab_5,opt_pred_5,average='macro'))

			accs_array2[seed_i][user_i][0][ui] = f1_score(rand_lab,rand_pred,average='macro')
			accs_array2[seed_i][user_i][1][ui] = f1_score(thresh_lab,thresh_pred,average='macro')
			accs_array2[seed_i][user_i][2][ui] = f1_score(opt_lab,opt_pred,average='macro')
			accs_array2[seed_i][user_i][3][ui] = f1_score(opt_lab_U,opt_pred_U,average='macro')
			accs_array2[seed_i][user_i][4][ui] = f1_score(opt_lab_1,opt_pred_1,average='macro')
			accs_array2[seed_i][user_i][5][ui] = f1_score(opt_lab_5,opt_pred_5,average='macro')
			with open(f"dsads_accs_array_{n_bins}_bins___U.pkl", 'wb') as file:
				pickle.dump(accs_array2, file)

			for other_sensor in range(len(body_parts)):
				curr_util = (sum(rand_sel == -1) + sum(rand_sel == other_sensor))/len(rand_sel)
				logger.info(f"\t\t\t\tsensor: {body_parts[other_sensor]}, rand_util: {curr_util}, {sum(rand_sel == -1) + sum(rand_sel == other_sensor)}/{len(rand_sel)}")
				rand_utils.append(curr_util)

				curr_util = (sum(thresh_sel == -1) + sum(thresh_sel == other_sensor))/len(thresh_sel)
				logger.info(f"\t\t\t\tsensor: {body_parts[other_sensor]}, thresh_util: {curr_util}, {sum(thresh_sel == -1) + sum(thresh_sel == other_sensor)}/{len(thresh_sel)}")
				threshold_utils.append(curr_util)

				curr_util = (sum(opt_sel == -1) + sum(opt_sel == other_sensor))/len(opt_sel)
				logger.info(f"\t\t\t\tsensor: {body_parts[other_sensor]}, opt_util: {curr_util}, {sum(opt_sel == -1) + sum(opt_sel == other_sensor)}/{len(opt_sel)}")
				optimal_utils.append(curr_util)
				curr_util_U = (sum(opt_sel_U == -1) + sum(opt_sel_U == other_sensor))/len(opt_sel_U)
				logger.info(f"\t\t\t\tsensor: {body_parts[other_sensor]}, opt_util_U: {curr_util_U}, {sum(opt_sel_U == -1) + sum(opt_sel_U == other_sensor)}/{len(opt_sel_U)}")
				optimal_utils_U.append(curr_util_U)

				curr_util_1 = (sum(opt_sel_1 == -1) + sum(opt_sel_1 == other_sensor))/len(opt_sel_1)
				logger.info(f"\t\t\t\tsensor: {body_parts[other_sensor]}, opt_util_1: {curr_util_1}, {sum(opt_sel_1 == -1) + sum(opt_sel_1 == other_sensor)}/{len(opt_sel_1)}")
				optimal_utils_1.append(curr_util_1)

				curr_util_5 = (sum(opt_sel_5 == -1) + sum(opt_sel_5 == other_sensor))/len(opt_sel_5)
				logger.info(f"\t\t\t\tsensor: {body_parts[other_sensor]}, opt_util_5: {curr_util_5}, {sum(opt_sel_5 == -1) + sum(opt_sel_5 == other_sensor)}/{len(opt_sel_5)}")
				optimal_utils_5.append(curr_util_5)

		plt.clf()
		plt.plot(utilization_budgets,rand_accs,label='Random Policy')
		plt.plot(utilization_budgets,threshold_accs,label='Threshold Policy')
		plt.plot(utilization_budgets,optimal_accs,label='Optimal Policy')
		plt.plot(utilization_budgets,optimal_accs_U,label='Optimal Policy U',c=plt.gca().get_lines()[-1].get_color(),linestyle='--')
		plt.plot(utilization_budgets,optimal_accs_1,label='Optimal Policy 1',c=plt.gca().get_lines()[-1].get_color(),linestyle='dotted')
		plt.plot(utilization_budgets,optimal_accs_5,label='Optimal Policy 5',c=plt.gca().get_lines()[-1].get_color(),linestyle='dashdot')
		plt.xlabel("Utilization Budget")
		plt.ylabel("Accuracy")
		plt.legend()
		plt.savefig(f"dsads__{seed}_{test_users}.png",dpi=200)
			# # exit()

accs_array = np.zeros((len(seeds),len(users),len(policies),len(utilization_budgets)))
for seed_i,seed in enumerate(seeds):
	for user_i,user in enumerate(users):
		for policy_i,policy in enumerate(policies):
			for ub_i,ub in enumerate(utilization_budgets):
				preds = np.array(results_dict[seed][user][policy][ub]['preds'])
				labels = np.array(results_dict[seed][user][policy][ub]['labels'])
				# acc = (preds == labels).mean()
				acc = f1_score(labels,preds,average='macro')
				accs_array[seed_i][user_i][policy_i][ub_i] = acc

with open(f"dsads_accs_array_{n_bins}_bins_U.pkl", 'wb') as file:
	pickle.dump(accs_array, file)