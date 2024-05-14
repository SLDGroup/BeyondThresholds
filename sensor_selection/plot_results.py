import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import argparse

from utils.setup_funcs import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Sensor Selection Plots')
	parser.add_argument('--dataset', type=str, help='which dataset to use')
	args = parser.parse_args()
	
	# constants
	policies = ['random','threshold','optimal','optimal_U','optimal_1','optimal_5']
	seeds = [123,456,789]
	datasets = ['dsads','opportunity','rwhar']

	n_bins = 10
	bw = 1/n_bins
	ensemble_fractions = np.arange(0,1+bw,bw)

	# first load all results
	results = {dataset: {policy: [] for policy in policies} for dataset in datasets}
	
	for policy in policies:
		for dataset in datasets:
			for seed in seeds:
				cache_path = os.path.join(PROJECT_ROOT,f"saved_data/cached_data/{dataset}")
				fn = os.path.join(cache_path,f"selection/{policy}_{seed}")
				results[dataset][policy].append(np.load(fn+'.npy'))
			results[dataset][policy] = np.stack(results[dataset][policy],axis=0)

	# define params per dataset
	if args.dataset == 'dsads':
		body_parts = ['torso','right_arm','left_arm','right_leg','left_leg']
		users = [1,2,3,4,5,6,7,8]
		rows = 2
		cols = 4
		figsize = (24,8)
		wspace=0.25
		hspace=0.35
	elif args.dataset == 'opportunity':
		body_parts = ["BACK","RUA","RLA","LUA","LLA","L-SHOE","R-SHOE"]
		users = [1,2,3,4]
		rows = 1
		cols = 4
		figsize = (24,5)
		wspace=0.25
		hspace=0.35
	elif args.dataset == 'rwhar':
		body_parts = ['chest','forearm','head','shin','thigh','upperarm','waist']
		users = [1,3,4,5,7,8,9,10,11,12,13,14,15]
		rows = 3
		cols = 5
		figsize = (24,12)
		wspace=0.35
		hspace=0.35

	policy_names = ['random','threshold','optimal_estimate','optimal_estimate_U','optimal_estimate_1','optimal_estimate_5']

	k = len(body_parts)
	utilization_budgets = (ensemble_fractions*(k-1)+1)/k


# plot average over seeds (one plot per user)
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
fig,ax = plt.subplots(rows,cols,figsize=figsize)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_colors = [colors[0],colors[1],colors[2],colors[3],colors[4],colors[5]]
line_styles = ['-','-','-','--','--','--']
for user_i,user in enumerate(users):
	row = user_i // cols
	col = user_i % cols
	for policy_i,policy in enumerate(policies): 
		mean_over_seeds = np.mean(results[args.dataset][policy],0)
		std_over_seeds = np.std(results[args.dataset][policy],0) 
		if args.dataset == 'opportunity':
			ax[col].errorbar(utilization_budgets,mean_over_seeds[user_i,:], \
							 std_over_seeds[user_i,:],label=policy_names[policy_i], \
							 capsize=5,color=line_colors[policy_i],linestyle=line_styles[policy_i])
		else:
			ax[row,col].errorbar(utilization_budgets,mean_over_seeds[user_i,:], \
								std_over_seeds[user_i,:],label=policy_names[policy_i], \
								capsize=5,color=line_colors[policy_i],linestyle=line_styles[policy_i])
	
	if args.dataset == 'opportunity':
		ax[col].set_title(f"User: {user}, {args.dataset} Dataset",fontsize=16)
		ax[col].set_xlabel("Utilization Budget",fontsize=16)
		ax[col].set_ylabel("F1 Score",fontsize=16)
		ax[0].legend(loc='lower right')
		ax[col].grid()
	else:
		ax[row,col].set_title(f"User: {user}, {args.dataset} Dataset",fontsize=16)
		ax[row,col].set_xlabel("Utilization Budget",fontsize=16)
		ax[row,col].set_ylabel("F1 Score",fontsize=16)
		ax[0,0].legend(loc='lower right')
		ax[row,col].grid()
if args.dataset == 'rwhar':
	fig.delaxes(ax[2][3])
	fig.delaxes(ax[2][4])
fig.subplots_adjust(wspace=wspace,hspace=hspace)
fig.savefig(f'{args.dataset}_users.svg')


# plot average over users (one plot per seed)
fig,ax = plt.subplots(1,3,dpi=300,figsize=(18,4))
line_colors = [colors[0],colors[1],colors[2],colors[3],colors[4],colors[5]]
line_styles = ['-','-','-','--','--','--']
for seed_i,seed in enumerate(seeds):
	for policy_i,policy in enumerate(policies):
		mean_over_users = np.mean(results[args.dataset][policy],1)
		std_over_users = np.std(results[args.dataset][policy],1)
		ax[seed_i].errorbar(utilization_budgets,mean_over_users[seed_i,:],std_over_users[seed_i,:],label=policy,capsize=5,
							color=line_colors[policy_i],linestyle=line_styles[policy_i])
	ax[seed_i].set_title(f"Seed: {seed}, {args.dataset} Dataset",fontsize=16)
	ax[seed_i].set_xlabel("Utilization Budget",fontsize=16)
	ax[seed_i].set_ylabel("F1 Score",fontsize=16)
	ax[0].legend(loc='lower right',fontsize=9)
	ax[seed_i].grid()
fig.subplots_adjust(wspace=0.25,bottom=0.2)
fig.savefig(f'{args.dataset}_seeds.svg')