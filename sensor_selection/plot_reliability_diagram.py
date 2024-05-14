import torch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import collections  as mc
import numpy as np
import argparse

from utils.setup_funcs import *
from datasets.dsads.dsads import *
from datasets.opportunity.opportunity import *
from datasets.rwhar.rwhar import *
from training.models import *
from training.train import *


class CalibrationToolsSTUniform:
	def __init__(self,s_models,s_test_loaders,t_models,t_test_loaders,num_bins):
		self.s_models = s_models
		self.s_test_loaders = s_test_loaders
		self.t_models = t_models
		self.t_test_loaders = t_test_loaders
		self.num_bins = num_bins

		# accumulate predictions
		self.get_model_outputs()
		
		# set up the histograms based on student confidence
		self.s_confs_sorted_idxs = np.argsort(self.s_confidence_values)
		self.ordered_confs = self.s_confidence_values[self.s_confs_sorted_idxs]
		self.ordered_s_preds = self.s_predictions[self.s_confs_sorted_idxs]
		self.ordered_t_preds = self.t_predictions[self.s_confs_sorted_idxs]
		self.ordered_labels = self.labels[self.s_confs_sorted_idxs]
		bin_amt = len(self.ordered_confs) // num_bins
		self.bin_boundaries = self.ordered_confs[::bin_amt]
		self.bin_boundaries[0] = 0.0
		self.bin_boundaries[-1] = 1.0
		self.bin_lowers = self.bin_boundaries[:-1]
		self.bin_uppers = self.bin_boundaries[1:]
		self.bin_marks = torch.cat([self.bin_lowers,self.bin_uppers[-1].unsqueeze(0)])

	def get_model_outputs(self):
		# accumulate all the predictions
		with torch.no_grad():
			# put in eval mode
			for s_model_i,s_model in enumerate(self.s_models):
				self.s_models[s_model_i].eval()
				
			for t_model_i,t_model in enumerate(self.t_models):
				self.t_models[t_model_i].eval()

			# collect preds
			s_confidence_values = []
			s_predictions = []
			t_predictions = []
			labels = []
			s_iters = []
			t_iters = []

			# get test loaders
			for s_tl in self.s_test_loaders:
				s_iters.append(iter(s_tl))

			for t_tl in self.t_test_loaders:
				t_iters.append(iter(t_tl))

			# get prediction for each batch
			for batch_idx in range(len(self.s_test_loaders[0])):
				s_ens_preds = []
				t_ens_preds = []

				# ====== student ======
				# get the prediction for the batch
				for s_it,s_model in zip(s_iters,self.s_models):
					data,target = next(s_it)
					sm_out = F.softmax(s_model(data.float()),dim=1)

					if len(self.s_models) > 1:
						s_ens_preds.append(sm_out)

				# get the avg prediction over each model
				if len(self.s_models) > 1:
					sm_out = torch.stack(s_ens_preds).mean(dim=0)

				# parse the output for the prediction
				cs,ps = sm_out.max(dim=1)

				s_confidence_values.append(cs)
				s_predictions.append(ps)
				labels.append(target)

				# ====== teacher ======
				# get the prediction for the batch
				for t_it,t_model in zip(t_iters,self.t_models):
					data,target = next(t_it)
					sm_out = F.softmax(t_model(data.float()),dim=1)

					if len(self.t_models) > 1:
						t_ens_preds.append(sm_out)

				# get the avg prediction over each model
				if len(self.t_models) > 1:
					sm_out = torch.stack(t_ens_preds).mean(dim=0)

				# parse the output for the prediction
				cs,ps = sm_out.max(dim=1)

				t_predictions.append(ps)
			
			# merge all
			self.s_confidence_values = torch.cat(s_confidence_values)
			self.s_predictions = torch.cat(s_predictions)
			self.t_predictions = torch.cat(t_predictions)
			self.labels = torch.cat(labels)


	def get_conf_hist(self):
		self.s_accs = [0.0]*self.num_bins # accuracy per conf bin
		self.t_accs = [0.0]*self.num_bins # accuracy per conf bin
		self.s_avg_confs = [0.0]*self.num_bins # avg conf in a bin
		self.s_confs_hist = [0.0]*self.num_bins # confidence histogram
		self.s_total_avg_conf = 0.0 # avg conf of a model
		self.s_total_acc = 0.0 # accuracy of a model

		# iterate over bins
		i = 0
		for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
			# mask to determine which confidence scores go in this bin
			in_bin = self.s_confidence_values.gt(bin_lower.item()) * self.s_confidence_values.le(bin_upper.item())

			# from the mask determine what percent of the confidence scores are in this bin
			prop_in_bin = in_bin.float().mean()

			# what percent of predictions are in this bin
			self.s_confs_hist[i] = len(self.s_confidence_values[in_bin])/len(self.s_confidence_values)

			# if the bin is not empty
			if prop_in_bin.item() > 0:
				# calculate accuracy of the bin using the mask of correct predictions
				s_accuracy_in_bin = (self.s_predictions==self.labels)[in_bin].float().mean()
				t_accuracy_in_bin = (self.t_predictions==self.labels)[in_bin].float().mean()

				# get the average confidence score of items in this bin
				s_avg_confidence_in_bin = self.s_confidence_values[in_bin].mean()
				self.s_accs[i] = s_accuracy_in_bin
				self.t_accs[i] = t_accuracy_in_bin
				self.s_avg_confs[i] = s_avg_confidence_in_bin
			i+=1
		
		# get the average accuracy and confidence and the expected calibration error
		self.s_total_avg_conf = sum([self.s_avg_confs[i]*self.s_confs_hist[i] for i in range(len(self.s_confs_hist))]) # weight by confidence histogram
		self.s_total_acc = sum([self.s_accs[i]*self.s_confs_hist[i] for i in range(len(self.s_confs_hist))])
		self.t_total_acc = (self.t_predictions==self.labels).float().mean()
		self.s_ece = sum([abs(self.s_accs[i] - self.s_avg_confs[i])*self.s_confs_hist[i] for i in range(len(self.s_avg_confs))])

	
	def plot_calib(self,user,body_part,best_policy,opt_policy):
		# confidence histogram
		bar_ticks = torch.cat([self.bin_lowers[0].unsqueeze(0),self.bin_uppers])
		bar_xtickpos = [round(bar_ticks[i].item(),2)*10-0.5 for i in range(len(bar_ticks))]
		bar_ticks = [str(round(bar.item(),2)) for bar in bar_ticks]
		bar_xpos = (np.array(bar_xtickpos[:-1])+np.array(bar_xtickpos[1:]))/2
		
		w = np.array(bar_xtickpos[1:])-np.array(bar_xtickpos[:-1])

		fig = plt.figure(figsize=(35,12))

		gs = fig.add_gridspec(2,2)
		ax1 = fig.add_subplot(gs[0, 0])
		ax2 = fig.add_subplot(gs[0, 1])
		ax3 = fig.add_subplot(gs[1, :])
		plt.rc('axes', titlesize=12)     # fontsize of the axes title
		plt.rc('axes', labelsize=12)

		ax1.set_axisbelow(True)
		ax2.set_axisbelow(True)
		ax3.set_axisbelow(True)

		
		ax1.bar(bar_xpos,self.s_accs,width=w,edgecolor='black',linewidth=2,label="accuracy")
		ax1.bar(bar_xpos,[self.s_avg_confs[i]-self.s_accs[i] for i in range(len(self.s_accs))],
				  label="confidence gap",width=w,edgecolor='red',bottom=self.s_accs,color='red',alpha=0.25,linewidth=2)
		ax1.set_xticks(bar_xtickpos,bar_ticks,rotation=90,fontsize=14)
		ax1.set_ylim([0,1])
		ax1.set_xlim(-0.5,self.num_bins-0.5)
		ax1.grid()
		ax1.set_title(fr'User {user} Reliability Diagram: ${{acc_{{ {body_part}}} }}$',fontsize=24)
		ax1.set_xlabel("Confidence",fontsize=24)
		ax1.set_yticks(np.linspace(0,1,11),np.round(np.linspace(0,1,11),2),fontsize=16)
		ax1.set_ylabel("Accuracy",fontsize=24)
		for acc_i,acc in enumerate(self.s_accs):
			try:
				acc = acc.item()
				conf = self.s_avg_confs[acc_i].item()
			except:
				continue
			acc_i  = bar_xtickpos[acc_i]+0.45
			
		ax1.legend(fontsize=16)
		i = 0
		
		colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
		for do in opt_policy[1]:
			if opt_policy[0][do] != 1:
				break
			label = ax1.text(bar_xpos[do], .075, f"{i}",size=22, ha="center",c=colors[2],
					path_effects=[pe.withStroke(linewidth=1, foreground="black")])
			i += 1
		
		lines = [[(best_policy*10-.5, 0), (best_policy*10-.5, 0.35)]]
		lc = mc.LineCollection(lines, colors=colors[1], linewidths=2,linestyle="--")
		ax1.add_collection(lc)
		ax1.arrow(best_policy*10-.5,0.3,-1.2,0,width=0.015,facecolor=colors[1])
		ax2.bar(bar_xpos,self.t_accs,width=w,edgecolor='black',linewidth=2,label="accuracy")
		ax2.bar(bar_xpos,[self.s_avg_confs[i]-self.t_accs[i] for i in range(len(self.t_accs))],
				  label="confidence gap",width=w,edgecolor='red',bottom=self.t_accs,color='red',alpha=0.25,linewidth=2)
		ax2.set_xticks(bar_xtickpos,bar_ticks,rotation=90,fontsize=14)
		ax2.set_ylim([0,1])
		ax2.set_xlim(-0.5,self.num_bins-0.5)
		ax2.grid()
		ax2.set_title(fr'User {user} Reliability Diagram: ${{acc_{{ ensemble | {body_part}}} }}$',fontsize=24)
		ax2.set_xlabel("Confidence",fontsize=24)
		ax2.set_yticks(np.linspace(0,1,11),np.round(np.linspace(0,1,11),2),fontsize=16)
		ax2.set_ylabel("Accuracy",fontsize=24)
		for acc_i,acc in enumerate(self.t_accs):
			try:
				acc = acc.item()
				conf = self.s_avg_confs[acc_i].item()
			except:
				continue
			
			acc_i  = bar_xtickpos[acc_i]+0.45
		ax2.legend(fontsize=16)

		
		ax3.bar(bar_xpos,np.array(self.t_accs)-np.array(self.s_accs),width=w,edgecolor='black',linewidth=2,label="accuracy")
		ax3.set_xticks(bar_xtickpos,bar_ticks,rotation=90,fontsize=14)
		ax3.set_ylim([0,1])
		ax3.set_xlim(-0.5,self.num_bins-0.5)
		ax3.grid()
		ax3.set_title(fr'User {user} Accuracy Margin: ${{acc_{{diff}} }}$',fontsize=24)
		ax3.set_xlabel("Confidence",fontsize=24)
		ax3.set_yticks(np.linspace(0,1,11),np.round(np.linspace(0,1,11),2),fontsize=16)
		ax3.set_ylabel("Accuracy",fontsize=24)
		for acc_i,acc in enumerate(self.t_accs):
			try:
				acc = acc.item()
				conf = self.s_avg_confs[acc_i].item()
			except:
				continue
			
			acc_i  = bar_xtickpos[acc_i]+0.45
		ax3.legend(fontsize=16)
		fig.subplots_adjust(hspace=0.25)
		fig.tight_layout()
		plt.savefig("reliability.svg")
		# plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Sensor Selection')
	parser.add_argument('--dataset', type=str,default='rwhar', help='which dataset to use')
	parser.add_argument('--user', type=int,default=11, help='which user to consider')
	parser.add_argument('--body_part', type=str,default='waist', help='which body part to consider')
	parser.add_argument('--ens_frac', type=float,default=0.5, help='number between [0,1], 1 means max budget')
	args = parser.parse_args()

	if args.dataset == 'dsads':
		body_parts = ['torso','right_arm','left_arm','right_leg','left_leg']
		users = [1,2,3,4,5,6,7,8]
	elif args.dataset == 'opportunity':
		body_parts = ["BACK","RUA","RLA","LUA","LLA","L-SHOE","R-SHOE"]
		users = [1,2,3,4]
	elif args.dataset == 'rwhar':
		body_parts = ['chest','forearm','head','shin','thigh','upperarm','waist']
		users = [1,3,4,5,7,8,9,10,11,12,13,14,15]
	
	for user_i,user in enumerate(users):
		train_users = users[:user_i] + users[user_i+1:]
		test_users = [user]
		if user_i == users.index(args.user):
			break

	# create the datasets
	test_loaders = []
	val_loaders = []
	models = []
	print(train_users,test_users)
	for bp_i,bp in enumerate(body_parts):
		if args.dataset == 'dsads':
			train_loader,val_loader,test_loader = load_dsads_person_dataset(32,train_users,test_users,[bp])
		elif args.dataset == 'opportunity':
			train_loader,val_loader,test_loader = load_opportunity_person_dataset(32,train_users,test_users,[bp])
		elif args.dataset == 'rwhar':
			train_loader,val_loader,test_loader = load_rwhar_person_dataset(32,train_users,test_users,[bp])
		val_loaders.append(val_loader)
		test_loaders.append(test_loader)

		# init models
		if args.dataset == 'dsads':
			model = DSADSNet(3)
		elif args.dataset == 'opportunity':
			model = OppNet(3)
		elif args.dataset == 'rwhar':
			model = RWHARNet(3)

		# load the model if already trained
		logging_prefix = f"{args.dataset}"
		ckpt_path = os.path.join(PROJECT_ROOT,f"saved_data/checkpoints/{logging_prefix}/{bp}_{train_users}_seed{123}.pth")
		
		model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
		models.append(model)

	nbins = 10
	ct = CalibrationToolsSTUniform([models[body_parts.index(args.body_part)]],[test_loaders[body_parts.index(args.body_part)]],models,test_loaders,nbins)
	ct.get_conf_hist()

	max_exp_acc = 0.0
	best_policy = 0.0
	bin_budget = int(args.ens_frac/(1/nbins))
	s_confs_sorted_idxs = np.argsort(ct.s_confidence_values)
	ordered_confs = ct.s_confidence_values[s_confs_sorted_idxs]
	ordered_s_preds = ct.s_predictions[s_confs_sorted_idxs]
	ordered_ens_preds = ct.t_predictions[s_confs_sorted_idxs]
	ordered_labels = ct.labels[s_confs_sorted_idxs]
	max_thresh = ct.bin_boundaries[bin_budget]
	
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


	diff = np.array(ct.t_accs) - np.array(ct.s_accs)
	diff_s_order = np.argsort(diff)[::-1]
	diff_s = diff[diff_s_order] # largest to smallest
	opt_count = 0
	opt_policy = np.zeros(nbins)
	for ds,ds_o in zip(diff_s,diff_s_order):
		if opt_count < bin_budget and ds > 0:
			opt_policy[ds_o] = 1
			opt_count += 1

	print(best_policy,opt_policy)

	ct.plot_calib(users[user_i],body_parts[body_parts.index(args.body_part)],best_policy,[opt_policy,diff_s_order])


	