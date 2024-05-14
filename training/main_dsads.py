import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from utils.setup_funcs import *
from datasets.dsads.dsads import *
from train import *
from models import *


if __name__ == '__main__':
	# dataset details
	body_parts = ['torso','right_arm','left_arm','right_leg','left_leg']
	users = [1,2,3,4,5,6,7,8]

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	logging_prefix = "dsads"

	# hyperparameters
	EPOCHS = 50
	LR = 0.01
	MOMENTUM = 0.9
	WD = 1e-4
	ESE = 20
	BATCH_SIZE = 128

	seeds = [123,456,789]

	for seed_i,seed in enumerate(seeds):
		# setup the session
		logger = init_logger(f"{logging_prefix}/train_seed{seed}")
		init_seeds(seed)
		
		results_table = np.zeros((len(body_parts)+1+1,len(users)+1))
		logger.info(f"Seed: {seed}")

		train_users_list = []
		test_users_list = []

		bp_models = {i:[] for i in range(len(users))} # list of models for each loocv
		test_loaders = {i:[] for i in range(len(users))} # list of test loaders for each loocv
		
		for user_i,user in enumerate(users):
			train_users = users[:user_i] + users[user_i+1:]
			test_users = [user]

			train_users_list.append(str(train_users))
			test_users_list.append(str(test_users))

			# individual bp training
			for bp_i,bp in enumerate(body_parts):

				logger.info(f"Train Group: {train_users} --> Test Group: {test_users}, Body Part: {bp}")

				# create the datasets
				train_loader,val_loader,test_loader = load_dsads_person_dataset(BATCH_SIZE,train_users,test_users,[bp])
				test_loaders[user_i].append(test_loader)

				# init models
				model = DSADSNet(3)
				opt = torch.optim.SGD(model.parameters(),lr=LR,momentum=MOMENTUM,weight_decay=WD)

				# load the model if already trained
				ckpt_path = os.path.join(PROJECT_ROOT,f"saved_data/checkpoints/{logging_prefix}/{bp}_{train_users}_seed{seed}.pth")
				if os.path.exists(ckpt_path):
					model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
				else:
					# otherwise train the model
					train(model,nn.CrossEntropyLoss(label_smoothing=0.1),opt,f"{logging_prefix}/{bp}_{train_users}_seed{seed}",EPOCHS,ESE,device,
						train_loader,val_loader,logger,torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS),200)
					
				# load the one with the best validation accuracy and evaluate on test set
				model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
				val_acc,val_f1,val_loss = validate(model, test_loader, device, nn.CrossEntropyLoss())
				logger.info(f"{bp} Test F1: {val_f1}")
				results_table[bp_i][user_i] = val_f1.item()
				bp_models[user_i].append(model)

			# train early fusion
			logger.info(f"EF -- Train Group: {train_users} --> Test Group: {test_users}")

			# create the datasets
			train_loader,val_loader,test_loader = load_dsads_person_dataset(BATCH_SIZE,train_users,test_users,body_parts)

			# init models
			model = DSADSNet(3*len(body_parts))
			opt = torch.optim.SGD(model.parameters(),lr=LR,momentum=MOMENTUM,weight_decay=WD)

			# load the model if already trained
			ckpt_path = os.path.join(PROJECT_ROOT,f"saved_data/checkpoints/{logging_prefix}/{body_parts}_{train_users}_seed{seed}.pth")
			if os.path.exists(ckpt_path):
				model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
			else:
				# otherwise train the model
				train(model,nn.CrossEntropyLoss(),opt,f"{logging_prefix}/{body_parts}_{train_users}_seed{seed}",EPOCHS,ESE,device,
					train_loader,val_loader,logger,torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS),200)
				
			# load the one with the best validation accuracy and evaluate on test set
			model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
			val_acc,val_f1,val_loss = validate(model, test_loader, device, nn.CrossEntropyLoss())
			logger.info(f"EF Test F1: {val_f1}")
			results_table[-1][user_i] = val_f1.item()

			# eval ensemble
			val_acc,val_f1,val_loss = validate_ens_bps(bp_models[user_i], test_loaders[user_i], device, nn.CrossEntropyLoss())
			logger.info(f"LF Test F1: {val_f1}")
			results_table[-2][user_i] = val_f1.item()

		results_table[:,-1] = np.mean(results_table[:,:-1],axis=1)

		# save the results table
		logname = f"{logging_prefix}/seed{seed}"
		cache_path = os.path.join(PROJECT_ROOT,"saved_data/cached_data/"+logname) + ".npy"
		path_items = logname.split("/")
		if  len(path_items) > 1:
			Path(os.path.join(PROJECT_ROOT,"saved_data/cached_data",*path_items[:-1])).mkdir(parents=True, exist_ok=True)
		np.save(cache_path,results_table)

		# plot the results and save them
		fig,ax = plt.subplots(1,1,dpi=300)
		conf_mat = ax.imshow(results_table,cmap="Blues")
		ax.tick_params(axis=u'both', which=u'both',length=0)
		ax.set_ylabel("Body Parts",fontsize=10)
		ax.set_xlabel("Test Group",fontsize=10)
		
		test_users_list = [str(tu) for tu in test_users_list]
		ax.set_xticks([0 + i for i in range(results_table.shape[1])],test_users_list+['Avg'],rotation=90)
		ax.set_yticks([0 + i for i in range(results_table.shape[0])],body_parts+['Ensemble','Early Fusion'])
		fig.colorbar(conf_mat, orientation='vertical')

		logger.info(results_table)

		mid = np.percentile(results_table,85)
		for row in range(results_table.shape[0]):
			for col in range(results_table.shape[1]):
				val = round(results_table[row,col].item(),3)
				ax.text(col, row, str(val), va='center', ha='center',color=plt.get_cmap('gray')(int(val>mid)-0.01),fontsize=8)

		plt.subplots_adjust(bottom=0.35)
		plt.tight_layout()
		logname = f"{logging_prefix}/seed{seed}"
		plot_path = os.path.join(PROJECT_ROOT,"saved_data/plots/"+logname) + ".png"
		path_items = logname.split("/")
		if  len(path_items) > 1:
			Path(os.path.join(PROJECT_ROOT,"saved_data/plots",*path_items[:-1])).mkdir(parents=True, exist_ok=True)
		plt.savefig(plot_path)

