import torch

from utils.setup_funcs import *
from datasets.dsads.dsads import *
from train import *
from models import *


# helper function to collect model outputs
def get_model_outputs(models,test_loaders):
	# accumulate all the predictions
	with torch.no_grad():
		# put in eval mode
		for model_i,model in enumerate(models):
			models[model_i].eval()

		# collect preds
		confidence_values = []
		predictions = []
		labels = []
		iters = []

		# get test loaders
		for tl in test_loaders:
			iters.append(iter(tl))

		# get prediction for each batch
		for batch_idx in range(len(test_loaders[0])):
			ens_preds = []

			# ====== student ======
			# get the prediction for the batch
			for it,model in zip(iters,models):
				data,target = next(it)
				sm_out = F.softmax(model(data.float()),dim=1)

				if len(models) > 1:
					ens_preds.append(sm_out)

			# get the avg prediction over each model
			if len(models) > 1:
				sm_out = torch.stack(ens_preds).mean(dim=0)

			# parse the output for the prediction
			cs,ps = sm_out.max(dim=1)

			confidence_values.append(cs)
			predictions.append(ps)
			labels.append(target)
		
		# merge all
		confidence_values = torch.cat(confidence_values)
		predictions = torch.cat(predictions)
		labels = torch.cat(labels)
	
	print(f"\t\tF1: {f1_score(labels,predictions,average='macro')}")

	return confidence_values, predictions, labels


body_parts = ['torso','right_arm','left_arm','right_leg','left_leg']
users = [1,2,3,4,5,6,7,8]
seeds = [123,456,789]

logging_prefix = "dsads"

# iterate over all scenarios
for seed in seeds:
	print(f"Seed: {seed}")
	# collect users, models, loaders for ensemble
	train_users_list = []
	test_users_list = []

	bp_models = {i:[] for i in range(len(users))} # list of models for each loocv
	test_loaders = {i:[] for i in range(len(users))} # list of test loaders for each loocv
	for user_i,user in enumerate(users):
		print(f"\tUser: {user}")
		train_users = users[:user_i] + users[user_i+1:]
		test_users = [user]

		train_users_list.append(str(train_users))
		test_users_list.append(str(test_users))

		# individual bp eval
		for bp_i,bp in enumerate(body_parts):
			print(f"\t\tBody Part: {bp}")
			# create the datasets
			train_loader,val_loader,test_loader = load_dsads_person_dataset(128,train_users,test_users,[bp])
			test_loaders[user_i].append(test_loader)

			# init models
			model = DSADSNet(3)

			# load the model if already trained
			ckpt_path = os.path.join(PROJECT_ROOT,f"saved_data/checkpoints/{logging_prefix}/{bp}_{train_users}_seed{seed}.pth")
			model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
			bp_models[user_i].append(model)
			
			confidence_values, predictions, labels = get_model_outputs([model],[test_loader])
			logname = f"{logging_prefix}/{bp}_{train_users}_seed{seed}"
			cache_path = os.path.join(PROJECT_ROOT,"saved_data/cached_data/"+logname) + ".pt"
			path_items = logname.split("/")
			if  len(path_items) > 1:
				Path(os.path.join(PROJECT_ROOT,"saved_data/cached_data",*path_items[:-1])).mkdir(parents=True, exist_ok=True)
			
			torch.save(torch.stack([confidence_values, predictions, labels]),cache_path)

		# save ensemble
		print("\t\tEnsemble")
		confidence_values, predictions, labels = get_model_outputs(bp_models[user_i],test_loaders[user_i])
		logname = f"{logging_prefix}/ens_{train_users}_seed{seed}"
		cache_path = os.path.join(PROJECT_ROOT,"saved_data/cached_data/"+logname) + ".pt"
		path_items = logname.split("/")
		if  len(path_items) > 1:
			Path(os.path.join(PROJECT_ROOT,"saved_data/cached_data",*path_items[:-1])).mkdir(parents=True, exist_ok=True)
		
		torch.save(torch.stack([confidence_values, predictions, labels]),cache_path)