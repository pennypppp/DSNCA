import os
import sys
import torch
import random
import pickle
import time
import datetime

from torch.optim import Adam, lr_scheduler

# from capsule.modules import MECapsuleNet
from capsule.modules import *
from capsule.loss import me_loss
from capsule.evaluations import Meter
from capsule.data import data_split, sample_data, get_meta_data, Datasetcom, MyDatasetTranscom , MyDatasetDualcom ####################################3

from torchvision import transforms

from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle


#casme
# data_apex_frame_path = '/home/pp/FP_program/CAPSULE3d_dualca/datasets/casme_apex_3.csv'
# data_four_frames_path = '/home/pp/FP_program/CAPSULE3d_dualca/datasets/casme_four_apex_3.csv'
# data_root = '/home/pp/DATASET'
# data_root_3d = '/home/pp/DATASET/output3d5apex/casmeapex'

#smic
# data_apex_frame_path = '/home/pp/FP_program/CAPSULE3d_dualca/datasets/smic_apex_3.csv'
# data_four_frames_path = '/home/pp/FP_program/CAPSULE3d_dualca/datasets/smic_four_apex_3.csv'
# data_root = '/home/pp/DATASET'
# data_root_3d = '/home/pp/DATASET/output3d5apex/smicapex'

#samm
# data_apex_frame_path = '/home/pp/FP_program/CAPSULE3d_dualca/datasets/samm_apex_3.csv'
# data_four_frames_path = '/home/pp/FP_program/CAPSULE3d_dualca/datasets/samm_four_apex_3.csv'
# data_root = '/home/pp/DATASET'
# data_root_3d = '/home/pp/DATASET/output3d5apex/sammapex'

#composite
data_apex_frame_path = '/home/pp/FP_program/CAPSULE3d_dualca/datasets/composite_apex_3_withglass.csv'
data_four_frames_path = '/home/pp/FP_program/CAPSULE3d_dualca/datasets/composite_four_apex_3.csv'
data_root = '/home/pp/DATASET'
data_root_3d = '/home/pp/DATASET/output3d5apex'


batch_size =16   #batch_size =32
lr = 0.0001  #lr = 0.0001   lr = 0.0002  lr = 0.00005
lr_decay_value = 0.9
num_classes = 3
epochs = 30##################################
x_meter = Meter()
batches_scores = []

def load_me_data(data_root, data_root_3d,file_path, subject_out_idx, batch_size=16, num_workers=1,is_3d=True,is_fusion=False):
	df_train, df_val = data_split(file_path, subject_out_idx)
	df_four = pd.read_csv(data_four_frames_path)
	df_train_sampled = sample_data(df_train, df_four)  # The number of frames expands 13 timesrough 3d
	df_train_sampled = shuffle(df_train_sampled)
	train_paths, train_labels = get_meta_data(df_train_sampled) #get the apex path for training
	#print("train_paths",train_paths)
	#orginal code as belowing
	# train_transforms = transforms.Compose([transforms.Resize((234, 240)),
	#                                        transforms.RandomRotation(degrees=(-8, 8)),
	#                                        transforms.RandomHorizontalFlip(),
	#                                        transforms.ColorJitter(brightness=0.2, contrast=0.2,
	#                                                               saturation=0.2, hue=0.2),
	#                                        transforms.RandomCrop((224, 224)),
	#                                        transforms.ToTensor()])

	# in order to fit the crop img
	# by chenming as belowing
	if is_3d == True:
		if is_fusion == True:
			train_transforms = transforms.Compose([transforms.Resize((224, 224)),
												   transforms.RandomRotation(degrees=(-8, 8)),
												   transforms.RandomHorizontalFlip(),
												   transforms.ColorJitter(brightness=0.2, contrast=0.2,
																		  saturation=0.2, hue=0.2),
												   transforms.RandomCrop((224, 224)),
												   transforms.ToTensor()])

			train_dataset = MyDatasetDualcom(root=data_root,
										   data_root_3d=data_root_3d,
										   img_paths=train_paths,
										   img_labels=train_labels,
										   transform=None)

			val_transforms = transforms.Compose([transforms.Resize((234, 240)),
												 transforms.RandomRotation(degrees=(-8, 8)),
												 transforms.CenterCrop((224, 224)),
												 transforms.ToTensor()])

			val_paths, val_labels = get_meta_data(df_val)
			val_dataset = MyDatasetDualcom(root=data_root,
										 data_root_3d=data_root_3d,
										 img_paths=val_paths,
										 img_labels=val_labels,
										 transform=None)

			train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
													   batch_size=batch_size,
													   num_workers=num_workers,
													   shuffle=True)

			val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
													 batch_size=batch_size,
													 num_workers=num_workers,
													 shuffle=False)


		####### RGB  transform  in 3D concat
		else:
			train_transforms = transforms.Compose([transforms.Resize((224, 224)),
												   transforms.RandomRotation(degrees=(-8, 8)),
												   transforms.RandomHorizontalFlip(),
												   transforms.ColorJitter(brightness=0.2, contrast=0.2,
																		  saturation=0.2, hue=0.2),
												   transforms.RandomCrop((224, 224)),
												   transforms.ToTensor()])

			train_dataset = MyDatasetTranscom(root=data_root,
									data_root_3d=data_root_3d,
									img_paths=train_paths,
									img_labels=train_labels,
									transform= None)

			val_transforms = transforms.Compose([transforms.Resize((234, 240)),
												 transforms.RandomRotation(degrees=(-8, 8)),
												 transforms.CenterCrop((224, 224)),
												 transforms.ToTensor()])


			val_paths, val_labels = get_meta_data(df_val)
			val_dataset = MyDatasetTranscom(root=data_root,
								  data_root_3d=data_root_3d,
								  img_paths=val_paths,
								  img_labels=val_labels,
								  transform=None)
			train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
													   batch_size=batch_size,
													   num_workers=num_workers,
													   shuffle=True)

			val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
													 batch_size=batch_size,
													 num_workers=num_workers,
													 shuffle=False)
		############ No transform  in 3D concat
		# train_transforms = transforms.Compose([transforms.Resize((224, 224)),
		# 									   transforms.RandomRotation(degrees=(-8, 8)),
		# 									   transforms.RandomHorizontalFlip(),
		# 									   transforms.ColorJitter(brightness=0.2, contrast=0.2,
		# 															  saturation=0.2, hue=0.2),
		# 									   transforms.RandomCrop((224, 224)),
		# 									   transforms.ToTensor()])
		#
		# train_dataset = MyDataset(root=data_root,
		# 						data_root_3d=data_root_3d,
		# 						img_paths=train_paths,
		# 						img_labels=train_labels,
		# 						transform= None)    ############################# orig transform=train_transforms
		#
		#
		# val_transforms = transforms.Compose([transforms.Resize((234, 240)),
		# 									 transforms.RandomRotation(degrees=(-8, 8)),
		# 									 transforms.CenterCrop((224, 224)),
		# 									 transforms.ToTensor()])
		#
		# # val_transforms = transforms.Compose([transforms.Resize((224, 224)),
		# #                                      # transforms.RandomRotation(degrees=(-8, 8)),
		# #                                      # transforms.CenterCrop((224, 224)),
		# #                                      transforms.ToTensor()])
		#
		#
		# val_paths, val_labels = get_meta_data(df_val)
		# val_dataset = MyDataset(root=data_root,
		# 					  data_root_3d=data_root_3d,
		# 					  img_paths=val_paths,
		# 					  img_labels=val_labels,
		# 					  transform=None)    ############################# transform=val_transforms
		#
		# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
		# 										   batch_size=batch_size,
		# 										   num_workers=num_workers,
		# 										   shuffle=True)
		#
		# val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
		# 										 batch_size=batch_size,
		# 										 num_workers=num_workers,
		# 										 shuffle=False)
	else :
		train_transforms = transforms.Compose([transforms.Resize((224, 224)),
											   transforms.RandomRotation(degrees=(-8, 8)),
											   transforms.RandomHorizontalFlip(),
											   transforms.ColorJitter(brightness=0.2, contrast=0.2,
																	  saturation=0.2, hue=0.2),
											   transforms.RandomCrop((224, 224)),
											   transforms.ToTensor()])

		train_dataset = Datasetcom(root=data_root,
								  data_root_3d=data_root_3d,
								  img_paths=train_paths,
								  img_labels=train_labels,
								  transform=train_transforms )  ############################# orig transform=train_transforms

		val_transforms = transforms.Compose([transforms.Resize((234, 240)),
											 transforms.RandomRotation(degrees=(-8, 8)),
											 transforms.CenterCrop((224, 224)),
											 transforms.ToTensor()])

		# val_transforms = transforms.Compose([transforms.Resize((224, 224)),
		#                                      # transforms.RandomRotation(degrees=(-8, 8)),
		#                                      # transforms.CenterCrop((224, 224)),
		#                                      transforms.ToTensor()])

		val_paths, val_labels = get_meta_data(df_val)
		val_dataset = Datasetcom(root=data_root,
								data_root_3d=data_root_3d,
								img_paths=val_paths,
								img_labels=val_labels,
								transform=val_transforms)  ############################# transform=val_transforms

		train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
												   batch_size=batch_size,
												   num_workers=num_workers,
												   shuffle=True)

		val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
												 batch_size=batch_size,
												 num_workers=num_workers,
												 shuffle=False)

	return train_loader, val_loader


def on_epoch(model, optimizer, lr_decay, train_loader, test_loader, epoch):
	model.train()
	#lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
	train_loss = 0.0
	correct = 0.
	meter = Meter()

	steps = len(train_loader.dataset) // batch_size + 1
	with tqdm(total=steps) as progress_bar:
		for i, (x, y) in enumerate(train_loader):  # batch training
			y = torch.zeros(y.size(0), num_classes).scatter_(1, y.view(-1, 1),
			                                                 1.)  # change to one-hot coding
			x, y = x.cuda(), y.cuda()  # convert input data to GPU Variable

			optimizer.zero_grad()  # set gradients of optimizer to zero
			y_pred = model(x, y)  # forward
			loss = me_loss(y, y_pred)  # compute loss
			loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
			train_loss += loss.item() * x.size(0)  # record the batch loss
			optimizer.step()  # update the trainable parameters with computed gradients

			y_pred = y_pred.data.max(1)[1]    ####max（1）return the max of the columm   max[1] return the index of the max
			y_true = y.data.max(1)[1]

			meter.add(y_true.cpu().numpy(), y_pred.cpu().numpy())
			correct += y_pred.eq(y_true).cpu().sum()

			progress_bar.set_postfix(loss=loss.item(), correct=correct)
			progress_bar.update(1)

		train_loss /= float(len(train_loader.dataset))
		train_acc = float(correct.item()) / float(len(train_loader.dataset))
		scores = meter.value()
		meter.reset()
		print('Training UAR: %.4f' % (scores[0].mean()), scores[0])
		print('Training UF1: %.4f' % (scores[1].mean()), scores[0].mean())
		print(type(scores[0]))
		#log_f.write('\nSubject %2d has the Training UAR:%.5f,\tTraining UARmean:%.5f,\tTraining UF1:%.5f,\tTraining UF1mean:%.5f\n'%(i, scores[0], scores[0].mean(), scores[1], scores[1].mean()))
		# log_f.write('\nSubject %2d has the Training UAR,\tTraining UARmean:%.5f,\tTraining UF1:%.5f,\tTraining UF1mean:%.5f\n'%(i, join(scores[0]), scores[0].mean(), scores[1], scores[1].mean()))

	lr_decay.step()




	correct = 0.
	test_loss = 0.

	model.eval()
	for i, (x, y) in enumerate(test_loader):  # batch training
		y = torch.zeros(y.size(0), num_classes).scatter_(1, y.view(-1, 1),
		                                                 1.)  # change to one-hot coding
		x, y = x.cuda(), y.cuda()  # convert input data to GPU Variable

		y_pred = model(x, y)  # forward
		loss = me_loss(y, y_pred)  # compute loss
		test_loss += loss.item() * x.size(0)  # record the batch loss

		y_pred = y_pred.data.max(1)[1]
		y_true = y.data.max(1)[1]

		meter.add(y_true.cpu().numpy(), y_pred.cpu().numpy())
		correct += y_pred.eq(y_true).cpu().sum()

		if (epoch + 1) % 10 == 0 and i % steps == 0:
			print('y_true\n', y_true[:30])
			print('y_pred\n', y_pred[:30])

	print('y_true', y.sum(dim=0))
	scores = meter.value()
	print('Testing UAR: %.4f' % (scores[0].mean()), scores[0])
	print('Testing UF1: %.4f' % (scores[1].mean()), scores[1])
	# log_f.write('\nSubject {} has the Testing UAR:{:.4f},\tTesting UARmean:{:.4f},\tTesting UF1:{:.4f},\tTesting UF1mean:{:.4f}\n'.format(\
	# 			i, scores[0], scores[0].mean(), scores[0].mean(), scores[1].mean()))
	test_loss /= float(len(test_loader.dataset))
	test_acc = float(correct.item()) / float(len(test_loader.dataset))
	return train_loss, train_acc, test_loss, test_acc, meter


def train_eval(subject_out_idx):
	best_val_uf1 = 0.0
	best_val_uar = 0.0

	# Model & others  signle CapsuleNet
	model = MECapsuleNet(input_size=(39, 224, 224), classes=num_classes, routings=3, is_freeze=True)     ##################3
	# model = MECapsuleNetDualCA(classes=num_classes, routings=3, is_freeze=False)
	model.cuda()
	# model.load_state_dict(torch.load(fer_weight_path)['model'])
	optimizer = Adam(model.parameters(), lr=lr)
	lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_value)

	for epoch in range(epochs):
		train_loader, test_loader = load_me_data(data_root, data_root_3d,data_apex_frame_path,
		                                         subject_out_idx=subject_out_idx,
		                                         batch_size=batch_size)
		train_loss, train_acc, test_loss, test_acc, meter = on_epoch(model, optimizer, lr_decay,
		                                                              train_loader, test_loader,
		                                                              epoch)

		print("==> Subject out: %02d - Epoch %02d: loss=%.5f, train_acc=%.5f, val_loss=%.5f, "
		      "val_acc=%.4f"% (subject_out_idx, epoch, train_loss, train_acc,test_loss, test_acc))
		log_f.write('\nSubject out: %02d - Epoch %02d: loss=%.5f, train_acc=%.5f, val_loss=%.5f, val_acc=%.5f}\n'%( \
						subject_out_idx, epoch, train_loss, train_acc,test_loss, test_acc))
		scores = meter.value()
		if scores[1].mean() >= best_val_uf1:
			best_val_uar = scores[0].mean()
			best_val_uf1 = scores[1].mean()
			Y_true = meter.Y_true.copy()
			Y_pred = meter.Y_pred.copy()

	x_meter.add(Y_true, Y_pred, verbose=True)

	return best_val_uar, best_val_uf1


if __name__ == '__main__':

	# random.seed(3)
	# torch.manual_seed(3)
	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = False

	runPath = os.getcwd()
	logPath = os.path.join(runPath, 'log', 'runlog.txt')
	now = datetime.datetime.now()
	log_f = open(logPath, 'a')
	log_f.write('{}\n'.format(now))
	log_f.write('-' * 80 + '\n')
	log_f.write('-' * 80 + '\n')
	log_f.write('Results:\n')
	time_s = time.time()

	for i in range(28):         ##### casme has 24 subjects , smic has subject 16, samm has subject28  com has 68； com with glass 28

		scores = train_eval(subject_out_idx=i)
		batches_scores.append(scores)
		x_scores = x_meter.value()

		print('final uar', x_scores[0], x_scores[0].mean())
		print('final uf1', x_scores[1], x_scores[1].mean())
		print('---- NEXT ---- \n\n')

		# log_f.write('\nSubject {} has the final uar:{:.5f},\tfinal uarmean:{:.5f},\tfinal uf1:{:.5f},\tfinal uf1mean:{:.5f}\n'.format(i, x_scores[0], x_scores[0].mean(),x_scores[1], x_scores[1].mean()))

		log_f.write('-' * 80 + '\n')
		log_f.write('-' * 80 + '\n')
		log_f.write('\n')


		with open('scores_capsule_resnet_sampled_fer_freeze.pkl', 'wb') as file:
			data = dict(meter=x_meter, batches_scores=batches_scores)
			pickle.dump(data, file)
	log_f.close()