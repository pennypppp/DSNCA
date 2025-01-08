#数据文件：mnist.pkl，大小约54M。
#文件读取，位置
#https://blog.csdn.net/u012132349/article/details/87357438  pkl文件用法
# import pandas as pd
# network=pd.read_pickle('scores_capsule_resnet_sampled_fer_freeze.pkl')
# type(network) #类型：字典
# network.keys() #关键字['train_label', 'train_img', 'test_img', 'test_label']
# #训练数据形状,(60000, 784),6万个样本，每个样本由784个数据组成(1·28·28)
# print(network.keys())
#ori MECapsuleNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from capsule.modules.capsule_layers import PrimaryCapsule, MECapsule
from capsule.modules.activations import squash
from torchvision import models
import torch.nn as nn
import torch, glob, cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt



#2D input  3 channels
class ResNetLayers2D(nn.Module):
	def __init__(self, is_freeze=True):
		super(ResNetLayers2D, self).__init__()
		self.model = models.resnet18(pretrained=True)
		delattr(self.model, 'layer4')
		delattr(self.model, 'avgpool')
		delattr(self.model, 'fc')

		if is_freeze:
			for index, p in enumerate(self.model.parameters()):
				if index == 15:
					break
				p.requires_grad = False

	def forward(self, x):
		output = self.model.conv1(x)             ##########################################################
		output = self.model.bn1(output)
		output = self.model.relu(output)
		output = self.model.layer1(output)
		output = self.model.layer2(output)
		output = self.model.layer3(output)
		return output



#2D&3D backbone
backbone = { 'resnet2d': ResNetLayers2D}

class MECapsuleNet(nn.Module):
	"""
	A Capsule Network on Micro-expression.
	:param input_size: data size = [channels, width, height]
	:param classes: number of classes
	:param routings: number of routing iterations
	Shape:
		- Input: (batch, channels, width, height), optional (batch, classes) .
		- Output:((batch, classes), (batch, channels, width, height))
	"""

	def __init__(self, input_size, classes, routings, conv_name='resnet2d', is_freeze=True):
		super(MECapsuleNet, self).__init__()
		self.input_size = input_size
		self.classes = classes
		self.routings = routings
		#self.ca = CoordAttention()   ################################################
		self.conv = backbone[conv_name](is_freeze)
		# print(self.conv)
		# print(backbone['resnet'].conv1)       #####################################################################
		self.conv1 = nn.Conv2d(256, 256, kernel_size=9, stride=1, padding=0)

		self.primarycaps = PrimaryCapsule(256, 32 * 8, 8, kernel_size=9, stride=2, padding=0)

		self.digitcaps = MECapsule(in_num_caps=32 * 6 * 6,
		                           in_dim_caps=8,
		                           out_num_caps=self.classes,
		                           out_dim_caps=16,
		                           routings=routings)

		self.relu = nn.ReLU()

	def forward(self, x, y=None):
		x = self.conv(x)
		#x = self.ca(x) * x    #########################################################3
		x = self.relu(self.conv1(x))


		x = self.primarycaps(x)
		x = self.digitcaps(x)
		length = x.norm(dim=-1)
		return length



def preict_one_img(img_path, model_path):
	net = MECapsuleNet(input_size=(3,224,224), classes=3, routings=3,)
	net.cuda()
	# model = MECapsuleNet(input_size=(42, 224, 224), classes=num_classes, routings=3, is_freeze=False)
	net.load_state_dict(torch.load(model_path))
	img = cv2.imread(img_path)
	img = cv2.resize(img, (224, 224))
	tran = transforms.ToTensor()
	img = tran(img)
	img = img.to(device)
	img = img.view(1, 3, 224, 224)
	out1 = net(img)
	out1 = F.softmax(out1, dim=1)
	proba, class_ind = torch.max(out1, 1)

	proba = float(proba)
	class_ind = int(class_ind)
	# print(proba, class_ind)
	img = img.cpu().numpy().squeeze(0)
	# print(img.shape)
	new_img = np.transpose(img, (1,2,0))
	print("the predict is %s . prob is %s" % (classes[class_ind], round(proba, 3)))  # round(proba, 3)
	plt.imshow(new_img)
	plt.title("the predict is %s . prob is %s" % (classes[class_ind], round(proba, 3)))   # round(proba, 3)保留三位小数
	plt.show()








if __name__ == '__main__':
    classes = ["pos", "neg","sur"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = r"/home/pp/FP_program/CAPSULE3d_dualca/trained/img69.jpg"
    model_path = r"/home/pp/FP_program/CAPSULE3d_dualca/trained/model_state.pt"
    preict_one_img(img_path, model_path)


