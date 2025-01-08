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

# 对单个图像可视化
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
from torchvision.models import resnet50
import cv2
import numpy as np
import os


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



# def preict_one_img(img_path, model_path):
# 	net = MECapsuleNet(input_size=(3,224,224), classes=3, routings=3,)
# 	net.cuda()
# 	# model = MECapsuleNet(input_size=(42, 224, 224), classes=num_classes, routings=3, is_freeze=False)
# 	net.load_state_dict(torch.load(model_path))
# 	img = cv2.imread(img_path)
# 	img = cv2.resize(img, (224, 224))
# 	tran = transforms.ToTensor()
# 	img = tran(img)
# 	img = img.to(device)
# 	img = img.view(1, 3, 224, 224)
# 	out1 = net(img)
# 	out1 = F.softmax(out1, dim=1)
# 	proba, class_ind = torch.max(out1, 1)
#
# 	proba = float(proba)
# 	class_ind = int(class_ind)
# 	# print(proba, class_ind)
# 	img = img.cpu().numpy().squeeze(0)
# 	# print(img.shape)
# 	new_img = np.transpose(img, (1,2,0))
# 	print("the predict is %s . prob is %s" % (classes[class_ind], round(proba, 3)))  # round(proba, 3)
# 	plt.imshow(new_img)
# 	plt.title("the predict is %s . prob is %s" % (classes[class_ind], round(proba, 3)))   # round(proba, 3)保留三位小数
# 	plt.show()



# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
img_path = r"/home/pp/FP_program/CAPSULE3d_dualca/trained/img69.jpg"
model_path = r"/home/pp/FP_program/CAPSULE3d_dualca/trained/model_state.pt"
# 1.加载模型
net = MECapsuleNet(input_size=(3, 224, 224), classes=3, routings=3, )
net.cuda()
# model = MECapsuleNet(input_size=(42, 224, 224), classes=num_classes, routings=3, is_freeze=False)
net.load_state_dict(torch.load(model_path))
# 2.选择目标层
# target_layer = model.layer4[-1]
target_layer = [net.layer3]
# 3. 构建输入图像的Tensor形式
image_path = "/home/pp/FP_program/CAPSULE3d_dualca/trained/img69.jpg"
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]   # 1是读取rgb
rgb_img = np.float32(rgb_img) / 255

# preprocess_image作用：归一化图像，并转成tensor
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])   # torch.Size([1, 3, 224, 224])
# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
# 4.初始化GradCAM，包括模型，目标层以及是否使用cuda
# cam = GradCAM(model=model, target_layer=target_layer, use_cuda=False)
cam = GradCAM(model=net, target_layers=target_layer, use_cuda=False)

# If target_category is None, the highest scoring category
# will be used for every image in the batch.
# target_category can also be an integer, or a list of different integers
# for every image in the batch.
# 5.选定目标类别，如果不设置，则默认为分数最高的那一类
target_category = None # 281

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
# 6. 计算cam
# grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # [batch, 224,224]
grayscale_cam = cam(input_tensor=input_tensor)
# In this example grayscale_cam has only one image in the batch:
# 7.展示热力图并保存, grayscale_cam是一个batch的结果，只能选择一张进行展示
grayscale_cam = grayscale_cam[0]
visualization = show_cam_on_image(rgb_img, grayscale_cam)  # (224, 224, 3)
cv2.imwrite(f'person.jpg', visualization)






# if __name__ == '__main__':
#     classes = ["pos", "neg","sur"]
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     img_path = r"/home/pp/FP_program/CAPSULE3d_dualca/trained/img69.jpg"
#     model_path = r"/home/pp/FP_program/CAPSULE3d_dualca/trained/model_state.pt"
#     preict_one_img(img_path, model_path)


