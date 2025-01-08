import pickle
from capsule.evaluations import Meter
import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import sklearn.metrics as sm
import matplotlib.pyplot as plt

with open('scores_capsule_resnet_sampled_fer_freeze.pkl', 'rb') as f:
	data = pickle.load(f)

# with open('/home/pp/FP_program/CAPSULE3d_dualca/outputs/scores_capsule_vgg_sampled_freeze.pkl', 'rb') as f:
# 	data = pickle.load(f)
scores = data['meter'].value()
# print("scores",scores)

Y_pred = data['meter'].Y_pred
Y_true = data['meter'].Y_true
matrixes = sm.confusion_matrix(Y_true,Y_pred)

corrects=matrixes .diagonal(offset=0)#抽取对角线的每种分类的识别正确个数
per_kinds=matrixes .sum(axis=1)#抽取每个分类数据总的测试条数

print("recall",scores[0])
print("F1",scores[1])

print(matrixes)
print("混淆矩阵总元素个数：{0}".format(int(np.sum(matrixes))))
print("每种情感总个数：", per_kinds)
print("每种情感预测正确的个数：", corrects)
print("每种情感的识别准确率为：{0}".format([rate * 100 for rate in corrects / per_kinds]))
# print(confusion)

# 绘制混淆矩阵
Emotion_kinds = 3 # 这个数值是具体的分类数，大家可以自行修改
labels = ['Neg', 'Pos', 'Sur']  # 每种类别的标签

# 显示数据
plt.imshow(matrixes , cmap=plt.cm.Blues)

# 在图中标注数量/概率信息
thresh = matrixes.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
for x in range(Emotion_kinds):
	for y in range(Emotion_kinds):
		# 注意这里的matrix[y, x]不是matrix[x, y]
		info = int(matrixes[y, x])
		plt.text(x, y, info,
				 verticalalignment='center',
				 horizontalalignment='center',
				 color="white" if info > thresh else "black")

plt.tight_layout()  # 保证图不重叠
plt.yticks(range(Emotion_kinds), labels)
plt.xticks(range(Emotion_kinds), labels, rotation=45)  # X轴字体倾斜45°
plt.show()
plt.close()




# me = pd.read_csv('/home/pp/FP_program/CAPSULE3d_dualca/datasets/casme_apex_3.csv')
# data = 'casme'
# subject = '24'
# print(data, subject)

# me = pd.read_csv('/home/pp/FP_program/CAPSULE3d_dualca/datasets/smic_apex_3.csv')
# data = 'smic'
# subject = '16'

# me = pd.read_csv('/home/pp/FP_program/CAPSULE3d_dualca/datasets/samm_apex_3.csv')
# data = 'samm'
# subject = '28'
# print(data, subject)

# me = pd.read_csv('/home/pp/FP_program/CAPSULE3d_dualca/datasets/composite_apex_3.csv')
# data = 'com'
# subject = '68'



# casme2 = {}
# smic = {}
# samm = {}

# confusion = multilabel_confusion_matrix(Y_true, Y_pred)   # 多分类问题转换为二分类

# for i in range(3):
# 	conf = confusion[i]
# 	tp = conf[1][1]
# 	tn = conf[0][0]
# 	fp = conf[0][1]
# 	fn = conf[1][0]
# 	precision = tp / (tp + fp)
# 	recall = tp / (tp + fn)
# 	f1 = 2/(1/precision + 1/recall)
# 	acc = (tp + tn) / (tp + fp + fn + tn)
#
# 	print('category:{}'.format(i))
# 	print('precision:{}'.format(precision))
# 	print('recall:{}'.format(recall))
# 	print('f1:{}'.format(f1))
# 	print('accuracy:{}'.format(acc))
#
#
#
# print(confusion)
#
# with open('result_log.csv', 'w') as f:
# 	for i in range(len(Y_true)):
# 		y_true = Y_true[i]
# 		y_pred = Y_pred[i]
# 		#print('y true', 'y_pred')
# 		#print(y_pred, y_true)
"""
		if data != me.iloc[i]['data'] or subject != me.iloc[i]['Subject']:
			print(me.iloc[i]['data'], me.iloc[i]['Subject'])
			data = me.iloc[i]['data']
			subject = me.iloc[i]['Subject']

		# log_str = str(me.iloc[i]['clip']), Y_true[i], Y_pred[i] + '\n'
		log_str = me.iloc[i]['clip']+ ','+str(Y_true[i])+','+str(Y_pred[i])+ '\n'
		# print("log_str",log_str)
		# print("Y_true",Y_true[i])
		# print("Y_pred", Y_pred[i])
		# f.write(me.iloc[i]['clip'], Y_true[i], Y_pred[i])
		f.write(str(log_str))
"""