import pickle
from capsule.evaluations import Meter
import pandas as pd

with open('scores_capsule_resnet_sampled_fer_freeze.pkl', 'rb') as f:
	data = pickle.load(f)

scores = data['meter'].value()
print("scores",scores)
print(scores[0])
print(scores[1])

Y_pred = data['meter'].Y_pred
Y_true = data['meter'].Y_true

me = pd.read_csv('/home/pp/program-pp/CapsuleNet-for-Micro-Expression-Recognition1/me_recognition-master/datasets/MEGC2019/data_apex_samm_rename.csv')
data = 'smic'
subject = '20'
# print(data, subject)

casme2 = {}
smic = {}
samm = {}

with open('result_log.csv', 'w') as f:
	for i in range(len(Y_true)):
		y_true = Y_true[i]

		if data != me.iloc[i]['data'] or subject != me.iloc[i]['subject']:
			print(me.iloc[i]['data'], me.iloc[i]['subject'])
			data = me.iloc[i]['data']
			subject = me.iloc[i]['subject']

		# log_str = str(me.iloc[i]['clip']), Y_true[i], Y_pred[i] + '\n'
		log_str = me.iloc[i]['clip']+ ','+str(Y_true[i])+','+str(Y_pred[i])+ '\n'
		# print("log_str",log_str)
		# print("Y_true",Y_true[i])
		# print("Y_pred", Y_pred[i])
		# f.write(me.iloc[i]['clip'], Y_true[i], Y_pred[i])
		f.write(str(log_str))
