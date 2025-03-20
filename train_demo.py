import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
import math
import torch.nn.functional as F

import model
import config

torch.backends.cudnn.benchmark = True

def setup_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

setup_seed(1)

cuda = config.cuda
print('cuda:',cuda)
if cuda != '-1':
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = cuda

rs = config.rs
rd = config.rd

# data preparation phase and should be replaced by real-world training and testing data
train_total = 64
test_total = 16
slice_num = 10

train_data = torch.rand(train_total, slice_num, 90, 90)  # subjects * slices * node * node
train_data = train_data.view(-1, 90, 90)
train_data = train_data[torch.randperm(train_data.size(0))]  # shuffle train data
train_data = train_data.view(10, 64, 1, 90, 90)  # batch number * batch size * 1 * node * node

train_label_s = torch.randint(0, 2, (train_total,1))		# train gender label
train_label_s = train_label_s.expand(train_total, slice_num).reshape(10, 64)
train_label_d = torch.randint(0, 2, (train_total,1))		# train disease label
train_label_d = train_label_d.expand(train_total, slice_num).reshape(10, 64)

test_data = torch.rand(test_total, slice_num, 90, 90)  # subjects * slices * node * node
test_data = test_data.unsqueeze(2)

test_label_s = torch.randint(0, 2, (test_total,1))		# test gender label
test_label_s = test_label_s.expand(test_total, slice_num)
test_label_d = torch.randint(0, 2, (test_total,1))		# test disease label
test_label_d = test_label_d.expand(test_total, slice_num)

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()

# net = model.MT_PCN(32, 32, 32, 64, 256)
net = model.MT_HPN(32, 32, 32, 64, 256)

net.apply(model.weights_init)
total = sum([param.nelement() for param in net.parameters()])
print("Number of parameter: %.3fM" % (total/1e6))
if config.cuda != '-1':
	net = nn.DataParallel(net)
	net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.00001,weight_decay=0)

acc_best= 0
acc_best_s= 0
acc_best_d= 0
starttime = time.time()

for epoch in range(60):
	train_correct_sex, train_correct_DX = 0, 0
	running_loss = 0.0
	running_loss_s = 0.0
	running_loss_d = 0.0

	net.train()
	for i in range(train_label_s.size(0)):
		inputs = train_data[i]
		labels_s = train_label_s[i]
		labels_d = train_label_d[i]
		if config.cuda != '-1':
			inputs=inputs.cuda()
			labels_s=labels_s.cuda()
			labels_d=labels_d.cuda()
		optimizer.zero_grad()
		output_s, output_d = net(inputs)
		loss_s = criterion1(output_s, labels_s)
		loss_d = criterion2(output_d, labels_d)
		loss = rs * loss_s + rd * loss_d
		loss.backward()
		optimizer.step()
		running_loss += loss.item() / inputs.size()[0]
		running_loss_s += loss_s.item() / inputs.size()[0]
		running_loss_d += loss_d.item() / inputs.size()[0]

		_, predicted_s = torch.max(output_s.data, 1)
		train_correct_sex += (predicted_s == train_label_s[i]).sum().item()
		_, predicted_d = torch.max(output_d.data, 1)
		train_correct_DX += (predicted_d == train_label_d[i]).sum().item()

	net.eval()
	correct_sex_s, correct_DX_s, total_s = 0, 0, 0
	prelist_s = []
	truelist_s = []
	prelist_d = []
	truelist_d = []
	test_loss = 0.0
	test_loss_s = 0.0
	test_loss_d = 0.0
	with torch.no_grad():
		for i in range(train_label_s.size(0)):
			inputs = train_data[i]
			labels_s = train_label_s[i]
			labels_d = train_label_d[i]
			if config.cuda != '-1':
				inputs = inputs.cuda()
				labels_s = labels_s.cuda()
				labels_d = labels_d.cuda()
			output_s, output_d = net(inputs)
			loss_s = criterion1(output_s, labels_s)
			loss_d = criterion2(output_d, labels_d)
			loss = rs * loss_s + rd * loss_d
			test_loss += loss.item() / inputs.size()[0]
			test_loss_s += loss_s.item() / inputs.size()[0]
			test_loss_d += loss_d.item() / inputs.size()[0]

			# Compute the gender accuracy for each person
			output_s = F.softmax(output_s, dim=1)
			output_s = torch.mean(output_s, 0)
			_, predicted_s = torch.max(output_s.data, 0)
			prelist_s.append(predicted_s.cpu())
			truelist_s.append(labels_s.cpu().numpy()[0][0])
			if predicted_s == labels_s[0][0].item():
				correct_sex_s += 1

			# Compute the disease accuracy for each person
			output_d = F.softmax(output_d, dim=1)
			output_d = torch.mean(output_d, 0)
			_, predicted_d = torch.max(output_d.data, 0)
			prelist_d.append(predicted_d.cpu())
			truelist_d.append(labels_d.cpu().numpy()[0][1])
			if predicted_d == labels_d[0][1].item():
				correct_DX_s += 1

	ltime = time.time()-starttime
	if (acc_best_s < correct_sex_s / test_total):
		acc_best_s = correct_sex_s / test_total
		print('Best gender acc')
		# torch.save(net.state_dict(), './save/' + config.modelname +'.pkl')
	if (acc_best_d < correct_DX_s / test_total):
		acc_best_d = correct_DX_s / test_total
		print('Best disease acc')
		# torch.save(net.state_dict(), './save/' + config.modelname +'.pkl')
	print('[%d]loss:%.3f loss_s:%.4f loss_d:%.4f acc_s:%.4f acc_d:%.4f testloss:%.4f testloss_s:%.4f testloss_d:%.4f acc_s_s:%.4f acc_d_s:%.4f time:%.2fm' %
		(epoch + 1, running_loss, running_loss_s, running_loss_d, train_correct_sex / train_total * slice_num,
		 train_correct_DX / train_total * slice_num, test_loss, test_loss_s, test_loss_d, correct_sex_s / test_total,
		 correct_DX_s / test_total, ltime / 60))

	if math.isnan(running_loss):
		print('break')
		break

print('Best_Acc_s:%.4f Best_Acc_d:%.4f'%(acc_best_s, acc_best_d))
