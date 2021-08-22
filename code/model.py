import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import PointNetEncoder

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def normalized_columns_initializer(weights,std=1.0):
	setup_seed(666)
	out = torch.randn(weights.size())
	out *= std/torch.sqrt(out.pow(2).sum(1,keepdim=True).expand_as(out))
	return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear')!=-1:
        torch.nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)

def command2action(command_ids, paras, terminal, step, rot_angle):
	'''
	Original by wuhuikai in https://github.com/wuhuikai/TF-A2RL
	
	paras[i,0] angle
	paras[i,1,2,3] axis[0:x,1:y,2:z]
	'''	
	# paras: [rad_x, rad_y, rad_z]
	
	batch_size = len(command_ids)
	
	for i in range(batch_size):
		if terminal[i] == 0:	
			if command_ids[i] == 0:   # rad_x + 1
				paras[i, 0] = rot_angle
				paras[i, 1] = 1
				
			elif command_ids[i] == 1: # rad_x - 10
				paras[i, 0] = (-rot_angle)
				paras[i, 1] = 1
				
			elif command_ids[i] == 2: # rad_y + 10
				paras[i, 0] = rot_angle
				paras[i, 2] = 1
				
			elif command_ids[i] == 3: # rad_y - 10
				paras[i, 0] = (-rot_angle)
				paras[i, 2] = 1

			elif command_ids[i] == 4:
				terminal[i]	= step + 1

	return paras, terminal

class ActorCritic(nn.Module):
	def __init__(self, action_num=5):
		super(ActorCritic, self).__init__()

		self.encoder = PointNetEncoder(global_feat=True, feature_transform=False, channel=3)
		self.fc0 = nn.Linear(1024, 1024)
		self.bn0 = nn.BatchNorm1d(1024)
		self.fc1 = nn.Linear(1024, 512)
		self.bn1 = nn.BatchNorm1d(512)
		self.fc2 = nn.Linear(512, 256)
		self.bn2 = nn.BatchNorm1d(256)
		self.fc3 = nn.Linear(256, 128)
		self.bn3 = nn.BatchNorm1d(128) 
  
		self.lstm = nn.LSTMCell(128, 128)
		#self.gru = nn.GRUCell(128,128)

		self.critic_linear = nn.Linear(128, 1)
		self.actor_linear = nn.Linear(128, action_num)  #5
		self.apply(weights_init)
		self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data,0.01)
		self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data,1.0)
		self.actor_linear.bias.data.fill_(0)
		self.critic_linear.bias.data.fill_(0)

		self.lstm.bias_ih.data.fill_(0)
		self.lstm.bias_hh.data.fill_(0)
		
	def forward(self, inputs):
		xyz, (hx,cx) = inputs  #xyz.size() : [2048,3]
		# xyz = xyz.unsqueeze(0)  #xyz.size() : [1,2048,3]
		# print(xyz.size())
		# print((xyz.transpose(2, 1)).size())
		B = xyz.size()[0]
		fea = self.encoder(xyz.transpose(2, 1))
		if B == 1:
			fea = F.relu((self.fc0(fea)))
			x = F.relu((self.fc1(fea)))
			x = F.relu((self.fc2(x)))
			x = F.relu((self.fc3(x)))
		else:
			fea = F.relu(self.bn0(self.fc0(fea)))
			x = F.relu(self.bn1(self.fc1(fea)))
			x = F.relu(self.bn2(self.fc2(x)))
			x = F.relu(self.bn3(self.fc3(x)))
		# hx_cx = self.gru(x, (hx_cx))
		hx, cx = self.lstm(x, (hx, cx))
		value = self.critic_linear(hx)
		logit = self.actor_linear(hx)
		return value, logit, (hx,cx)