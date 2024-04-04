import  torch
import torch.nn as nn
import torch.nn.functional as F


class OneHiddenLayer(nn.Module):
	def __init__(self,d_int, H, non_linearity = torch.nn.SiLU(), bias=True):
		super(OneHiddenLayer,self).__init__()
		self.linear1 = torch.nn.Linear(d_int, H,bias=bias)
		self.linear2 = torch.nn.Linear( 1,H, bias=False)
		self.non_linearity = non_linearity
		self.d_int = d_int
		self.H = H

	def weights_init(self,center, std):
		self.linear1.weights_init(center,std)
		self.linear2.weights_init(center,std)

	def forward(self, x):
		x = self.non_linearity(self.linear1(x))
		return torch.einsum('hi,nh->ni',self.linear2.weight,x)/self.H
		#return self.model(x)/self.H