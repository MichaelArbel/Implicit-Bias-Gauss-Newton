import torch

from torch.utils import data


class TeacherDataset(torch.utils.data.Dataset):

	def __init__(self,network, N_samples, dtype, device, normalize=False):
		D = network.d_int
		self.device = device

		
		self.X = torch.normal(mean= torch.zeros(N_samples,D,dtype=dtype,device=device),std=1.)

		if normalize:			
			inv_norm = 1./tr.norm(self.X,dim=1)
			self.X = tr.einsum('nd,n->nd',self.X,inv_norm)

		self.total_size = N_samples
		self.network = network

		with torch.no_grad():
			self.Y = self.network(self.X)

	def __len__(self):
		return self.total_size 
	def __getitem__(self,index):
		return self.X[index,:],self.Y[index,:]

