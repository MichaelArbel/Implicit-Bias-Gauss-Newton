
import torch
from torch.optim import Optimizer
from functools import partial

def chunked_vmap(func, chunk,matrix):
	if chunk==1:
		#par_func = partial(func, retain_graph=False)
		return torch.vmap(func)(matrix)
	else:
		out = None
		sub_matrices = torch.chunk(matrix,chunk,axis=0)
		final_index = len(sub_matrices)-1
		for i, chunck in enumerate(sub_matrices):
			#par_func = partial(func, retain_graph=retain_graph)
			tmp = torch.vmap(func)(chunck)
			if out is None:
				out = [[val] for val in tmp ]
			else:
				for val,val_list in zip(tmp, out):
					val_list.append(val)
			
		return tuple([torch.cat(val,dim=0) for val in out])

class GN(Optimizer):
	def __init__(self,
				params,
				lr = 1.,
				use_GN=True,
				is_linear=False,
				epsilon=1e-7,
				chunk_max=128):
		defaults = dict(lr=lr,
						epsilon=epsilon,
						grad_state_dict=None,
						first_eval=True,
						use_GN=use_GN,
						chunk=1,
						chunk_max=chunk_max,
						is_linear=is_linear)
		
		#self.epsilon = epsilon
		#self.grad_state_dict = None
		super(GN,self).__init__(params, defaults)
		#self._params = self.param_groups[0]['params']
		#self.lr = lr
		#self.first_eval = True
		#self.use_GN = use_GN

	def custom_closure(self,x,y,objective,model):
		def closure():
			self.zero_grad()
			pred = model(x)
			loss = objective(pred,y)

			
			
			grad_loss = torch.autograd.grad(loss,pred,retain_graph=True)[0]
			grad_loss = grad_loss.view(-1)
			#flat_params = get_flat_params(model.parameters())
			I_N = torch.eye(pred.shape[0]).to(pred.device)
			I_N = I_N[:,:]
			
			trainable_params = [param for param in self.param_groups[0]['params'] if param.requires_grad]

			def get_vjp(v):
				return torch.autograd.grad(pred.view(-1),trainable_params,v,allow_unused=True,retain_graph=True)
			done = False
			while self.defaults["chunk"] <= self.defaults["chunk_max"] and not done:
				try: 
					jacobian = chunked_vmap(get_vjp,self.defaults["chunk"], I_N)
					done = True
				except:
					self.defaults["chunk"]*=2
			return loss, grad_loss, jacobian, trainable_params, y

		return closure


	def step(self,closure):
		assert closure is not None

		closure = torch.enable_grad()(closure)
		
		if self.defaults['use_GN']:
			loss, grad_loss, jacobian, trainable_params, y = closure()
			with torch.no_grad():
				A = 0.
				for jac in jacobian:
					jac=jac.double()
					A +=  torch.einsum('n...,m...->nm',jac,jac)

				uu,ss,vv = torch.linalg.svd(A)
				if self.defaults['first_eval']:
					uu,ss,vv = torch.linalg.svd(A)
					self.defaults['first_eval']=False
				ss = torch.clamp(ss,min=0.)
				#print(ss)
				min_sigma = max(self.defaults['epsilon'],torch.min(ss).item())
				max_sigma = torch.max(ss).item()
				if self.defaults['is_linear']:
					eps = 1e-10
					inv_ss = ss
					inv_ss[ss<eps*ss[0]] = 0.
					inv_ss[ss>=eps*ss[0]]= 1./inv_ss[ss>=eps*ss[0]]
					#inv_ss = 1./ss
					grad_loss = y.view(-1)
				else:
					inv_ss = 1./(ss+min_sigma) 
				grad = torch.einsum('m...,mk,k->k...',grad_loss.double(),uu,inv_ss)
				grad = torch.einsum('k...,kn->n...',grad,vv)
				incr_path_lenght = 0.
				for param,jac in zip(trainable_params,jacobian):
					if self.defaults['is_linear']:
						descent_dir = param.data- torch.einsum('n...,n->...',jac,grad.float())
					else:	
						if self.defaults['use_GN']:
							descent_dir = self.defaults['lr']*torch.einsum('n...,n->...',jac,grad.float())
						else:
							descent_dir = self.defaults['lr']*torch.einsum('n...,n->...',jac,grad_loss)
					
					param.data -= descent_dir
					incr_path_lenght += torch.sum(torch.mean(descent_dir**2, axis=0))
				incr_path_lenght = torch.sqrt(incr_path_lenght)
				grad_norm = torch.norm(grad_loss)
				self.defaults['grad_state_dict'] = {'min_sigma':min_sigma,
						'max_sigma':max_sigma,
						'norm_grad': grad_norm.item(),
						'cond': max_sigma/min_sigma,
						'incr_path_lenght': incr_path_lenght.item()}
		else:
			loss = closure()
			incr_path_lenght = 0.
			for param in self.param_groups[0]['params']:
				if param.requires_grad and param.grad is not None:
					descent_dir = self.defaults['lr']*param.grad
					param.data -= descent_dir
					incr_path_lenght += torch.sum(torch.mean(descent_dir**2, axis=0))
			incr_path_lenght = torch.sqrt(incr_path_lenght)
			grad_norm = 0.
			self.defaults['grad_state_dict'] = {'min_sigma':0.,
						'max_sigma':0.,
						'norm_grad': 0.,
						'cond': 0.,
						'incr_path_lenght': incr_path_lenght.item()}
		return loss.detach().item()
