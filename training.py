import torch
#import ot
import numpy as np
import copy
from functools import partial
import torch.nn as nn
import time




def assign_device(device):
    if device >-1:
        device = (
            'cuda:'+str(device) 
            if torch.cuda.is_available() and device>-1 
            else 'cpu'
        )
    elif device==-1:
        device = 'cuda'
    elif device==-2:
        device = 'cpu'
    return device


def get_dtype(dtype):
    if dtype==64:
        return torch.double
    elif dtype==32:
        return torch.float
    else:
        raise NotImplementedError('Unkown type')

def weights_init(args,m):
    print("noise value:")
    print(args['mean'])
    print(args['std'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    if isinstance(m, nn.Linear):
        m.weight.data = torch.normal(mean=args['mean'], std=args['std'], 
                                    size= m.weight.data.shape,
                                    device=m.weight.data.device)
        print("params")
        print(m.weight)
        #m.weight.data.normal_(mean=args['mean'],std=args['std'])
        if m.bias is not None:
            m.bias.data = torch.normal(mean=args['mean'], std=args['std'], 
                                       size= m.bias.data.shape,
                                       device=m.bias.data.device)
            print(m.bias) 
            #m.bias.data.normal_(mean=args['mean'],std=args['std'])


class Trainer:
    
    def __init__(self, config, logger):
        self.logger= logger

        self.args = config
        self.device = assign_device(self.args.system.device)
        self.dtype = get_dtype(self.args.system.dtype)
        self.build_trainer()    
    
    def build_trainer(self):
        def squared_loss(a,b):
            return 0.5*torch.mean((a-b)**2)
        self.loss = squared_loss

        from models.multilayer_net import OneHiddenLayer
        from core.optimizers.GN  import GN 
        from loaders.teacher import TeacherDataset

        if self.args.model.non_linearity=='ReLU':
            non_linearity = torch.nn.ReLU()
        elif self.args.model.non_linearity=='SiLU':
            non_linearity = torch.nn.SiLU()


        self.model = OneHiddenLayer(self.args.model.d_int,self.args.model.H,
                                    non_linearity=non_linearity#,beta=self.args.model.beta
                                    ).to(self.device)
        self.model_init = copy.deepcopy(self.model)
        self.teacher_network = OneHiddenLayer(self.args.model.d_int,self.args.model.teacher_H,
                                    non_linearity=non_linearity#,beta=self.args.model.beta
                                    ).to(self.device)

        self.init_net(self.teacher_network,
                     self.args.init.mean_teacher,
                     self.args.init.std_teacher
                     )

        self.init_net(self.model,
                     self.args.init.mean,
                     self.args.init.std
                     )

        if self.args.model.copy_teacher_network:
            copy_teacher_network(self.model,self.teacher_network)


        self.init_net_last_zero()
        test_set = TeacherDataset(self.teacher_network,
                                  self.args.data.n_test,
                                  self.dtype,self.device)

        n_train_max = 2000
        n_train = max(self.args.data.n_train, n_train_max)
        train_set = TeacherDataset(self.teacher_network, n_train,
                                  self.dtype,self.device)
        train_set.total_size= self.args.data.n_train
        train_set.X = train_set.X[:self.args.data.n_train]
        train_set.Y = train_set.Y[:self.args.data.n_train]


        params = {'batch_size': self.args.data.n_train,
                  'shuffle':False,
                  'num_workers':0}
 
        self.train_loader = [(train_set.X,train_set.Y)]
        self.test_loader = [(test_set.X,test_set.Y)]
        self.optimizer = GN(self.model.parameters(), 
                            lr= self.args.optimizer.lr,
                            use_GN=self.args.optimizer.use_GN)
        self.max_iter = self.args.optimizer.max_iter
        self.warmstart = {'aa':None, 'bb':None, 'ab':None}
        
        self.precision = 1e-8
        self.alg_time = 0.
        self.iteration = 0
        self.loss_0 = 0.
        self.old_weights = self.model_init.linear1.weight

        self.model_init = copy.deepcopy(self.model)
        self.cum_dict = 0.

        if self.args.init.pre_train:
            self.pre_train_last_layer()
            self.optimizer.defaults['is_linear']=False



    def init_net(self,model,mean,std):
        weights_init_loc = partial(weights_init,{'mean':mean,
                                             'std':std,
                                             'seed':self.args.system.seed})
        model.apply(weights_init_loc)

    def init_net(self,model,mean,std):
        for param in model.parameters():
            param.data = torch.normal(mean=mean, std=std, 
                                    size= param.shape,
                                    device=param.device)
    
    def init_net_last_zero(self):
        for param in self.model.linear2.parameters():
            param.data = torch.zeros_like(param.data)       


    def train_last_layer(self,log_name='post_train',ckpt_name='post_train_ckpt'):
        self.optimizer.defaults['first_eval']= True
        self.optimizer.defaults['lr'] = 100.
        use_GN = self.args.optimizer.use_GN
        self.optimizer.defaults['use_GN'] = True
        self.args.optimizer.use_GN=True
        self.iteration = 0
        for param in self.model.linear1.parameters():
            param.requires_grad = False
        self.init_net_last_zero()
        self.optimizer.defaults['is_linear']=True
        self.train(max_iter=1, prefix="", log_name= log_name, ckpt_name=ckpt_name)
        self.optimizer.defaults['is_linear']=False
        #self.optimizer.defaults['use_GN'] = False
        #self.args.optimizer.use_GN=False
        self.train(max_iter=self.args.init.pre_train_iter, prefix="", log_name= log_name, ckpt_name=ckpt_name)
        
        for param in self.model.linear1.parameters():
            param.requires_grad = True
        
        self.optimizer.defaults['use_GN'] = use_GN
        self.args.optimizer.use_GN=use_GN

    def pre_train_last_layer(self):
        self.train_last_layer(log_name='metrics',ckpt_name='last_ckpt')


    def post_train_last_layer(self):
        self.train_last_layer(log_name='post_train',ckpt_name='post_train_ckpt')


    def get_closure(self,x,y,default=False):
        def closure():
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss(pred,y)
            loss.backward()
            return loss

        if hasattr(self.optimizer,'custom_closure') and self.args.optimizer.use_GN and not default: 
            closure = self.optimizer.custom_closure(x,y,self.loss,self.model)

        return closure


    def distance_matrix(self, ref_model):
        Mab = 0.
        Maa = 0.
        Mbb = 0.
        for param, param_star in zip(self.model.linear1.parameters(),
                                    ref_model.linear1.parameters()):
            sum_param = torch.einsum('n...,m...->nm',param,param)
            sum_param_star = torch.einsum('n...,m...->nm',param_star,param_star)
            prod = torch.einsum('n...,m...->nm',param,param_star)
            diag_param = torch.diag(sum_param)[:,None]
            diag_param_star = torch.diag(sum_param_star)[:,None]


            Mab +=  diag_param +  diag_param_star.T -2*prod
            Maa+= diag_param +  diag_param.T -2*sum_param
            Mbb+= diag_param_star +  diag_param_star.T -2*sum_param_star
        
        M = {'aa': Maa, 'bb': Mbb, 'ab':Mab}
        return M

    def train(self,max_iter=None,prefix='', ckpt_name='last_ckpt', log_name='metrics'):
        #start_time = time.time()
        self.alg_time = 0.
        if max_iter is None:
            max_iter = self.max_iter
        done = self.iteration>= max_iter 
        while not done:
            self.iteration +=1 

            for batch_idx, data in enumerate(self.train_loader):
                x,y = data
                x = x.to(self.device)
                y = y.to(self.device)
                closure = self.get_closure(x,y)
                start_time = time.time()
                loss = self.optimizer.step(closure)
                end_time = time.time()
                
                metrics = {"train_loss":loss,
                            "iter":self.iteration, 
                            "time":end_time-start_time}
                metrics.update(self.optimizer.defaults['grad_state_dict'])
            if self.iteration==1:
                self.loss_0 = loss
            done = self.iteration>= max_iter or loss<=self.loss_0*self.precision
            save_metrics = np.mod(self.iteration,self.args.metrics.save_freq)==0 or self.iteration==1 or done

            if save_metrics:
                
                for batch_idx, data in enumerate(self.test_loader):
                    x,y = data
                    x = x.to(self.device)
                    y = y.to(self.device)
                    closure = self.get_closure(x,y, default=True)
                    loss = closure()
                    metrics.update({"test_loss":loss.item()})
#                if np.mod(self.iteration,self.args.metrics.dist_freq_eval)==0:
                    dist_cur = min_distance(self.model.linear1.weight.data, 
                                                  self.old_weights)
                    dist_init = min_distance(self.model.linear1.weight.data, 
                                                  self.model_init.linear1.weight.data)
                    #if self.iteration==100:
                    #    print("Update reference weight")
                    self.old_weights = copy.deepcopy(self.model.linear1.weight.data)
                    #self.cum_dict += dist_init['weighted_dist']
                    #dist_init.update({'cum_weighted_dist': self.cum_dict})

                    dist_target = min_distance(self.model.linear1.weight.data, 
                                                    self.teacher_network.linear1.weight.data)
                    w_metrics = weight_metrics(self.model.linear1.weight.data)

                    dist_target = add_prefix('target_',dist_target)
                    dist_init = add_prefix('init_',dist_init)
                    dist_cur = add_prefix('cur_',dist_cur)
                    w_metrics = add_prefix('weights_',w_metrics)
                    metrics.update(dist_init)
                    metrics.update(dist_target)
                    metrics.update(dist_cur)
                    metrics.update(w_metrics)

                metrics = add_prefix(prefix,metrics)
                print(metrics)
                self.logger.log_checkpoint(self, log_name= ckpt_name)
                self.logger.log_metrics(metrics, log_name=log_name)


def copy_teacher_network(s_model, t_model):
    M,N = t_model.linear1.weight.data.shape
    weights = torch.zeros_like(s_model.linear1.weight.data)
    bias = torch.zeros_like(s_model.linear1.bias.data)
    weights[:M,:] = t_model.linear1.weight.data
    bias[:M] = t_model.linear1.bias.data
    weights[M:2*M,:] = t_model.linear1.weight.data
    bias[M:2*M] = t_model.linear1.bias.data


    s_model.linear1.weight.data = weights
    s_model.linear1.bias.data = bias



def add_prefix(prefix, dico):
    return {prefix+key: value for key,value in dico.items()}


def get_basis_and_effective_dim(weight, epsilon=1e-9):
        ## weight: N times K matrix (extracts a basis for the subspace defined by the columns of the input matrix)
    
    N,K = weight.shape  
    uu,ss,vv = torch.linalg.svd(weight)
    ss = torch.clamp(ss, min=0.)
    ss = ss/ss[0]
    effective_dim = torch.sum(ss/(epsilon + ss))
    if int(effective_dim)==effective_dim:
        rank = int(effective_dim)
    else:
        rank = int(effective_dim)+1


    return uu[:,:rank], rank, effective_dim, ss

def cosine_ditance(weight,ref_weight):
    norm_weights  = torch.norm(weight,dim=1)
    normalized_weight = weight/norm_weights.unsqueeze(dim=1)
    normalized_weight[norm_weights==0.,:] = 0.
    normalized_ref_weight = ref_weight/torch.norm(ref_weight,dim=1).unsqueeze(dim=1)
    K = normalized_weight@normalized_ref_weight.T
    gram = 2*(1.-K)
    gram = torch.clamp(gram, min=0.)
    
    return torch.sqrt(gram)
    
def min_distance(weight, ref_weight):
    weight = weight.double()
    ref_weight = ref_weight.double()
    dist_matrix = cosine_ditance(weight,ref_weight)
    ref_dist_matrix = cosine_ditance(ref_weight,ref_weight)

    

    value, indices_col = torch.min(dist_matrix,axis=1)
    value = torch.clamp(value, min=0.)
    dist = torch.mean(value)
    
    norm_weight = dist_weights(weight)
    # weighted_dist_matrix = torch.abs(dist_matrix- ref_dist_matrix[indices_col,:])
    # weighted_dist_matrix = torch.einsum('ij,i->ij',
    #                             weighted_dist_matrix,
    #                             norm_weight)
    # mean_weighted_dist = torch.sum(torch.mean(weighted_dist_matrix, axis=1))

    mean_weighted_dist = torch.sum(
                                torch.einsum('i,i->',value,norm_weight))
    
    value_ref, indices_row = torch.min(dist_matrix,axis=0)
    value_ref = torch.clamp(value_ref, min=0.)
    dist_ref = torch.mean(value_ref)

    weighted_mean = torch.einsum('ij,i->j',weight,norm_weight)
    variance = torch.einsum('ij,i->j', (weight-weighted_mean.unsqueeze(dim=0))**2, norm_weight)
    variance = torch.mean(torch.sqrt(variance))
    out_dict = {#'param_to_target_dist': dist.item(),
            #'target_to_param_dist': dist_ref.item(),
            'weighted_dist': mean_weighted_dist.item(),
            }
    # if weight.shape == ref_weight.shape:
    #     euclidean = torch.norm(weight-ref_weight)
    #     out_dict.update({'euclidean':euclidean.item()})


    return out_dict



def weight_metrics(weight):
    norm_weight = dist_weights(weight)
    entropy = torch.sum(torch.special.entr(norm_weight))
    
    weighted_mean = torch.einsum('ij,i->j',weight,norm_weight)
    variance = torch.einsum('ij,i->j', (weight-weighted_mean.unsqueeze(dim=0))**2, norm_weight)
    variance = torch.mean(torch.sqrt(variance))
    out_dict = {'variance':variance.item()}
    return out_dict


def dist_weights(weight,power=2):
    norm_weight = torch.pow(torch.norm(weight,dim=1),power)
    norm_weight = norm_weight/torch.sum(norm_weight)
    return norm_weight

def subspace_distance(weight, ref_weight, with_sigma=False):
    ### weight and ref_weight are the weights of a linear layer: 
    ### weight and ref_weight have a shape K x N and K' x N.

    A_model, rank_model, eff_dim_model, ss_model = get_basis_and_effective_dim(weight.T)

    A_ref, rank_ref, eff_dim_ref, ss_ref = get_basis_and_effective_dim(ref_weight.T)

    B = torch.einsum('ki,kj->ij' ,A_model,A_ref)
    uu,ss,vv = torch.linalg.svd(B)

    ss = torch.clamp(ss, min=0., max=1.)
    arccos_ss = torch.arccos(ss)
    rank = min(rank_model,rank_ref)
    dim_dist = torch.abs(eff_dim_model-eff_dim_ref)        

    grassmann_dist = torch.norm(arccos_ss[:rank])

    out_dict = {'grassmann_dist': grassmann_dist.item(),
            'dim_dist': dim_dist.item()}
    if with_sigma:
        singular_values = { 'sigma_weight_'+str(i): sigma for i, sigma in enumerate(ss_model)}
        out_dict.update(singultar_values)

    return out_dict

