import torch
import torch.nn as nn

import seaborn as sns
import matplotlib.pyplot as plt

import copy
import math

LL_CONST= math.log(math.sqrt(2 * math.pi))
cat= torch.cat

relu= nn.functional.relu

def temp_softmax(logits, dim=-1, temp=1.0):
    probs = (logits / temp).softmax(dim=dim)
    return probs

def choice(w, nsamples):
    """
    Pytorch implementation of Numpy's np.random.choice method. Higher speed than
    the numpy implementation.

    Parameters
    ----------
    w : Weights for each choice
    nsamples : Number of samples

    Returns
    -------
    The index of the random choice (for each sample)

    """
    device= w.device
    w_tot= torch.cumsum(w, dim=1)
    # w_tot= torch.zeros(w.shape, device= device)
    # for n in range(w.shape[1]):
    #     w_tot[:,n] = w[:,:n+1].sum(dim=-1)
        
    w_tot= w_tot.unsqueeze(1).expand(-1, nsamples, -1)
    r= torch.rand([w_tot.shape[0],w_tot.shape[1], 1], device=device)
    return (r < w_tot).long().argmax(dim=-1)

class MdnLinear(nn.Module):
    def __init__(self, input_size, nhidden, nmix, layers=1, drop_rate= 0.0):
        """
        Adds the MDN Layer. For simple networks such as the hyperbola demo, this can be the entire model.
        Can also be the last layer of a more complex model.

        Parameters
        ----------
        input_size : INT
            Size of the input
        nhidden : INT
            Size of the hidden layers of the network
        nmix : INT
            Number of Gaussian mixtures in the model
        drop_rate : FLOAT, optional
            Dropout rate The default is 0.0.
        layers : INT, optional
            Number of layers for the embedding and each of the three MDN outputs. The default is 1.

        """
        
        super(MdnLinear, self).__init__()
        
        self.nmix= nmix
        
        if layers > 0:
            net= [nn.Linear(input_size, nhidden), nn.Dropout1d(drop_rate), nn.GELU()]
            for _ in range(layers - 1):
                net.extend([nn.Linear(nhidden, nhidden), nn.Dropout1d(drop_rate), nn.GELU()])
            self.embed= nn.Sequential(*net)
        else:
            self.embed= nn.Identity()
            nhidden= input_size
        
        self.lin_pi= nn.Linear(nhidden, nmix)
        self.lin_mean= nn.Linear(nhidden, nmix)
        self.lin_var= nn.Sequential(nn.Linear(nhidden, nmix), nn.Softplus())
        
    def forward(self, x, temp= 1.0):
        x= self.embed(x)
        
        pi= temp_softmax(self.lin_pi(x), temp=temp, dim=-1)
        
        # min_weight = 0.01  # hard floor per mixture
        # pi = min_weight + (1 - self.nmix * min_weight) * temp_softmax(self.lin_pi(x), temp= temp, dim=-1)
        
        mean= self.lin_mean(x)
        # mean[:,0] = 0
        
        var= 10*self.lin_var(x) + 1e-4
        # var[:,0] = 1e-4
        
        pi= pi.unsqueeze(1)
        mean= mean.unsqueeze(1)
        var= var.unsqueeze(1)
        return cat([pi, mean, var], dim=1).permute(0,2,1).flatten(1)

class GaussianMix():
    def __init__(self, inp, min_var= 0.1, min_pi= 0.0001):
        """
        Helper class for interpreting a flattened Gaussian mixture. This class is meant to more or less mimick the functionality of torch.distributions.normal
        for Gaussian mixtures.

        Parameters
        ----------
        inp : torch.Tensor
            Designed to take the output of the MDNLinear Module, which is flattened to 1 dimension in the forward function of that module.
            This tensor can be any shape, as long as the last dimension represents the flattened MDN.
            See demo.py for an example on initilazing this class manually.
        min_var : float
            The minimum variance for any Gaussian in the mixture.This is needed for regression problems with low variance. 
            Log probability is an unstable loss function for low variances.
        """
        
        self.min_var= min_var
        self.min_pi= min_pi
        
        self.ogshape= list(inp.shape)
        self.bs= inp.flatten(0, -2).shape[0]
        
        i= inp.view(self.bs, -1, 3)
        self.N= i.shape[1]
        
        self.pi= i[:,:,0]
        # self.pi = self.pi * (self.pi >= self.min_pi)
        # self.pi = self.pi / self.pi.sum(dim=-1, keepdim=True)
        
        self.mean= i[:,:,1]
        self.var= i[:,:,2]
        
        # print(inp.flatten())
        assert(not self.var.isnan().any())
        assert(not (self.var == -2).any())
        
        self.device= inp.device
        
    def reshape_output(self, out, feature_size):
        outshape= copy.deepcopy(self.ogshape)
        outshape[-1] = feature_size
        return out.view(outshape)
    
    def reshape_mixture(self, target_shape):
        pi= self.pi
        mean= self.mean
        var= self.var
        
        if len(target_shape) > 2:
            pi= pi.unsqueeze(-1).tile([1,1,target_shape[-1]])
            mean= mean.unsqueeze(-1).tile([1,1,target_shape[-1]])
            var= var.unsqueeze(-1).tile([1,1,target_shape[-1]])
            
        return pi, mean, var
        
    def sample(self, nsamples= 1):
        # adj_var= torch.clone(self.var)
        # adj_var[adj_var < self.min_var] = 0.0001
        std= self.var.sqrt()
        
        pi= torch.clone(self.pi)
        # pi = pi * (pi >= self.min_pi)
        # pi = pi / pi.sum(dim=-1, keepdim=True)
        # print(float(pi[0].min()))
        
        ind= choice(pi, nsamples)
        
        m= torch.gather(self.mean, 1, ind)
        s= torch.gather(std, 1, ind)
        out= torch.normal(m, s)
        return self.reshape_output(out, nsamples)
    
    def cdf(self, target):
        std= self.var.sqrt()
        cdf= (self.pi * (0.5 * (1 + torch.erf((target - self.mean) * std.reciprocal() / math.sqrt(2))))).sum(dim=1)
            
        return self.reshape_output(cdf, 1)
    
    def expectation(self, return_var=False):
        m = (self.pi * self.mean).sum(dim=1)
        
        if return_var:
            within_var = (self.pi * self.var).sum(dim=1)
            
            # Need to unsqueeze m to [batch_size, 1] for proper broadcasting
            between_var = (self.pi * torch.square(self.mean - m.unsqueeze(1))).sum(dim=1)
            var = within_var + between_var
            
            return self.reshape_output(m, 1), self.reshape_output(var, 1)
        else:
            return self.reshape_output(m, 1)
    
    def nll_loss(self, target_mean, target_var= None):
        
        if target_var is None:
            target_var = torch.zeros_like(target_mean)
        
        pi, mean, var= self.reshape_mixture(target_mean.shape)
        
        adj_var= torch.clamp(var, min= self.min_var)
        
        std= adj_var.sqrt()
        
        log_p_approx = torch.log(pi) - (torch.square(target_mean - mean) + target_var) / (2 * adj_var) - torch.log(std) - LL_CONST
        loglik= torch.logsumexp(log_p_approx, dim=1)
        
        loss= -loglik
        return torch.nan_to_num(loss, nan=50, posinf= 50, neginf= -5)
    
    def low_entropy_loss(self, target_shape):
        
        pi, mean, var= self.reshape_mixture(target_shape)
        return (pi * torch.log(pi)).sum(dim=-1)
    
    def log_prob(self, target):
        
        pi, mean, var= self.reshape_mixture(target.shape)
        
        log_probs = torch.log(pi) -((target - mean) ** 2) / (2 * var) - torch.log(var.sqrt()) - LL_CONST
        return torch.logsumexp(log_probs, dim=1)
    
    def plot_sample_dist(self, nsamples):
        s= self.sample(nsamples).detach().cpu()
        
        plt.clf()
        plt.xlabel("Value")
        for n,x in enumerate(s.detach().cpu()):
            sns.kdeplot(x, label= n)
        plt.legend()
        plt.pause(0.1)
        
    def plot_prob_dist(self, fval= 0.1, prange= (-10, 10)):
        r= torch.arange(prange[0],prange[1], 0.01).unsqueeze(0).unsqueeze(0).tile([self.bs,self.N, 1])
        p= torch.exp(self.log_prob(r.to(self.mean.device)).detach()).cpu()
        mu= self.expectation()
        
        plt.clf()
        plt.xlabel("Value")
        plt.ylabel("$p$")
        plt.title("Probability Distribution")
        for n,x in enumerate(p):
            plt.plot(r[0,0][x>fval], x[x>fval], label= f"($\mu = {float(mu[n])}$)")
        plt.legend()
        plt.pause(0.1)

