import torch
import torch.optim as optim

from tqdm import trange

from mdn import MdnLinear, GaussianMix
import seaborn as sns
import matplotlib.pyplot as plt

def sample_hyperbola(x, a=1, b=1):
    
    y_sq= ((x**2/a**2) - 1) * b**2
    
    sign= (torch.rand(x.shape) > 0.5).float() * 2 - 1
    
    return torch.sqrt(torch.abs(y_sq)) * sign

def demo_GM():
    g1= torch.Tensor([[0.25, -2, 0.1]])
    g2= torch.Tensor([[0.25, -1, 0.1]])
    g3= torch.Tensor([[0.25, 1, 0.1]])
    g4= torch.Tensor([[0.25, 2, 0.1]])
    
    mix= torch.cat([g1, g2, g3, g4], dim=0).unsqueeze(0).flatten(1)
    
    gmix= GaussianMix(mix)
    gmix.plot_prob_dist()
    
    
def hyperbola_demo():
    model= MdnLinear(1, 256, 2, layers= 2)
    model_op= optim.Adam(model.parameters(), lr=0.0001)
    
    epoch_bar = trange(10000, desc=f"LOSS: 0", position=0, leave=True)
    
    model.train()
    for n in epoch_bar:
        
        model_op.zero_grad()
        
        x= ((torch.rand([256]) * 2 - 1) * 10).unsqueeze(1)
        y= sample_hyperbola(x) + torch.normal(torch.zeros_like(x), 0.1)
        
        out= model(x)
        gout= GaussianMix(out)
        
        loss= -gout.log_prob(y).mean()
        loss.backward()
        
        model_op.step()
        
        epoch_bar.set_description(f"Epoch: {n}, LOSS: {float(loss):.4f}")
        if n % 100 == 0:
            model_y= gout.sample(1)
            
            plt.clf()
            plt.scatter(x,y, label= 'Real Sample')
            plt.scatter(x,model_y.detach(), label= 'Model Sample')
            plt.legend()
            
            ax = plt.gca()
            ax.set_xlim([-10, 10])
            ax.set_ylim([-11, 11])
    
            plt.pause(0.01)
            
if __name__ == '__main__':
    hyperbola_demo()