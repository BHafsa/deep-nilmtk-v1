import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from tqdm import tqdm
from pyro.contrib.forecast import eval_crps
from pyro.ops.stats import quantile
from .layers import  create_conv1, create_deconv1, create_linear, elu_plus_one_plus_epsilon, Up, MDGMM





class UNETNILM(nn.Module):

    """
    
    UNET-NILM impelementation 
    The orginal paper can be found here: https://dl.acm.org/doi/abs/10.1145/3427771.3427859

    .. _unet:
    """
       
    def __init__(
            self, 
            params):
        super().__init__()
        
        
        out_size= len(params['appliances']) if 'appliance' in params else params['out_size']
        
        if 'feature_type' in params:
            n_channels=4 if params['feature_type']=="combined" else 1
        else:
            n_channels = params['n_channels']
        
        pool_filter = params['pool_filter'] if 'pool_filter' in params  else 8
        latent_size = params['latent_size'] if 'latent_size' in params  else 1024
        
        layers = [nn.Sequential(create_conv1(n_channels, 30, 
                                             kernel_size=10, 
                                             stride=2),
                                nn.ReLU())]
        
        layers.append(nn.Sequential(create_conv1(30, 30, 8, bias=True, stride=2),
                           nn.ReLU()))
        layers.append(nn.Sequential(create_conv1(30, 40, 6, bias=True, stride=1),
                           nn.ReLU()))
        layers.append(nn.Sequential(create_conv1(40, 50, 5, bias=True, stride=1),
                           nn.ReLU(),
                            nn.Dropout(0.2)))
        layers.append(nn.Sequential(create_conv1(50, 50, 5, bias=True, stride=1),
                           nn.ReLU()))
        self.enc_layers = nn.ModuleList(layers)
        
        layers = []
        layers.append(Up(50, 40, 5, 1, nn.ReLU()))
        layers.append(Up(40, 30, 5, 1, nn.ReLU()))
        layers.append(Up(30, 30, 6, 1, nn.ReLU()))
        self.dec_layers = nn.ModuleList(layers)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(pool_filter),
                                nn.Flatten(),
                                create_linear(30*pool_filter, latent_size),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(latent_size, out_size))
        
        self.mean = params['mean'] if 'mean' in params else 0
        self.std = params['std'] if 'std' in params else 1
        
    
            

    def forward(self, x):
        if x.ndim!=3:
            x = torch.unsqueeze(x, 1)
        else:
            x = x.permute(0,2,1)
        xi = [self.enc_layers[0](x)]
       
        for layer in self.enc_layers[1:]:
            
            xi.append(layer(xi[-1]))
            
        for i, layer in enumerate(self.dec_layers):
           
            xi[-1] = layer(xi[-1], xi[-2 - i])
            
        out = self.fc(xi[-1])
        return out


    def step(self, batch):
        x, y  = batch 
        out   = self(x)  # BxCxT
        error = (y - out)
        loss = F.mse_loss(out, y)
        mae = error.abs().data.mean()
        return  loss, mae   
    
    def predict(self,  model, test_dataloader):
        
        net = model.model.eval()
        num_batches = len(test_dataloader)
        values = range(num_batches)
        
        pred = []
        true = []
        
        with tqdm(total = len(values), file=sys.stdout) as pbar:
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    if len(batch)==2:
                        x, y  = batch
                        out = net(x)
                        pred.append(out)
                        true.append(y)
                    else:
                        x  = batch
                        out = net(x)
                        pred.append(out)
                    del  batch    
                    pbar.set_description('processed: %d' % (1 + batch_idx))
                    pbar.update(1)  
                
                pbar.close()
        
        # Perform the denormalization Here
        # Denormalise the output 
        if self.target_norm == 'z-norm':
            # z-normalisation
            pred = self.mean + self.std * pred
            pred = torch.tensor(np.where(pred > 0, pred, 0))
        elif self.target_norm =='min-max':
            # min-max normalisation
            pred = self.min + (self.max - self.min) * pred
            pred = torch.tensor(np.where(pred > 0, pred, 0))
        else:
            # log normalisation was perfomed
            pred = pred.expm1() 
        
        if len(true)!=0:
            true  = torch.cat(true, 0).expm1() 
            results = {"pred":pred, 'true':true}
        else:
            results = {"pred":pred}
        return results



class UNETNILMSeq2Quantile(nn.Module):
       
    def __init__(self, params):
        super().__init__()
        
        out_size=len(params['appliances'])
        seq_len= params['in_size']
        n_channels=4 if params['feature_type']=="combined" else 1
        quantiles= params['quantiles']
                            
        pool_filter = params['pool_filter'] if 'pool_filter' in params  else 8
        latent_size = params['latent_size'] if 'latent_size' in params  else 1024
        
        self.q = torch.tensor(quantiles)
        self.out_size = out_size
        self.unet = UNETNILM({
                         'n_channels' : n_channels, 
                         'out_size' : out_size*seq_len//2, 
                         'pool_filter' : pool_filter, 
                         'latent_size': latent_size
                         })
        
        self.target_norm = params['target_norm']
        self.mean = params['mean'] if 'mean' in params else 0
        self.std = params['std'] if 'std' in params else 1
        
        
    def forward(self, x):
        out = self.unet(x).reshape(x.size(0), -1, self.out_size)
        return out
    
    @staticmethod
    def suggest_hparams(self, trial):
        '''
        Function returning list of params that will be suggested from optuna
    
        Parameters
        ----------
        trial : Optuna Trial.
    
        Returns
        -------
        dict: Dictionary of parameters with values suggested from optuna
    
        '''
        
        return {
            }
    
    def smooth_pinball_loss(self, y, q, tau, alpha = 1e-2, kappa = 1e3, margin = 1e-2):
        #https://github.com/hatalis/smooth-pinball-neural-network/blob/master/pinball_loss.py
        #Hatalis, Kostas, et al. "A Novel Smoothed Loss and Penalty Function 
        #for Noncrossing Composite Quantile Estimation via Deep Neural Networks." arXiv preprint (2019).
        error = (y - q)
        q_loss = (tau * error + alpha * F.softplus(-error / alpha)).sum(0).mean()
        # calculate smooth cross-over penalty
        diff = q[1:,:,:] - q[:-1,:,:]
        penalty = kappa * torch.square(F.relu(margin - diff)).mean()
        loss = penalty+q_loss
        return loss
    
    def step(self, batch):
        x, y  = batch
        out = self(x)
        self.q = self.q.to(x.device)
        q_pred = torch.quantile(out, q=self.q, dim=1)
        
        y_q    = y.permute(1,0,2)
        
      
        tau      = self.q[:,None, None].expand_as(q_pred)
        loss = self.smooth_pinball_loss(y_q, q_pred, tau)
        mae = (y_q - q_pred).abs().data.sum(0).mean()
        return loss, mae
       
    
    def predict(self,  model, test_dataloader):
        
        net = model.model.eval()
        num_batches = len(test_dataloader)
        values = range(num_batches)
        
        pred = []
        true = []
        
        with tqdm(total = len(values), file=sys.stdout) as pbar:
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    if len(batch)==2:
                        x, y  = batch
                        out = net(x)
                        pred.append(out)
                        true.append(y)
                    else:
                        x  = batch
                        out = net(x)
                        pred.append(out)
                    del  batch    
                    pbar.set_description('processed: %d' % (1 + batch_idx))
                    pbar.update(1)  
                
                pbar.close()
         
        # Perform the denormalization Here
        # Denormalise the output 
        if self.target_norm == 'z-norm':
            # z-normalisation
            pred = self.mean + self.std * pred
            pred = torch.tensor(np.where(pred > 0, pred, 0))
        elif self.target_norm =='min-max':
            # min-max normalisation
            pred = self.min + (self.max - self.min) * pred
            pred = torch.tensor(np.where(pred > 0, pred, 0))
        else:
            # log normalisation was perfomed
            pred = pred.expm1() 
        
        
        
        
        q = self.q.to(x.device)
        q_pred = torch.quantile(pred, q=q, dim=1).permute(1,0,2)
        
        
        pred  = q_pred[:,q_pred.size(1)//2,:]
        
        
        results = {"pred":pred, "q_pred":q_pred, "pred_quantile":pred}
        return results
    
    





class UNETNILMDN(nn.Module):
       
    def __init__(self, params):
        super().__init__()
        
        out_size=len( params['appliances'])
        n_channels=4 if params['feature_type']=="combined" else 1
        dist_type= params['mdn_dist_type']
        
        pool_filter= params['pool_filter'] if 'pool_filter' in params else 8 
        latent_size= params['latent_size'] if 'latent_size'in params else 1024
        mdn_latent=params['mdn_latent'] if 'mdn_latent' in params  else 50
        kmix=params['kmix'] if 'kmix' in params else 5
        dist_type=params['dist_type'] if 'dist_type' in params else "lognormal"
        activation=params['activation'] if 'activation' in params else nn.SiLU()
        
        
        self.unet = UNETNILM({
            'n_channels' : n_channels, 
            'out_size' : mdn_latent, 
            'pool_filter' : pool_filter, 
            'latent_size': latent_size
                         })
        self.mdn = MDGMM(in_dims=mdn_latent,out_dims=out_size,
                 kmix=kmix, dist_type=dist_type, activation=activation)
        
    @staticmethod
    def suggest_hparams(self, trial):
        '''
        Function returning list of params that will be suggested from optuna
    
        Parameters
        ----------
        trial : Optuna Trial.
    
        Returns
        -------
        dict: Dictionary of parameters with values suggested from optuna
    
        '''
        
        return {
            }
    
    def forward(self, x):
        z = self.unet(x)
        pi, mu, sigma, gmm = self.mdn(z)
        return pi, mu, sigma, gmm
    
    def log_nlloss(self, y, gmm):
        logprobs = gmm.log_prob(y)
        return -torch.mean(logprobs)
    
    def sample(self, gmm, n_sample=1000):
        samples = gmm.sample(sample_shape=(n_sample,))
        return samples
    
    def step(self, batch):
        x, y  = batch
        pi, mu, sigma, gmm  = self(x)
        
        samples = self.sample(gmm)
        p10, p50, p90 = quantile(samples, (0.1, 0.5, 0.9)).squeeze(-1)
       
        loss = self.log_nlloss(y, gmm) + F.mse_loss(p50, y)
        crps = eval_crps(samples, y)
        mae = (y - p50).abs().data.mean()
        return  loss, mae
    
    def predict(self,  model, test_dataloader):
        
        net = model.model.eval()
        num_batches = len(test_dataloader)
        values = range(num_batches)
        
        pred = []
        true = []
        mu_probs = []
        pi_probs = []
        simga_probs = []
        samples_pred = []
        
        with tqdm(total = len(values), file=sys.stdout) as pbar:
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    if len(batch)==2:
                        x, y  = batch
                        true.append(y)
                    else:
                        x  = batch
                       
                    pi, mu, sigma, gmm = net(x)
                    pred_sample = net.sample(gmm)
                    mu_probs.append(mu)
                    pi_probs.append(pi)
                    simga_probs.append(sigma)
                    samples_pred.append(pred_sample)
                    del  batch    
                    pbar.set_description('processed: %d' % (1 + batch_idx))
                    pbar.update(1)  
                
                pbar.close()
        
        mu_probs = torch.cat(mu_probs, 0).expm1()
        sigma_probs = torch.cat(simga_probs, 0)
        samples_pred = torch.cat(samples_pred, 1).expm1()
        p10, p50, p90 = quantile(samples_pred, (0.1, 0.5, 0.9)).squeeze(-1)

        if len(true)!=0:
            true  = torch.cat(true, 0).expm1() 
            results = {
                    "pred": p50, "sigma_pred":sigma_probs, "mu_pred":mu_probs, "pi_probs":pi_probs,
                    "sample_pred":samples_pred,"p10":p10, "p90":p90, "true":true}
        else:
             results = {
                    "pred": p50, "sigma_pred":sigma_probs, "mu_pred":mu_probs, "pi_probs":pi_probs,
                    "sample_pred":samples_pred,"p10":p10, "p90":p90}
        return results

      
        
             
            
        