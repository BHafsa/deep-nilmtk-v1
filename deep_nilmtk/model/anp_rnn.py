import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import math
import numpy as np
from tqdm import tqdm

from .layers import  BatchMLP, Attention, LSTMBlock, BatchNormSequence,create_conv1, Up
import sys


class UNET(nn.Module):
       
    def __init__(
            self, 
            n_channels: int =1, out_size:int =1):
        super().__init__()
        
        layers = [nn.Sequential(create_conv1(n_channels, 30, 
                                             kernel_size=10, bias=True,
                                             stride=1, padding=4, padding_mode='replicate'),
                                nn.ReLU())]
        
        layers.append(nn.Sequential(create_conv1(30, 30, 7, bias=True, stride=1, padding=3, padding_mode='replicate'),
                           nn.ReLU()))
        layers.append(nn.Sequential(create_conv1(30, 40, 6, bias=True, stride=1, padding =3, padding_mode='replicate'),
                           nn.ReLU()))
        layers.append(nn.Sequential(create_conv1(40, 50, 5, bias=True, stride=1, padding=2, padding_mode='replicate'),
                           nn.ReLU(),
                            nn.Dropout(0.2)))
        layers.append(nn.Sequential(create_conv1(50, 50, 5, bias=True, stride=1, padding= 2,padding_mode='replicate'),
                           nn.ReLU()))
        self.enc_layers = nn.ModuleList(layers)
        
        layers = []
        layers.append(Up(50, 40, 5, 1, nn.ReLU(), pad_out= True))
        layers.append(Up(40, 30, 5, 1, nn.ReLU(), pad_out= True))
        layers.append(Up(30, out_size, 6, 1, nn.ReLU(), pad_out= True))
        self.dec_layers = nn.ModuleList(layers)
        
            

    def forward(self, x):
        
        if x.ndim!=3:
            x = torch.unsqueeze(x, 1)
        else:
            x = x.permute(0,2,1)
        
        xi = [self.enc_layers[0](x)]
       
        for layer in self.enc_layers[1:]:
            xi_ = layer(xi[-1])
            xi.append(xi_)

            
        for i, layer in enumerate(self.dec_layers):
            xi_ = layer(xi[-1], xi[-2 - i])
            xi[-1] = xi_
            
        out = xi[-1]
        return out.permute(0,2,1)



class LatentEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        latent_dim=32,
        self_attention_type="dot",
        n_encoder_layers=2,
        min_std=0.1,
        batchnorm=True,
        dropout=0.8,
        attention_dropout=0.8,
        attention_layers=1,
        use_unet= True
    ):
        super().__init__()
        # self._input_layer = nn.Linear(input_dim, hidden_dim)
        
        if use_unet:
            self._encoder = UNET(
                n_channels = input_dim,
                out_size =  hidden_dim
            )
        else:
            self._encoder = BatchMLP(input_dim,
                            num_layers=n_encoder_layers,
                             output_size = hidden_dim,
                             dropout=dropout
                             )
        
        self._self_attention = Attention(
                hidden_dim,
                self_attention_type,
                attention_layers,
                rep="identity",
                dropout=attention_dropout,
            )

        


        self._penultimate_layer = nn.Linear(hidden_dim, hidden_dim)
        # nn.utils.weight_norm(self._penultimate_layer)
        # nn.init.xavier_uniform_(self._penultimate_layer.weight)
        self._penultimate_layer.bias.data.fill_(0)

        self._mean = nn.Linear(hidden_dim, latent_dim)
        # nn.utils.weight_norm(self._mean)
        # nn.init.xavier_uniform_(self._mean.weight)
        nn.init.zeros_(self._mean.bias)

        self._log_var = nn.Linear(hidden_dim, latent_dim)
        # nn.utils.weight_norm(self._log_var)
        # nn.init.xavier_uniform_(self._log_var.weight)
        nn.init.zeros_(self._log_var.bias)

        self._min_std = min_std
        

    def forward(self, x, y):
        encoder_input = torch.cat([x, y], dim=-1)
        
        # Pass final axis through MLP
        
        encoded = self._encoder(encoder_input)
        # print(f'this from teh latent encoder {encoded} {x}{y}')
        # Aggregator: take the mean over all points
           
        attention_output = self._self_attention(encoded,encoded, encoded)
        mean_repr = attention_output.mean(dim=1)
        # mean_repr = torch.logsumexp(attention_output, dim=1, keepdim=True).squeeze()
        
        # Have further MLP layers that map to the parameters of the Gaussian latent
        # print(f'weights:{self._penultimate_layer.weight}')
        # print(f'Biases:{self._penultimate_layer.bias}')
        
        mean_repr = torch.relu(self._penultimate_layer(mean_repr))
    
        # print(mean_repr)
        # Then apply further linear layers to output latent mu and log sigma
        mean = self._mean(mean_repr)
        log_var = self._log_var(mean_repr)
      
        sigma = self._min_std + (1 - self._min_std) * torch.sigmoid(log_var * 0.5)

        dist = torch.distributions.Normal(mean, sigma)
        return mean, dist, sigma


class DeterministicEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        x_dim,
        hidden_dim=32,
        n_d_encoder_layers=2,
        self_attention_type="dot",
        cross_attention_type="dot",
        attention_layers=1,
        batchnorm=True,
        dropout=0.8,
        attention_dropout=0.5,
        use_unet= True
    ):
        super().__init__()
        # self._input_layer = nn.Linear(input_dim, hidden_dim)
        if use_unet:
            self._d_encoder = UNET(
                n_channels = input_dim,
                out_size =  hidden_dim
            )
        else:
            self._d_encoder = BatchMLP(input_dim,
                    dropout=dropout,
                    num_layers=n_d_encoder_layers,
                    output_size = hidden_dim)
        
        self._self_attention = Attention(
                hidden_dim,
                self_attention_type,
                attention_layers,
                rep="identity",
                dropout=attention_dropout,
            )
        self._cross_attention = Attention(
            hidden_dim,
            cross_attention_type,
            x_dim=x_dim,
            rep="identity",
            attention_layers=attention_layers,
        )

    def forward(self, context_x, context_y, target_x):
        # Concatenate x and y along the filter axes
        d_encoder_input = torch.cat([context_x, context_y], dim=-1)

        # Pass final axis through MLP
        d_encoded = self._d_encoder(d_encoder_input)

        d_encoded = self._self_attention(d_encoded, d_encoded, d_encoded)
        # print(d_encoded.shape)
        # Apply attention as mean aggregation
        h = self._cross_attention(context_x, d_encoded,  target_x)
        
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim=32,
        latent_dim=32,
        n_decoder_layers=3,
        min_std=0.01,
        batchnorm=False,
        dropout=0.2,
        use_unet=True,
        use_latent =False
    ):
        super(Decoder, self).__init__()
        self._target_transform = nn.Linear(x_dim, hidden_dim)
        # nn.utils.weight_norm(self._target_transform)
        # nn.init.xavier_uniform_(self._target_transform.weight)
        if use_latent:
            hidden_dim_2 = 2 * hidden_dim + latent_dim
        else:
            hidden_dim_2 =  2 * hidden_dim 
        
        if use_unet:
            self._decoder = UNET(
                n_channels = hidden_dim_2,
                out_size =  hidden_dim_2
            )
        else:
            self._decoder = BatchMLP(hidden_dim_2,
                                num_layers=n_decoder_layers,
                                output_size = hidden_dim_2,
                                dropout=dropout)
        
        # Normal Function estimation
        self._mean = nn.Linear(hidden_dim_2, y_dim)
        nn.utils.weight_norm(self._mean)
        nn.init.xavier_uniform_(self._mean.weight)
        self._std = nn.Linear(hidden_dim_2, y_dim)
        nn.utils.weight_norm(self._std)
        nn.init.xavier_uniform_(self._std.weight)

        # Logit Function estimation

        self.p = nn.Linear(hidden_dim_2, y_dim)
        nn.utils.weight_norm(self.p)
        nn.init.xavier_uniform_(self.p.weight)
        


        self._min_std = min_std




        

    def forward(self, r, z, target_x):
        # concatenate target_x and representation
        x = self._target_transform(target_x)
        if z is not None:
            z = torch.cat([r, z], dim=-1)
        else:
            z=r
        # print(x)
        r = torch.cat([z, x], dim=-1)
        
        r = self._decoder(r)

        # Get the ON/OFF states estimation



        # Get the mean and the variance
        mean = self._mean(r)
        log_sigma =  self._std(r)
        
        # Bound or clamp the variance

        

        sigma = self._min_std + (1 - self._min_std) * F.softplus(0.5 * log_sigma)
        # print(mean.shape)
        y_dist = torch.distributions.Normal(mean, sigma)

        p = nn.Sigmoid(self.p(r))
        
        s_dist = torch.distributions.bernoulli.Bernoulli(p)

        return mean,sigma, y_dist, p, s_dist


class NeuralProcess(nn.Module):
    
    
    def __init__(self, params):
        super(NeuralProcess, self).__init__()

        n_channels = 8 if params['feature_type'] == 'combined_with_time' else 1 # features in input
        self.output_size = len(params['appliances']) # number of features in output
        self.batch_size = params['batch_size']
        self.num_context_point  = params['in_size']
        self.num_target_point = params['out_size']
        
        hidden_dim=params['hidden_dim'] if 'hidden_dim' in params else 32 # size of hidden space
        latent_dim=params['latent_dim'] if 'latent_dim' in params else 32 # size of latent space
        latent_enc_self_attn_type= params['latent_enc_self_attn_type'] if 'latent_enc_self_attn_type' in params else "ptmultihead" # type of attention: "uniform", "dot", "multihead" "ptmultihead": see attentive neural processes paper
        det_enc_self_attn_type= params['det_enc_self_attn_type'] if 'det_enc_self_attn_type' in params else "ptmultihead"
        det_enc_cross_attn_type= params['det_enc_cross_attn_type'] if 'det_enc_cross_attn_type' in params else "ptmultihead"
        n_latent_encoder_layers= params['n_latent_encoder_layers'] if 'n_latent_encoder_layers' in params else 2
        n_det_encoder_layers= params['n_latent_encoder_layers'] if 'n_latent_encoder_layers' in params else 2  # number of deterministic encoder layers
        n_decoder_layers= params['n_decoder_layers'] if 'n_decoder_layers' in params else 2
        min_std=  params['min_std'] if 'min_std' in params else  0.01 # To avoid collapse use a minimum standard deviation, should be much smaller than variation in labels
        dropout= params['dropout'] if 'dropout' in params else 0.3
        attention_dropout= params['attention_dropout'] if 'attention_dropout' in params else 0.5
        batchnorm= params['batchnorm'] if 'batchnorm' in params else True
        attention_layers= params['attention_layers'] if 'attention_layers' in params else 2
        
        
        context_in_target= params['context_in_target'] if 'context_in_target' in params else False
        self.target_norm = params['target_norm'] if 'target_norm' in params else 'z-norm'
       

        self.mean = params['mean'] if 'mean' in params else 0
        self.std = params['std'] if 'std' in params else 1

        # Sometimes input normalisation can be important, an initial batch norm is a nice way to ensure this https://stackoverflow.com/a/46772183/221742
        self.norm_x = BatchNormSequence(n_channels, affine=False)
        self.norm_y = BatchNormSequence(self.output_size, affine=False)
        
        self.norm_h = BatchNormSequence(hidden_dim, affine=False)

        self._use_latent= params['use_latent'] if 'use_latent' in params else False
        self.use_rnn = params['use_rnn'] if 'use_rnn' in params else True

        if self.use_rnn:
            self.__x = LSTMBlock(
                    in_channels=n_channels,
                    out_channels=hidden_dim,
                    num_layers=attention_layers,
                    dropout=dropout
                )
            self.__y = LSTMBlock(
                in_channels=self.output_size,
                out_channels=hidden_dim,
                num_layers=attention_layers,
                dropout=dropout
            )
        else:
            self.__x = create_conv1(in_channels=n_channels, 
                                    out_channels=hidden_dim,
                               kernel_size=5, 
                               stride=1,
                               padding=2, 
                               padding_mode='replicate')
        

            self.__y = create_conv1(in_channels=self.output_size, 
                                    out_channels=hidden_dim,
                               kernel_size=5, 
                               stride =1 ,
                               padding=2, 
                               padding_mode='replicate')
        x_dim = hidden_dim
        y_dim2 = self.output_size            
        
        if self._use_latent:
            self._latent_encoder = LatentEncoder(
                x_dim + y_dim2,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                self_attention_type=latent_enc_self_attn_type,
                n_encoder_layers=n_latent_encoder_layers,
                attention_layers=attention_layers,
                dropout=dropout,
                attention_dropout=attention_dropout,
                batchnorm=batchnorm,
                min_std=min_std,
            )

        self._deterministic_encoder = DeterministicEncoder(
            input_dim=x_dim + y_dim2,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            self_attention_type=det_enc_self_attn_type,
            cross_attention_type=det_enc_cross_attn_type,
            n_d_encoder_layers=n_det_encoder_layers,
            attention_layers=attention_layers,
            dropout=dropout,
            batchnorm=batchnorm,
            attention_dropout=attention_dropout,
        )

        self._decoder = Decoder(
            x_dim,
            self.output_size,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            batchnorm=batchnorm,
            min_std=min_std,
            n_decoder_layers=n_decoder_layers,
            use_latent = self._use_latent

        )
        
        self.r = None
        self.z = None
        

    def forward(self, context_x, context_y, target_x, target_y=None, sample_latent=None):
        if sample_latent is None:
            sample_latent = self.training
        
        
        device = next(self.parameters()).device
        
        # if self.hparams.get('bnorm_inputs', True):
        # https://stackoverflow.com/a/46772183/221742
        
        target_x = self.norm_x(target_x)
        context_x = self.norm_x(context_x)

        # see https://arxiv.org/abs/1910.09323 where x is substituted with h = RNN(x)
        # x need to be provided as [B, T, H]
 
        
        if self.use_rnn:
            target_x = self.__x(target_x)
            context_x = self.__x(context_x)
        else:
            target_x = self.__x(target_x.permute(0,2,1)).permute(0,2,1)
            context_x = self.__x(context_x.permute(0,2,1)).permute(0,2,1)
            
        if self._use_latent:
            mean_prior, dist_prior, log_var_prior = self._latent_encoder(context_x, context_y)
    
            if (target_y is not None):
                target_y2 = target_y
               
                mean_post, dist_post, log_var_post = self._latent_encoder(target_x, target_y2)
                z = dist_post.rsample() if sample_latent else dist_post.loc
                   
            else:
                z = dist_prior.rsample() if sample_latent else dist_prior.loc
            
            num_targets = target_x.size(1)
            self.z = z
            z = z.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T_target, H]
            z = self.norm_h(z)
        else:
            z = None
            
        
        r = self._deterministic_encoder(context_x, context_y,
                                            target_x)  # [B, T_target, H]  
        self.r = r
        
        
        mu, log_sigma, dist, p, s_dist = self._decoder(r, z, target_x)

        y_pred = dist.rsample() if self.training else mu
        
        s_pred = s_dist.rsample() if self.training else p

        if self.target_norm =='lognorm':
            y_pred = nn.Softplus()(y_pred)
                  

        if target_y is not None:
            log_p = dist.log_prob(target_y).mean(-1)
            s_log_p = - s_dist.log_prob(s_pred).mean(-1)
            loss_p = -log_p
            # Weight loss nearer to prediction time?
            weight = (torch.arange(loss_p.shape[1]) + 1).float().to(device)[None, :]
            loss_p_weighted = loss_p / torch.sqrt(weight)  # We want to  weight nearer stuff more
            
            mse_loss = F.mse_loss(y_pred * s_pred, target_y, reduction='none')[:,:context_x.size(1)]
            
            if self._use_latent:
                loss_kl = torch.distributions.kl.kl_divergence(
                    dist_post, dist_prior).mean(-1)  # [B, R].mean(-1)
        
                loss_kl = loss_kl[:, None].expand(log_p.shape)
                loss = (loss_kl + loss_p_weighted + s_log_p).mean() 
                loss_kl = loss_kl.mean()
            else:
                
                loss = loss_p.mean() + s_log_p.mean()
            

            loss_p_weighted = loss_p_weighted.mean()
            
            mse_loss = mse_loss.mean()
            
            log_p = log_p.mean()
            loss_p = loss_p.mean()
        else:
            loss_p = None
            mse_loss = None
            loss_kl = None
            loss = None


        return y_pred * s_pred , dict(loss=loss, 
                            loss_p=loss_p, 
                            loss_mse=mse_loss, 
                            # loss_p_weighted=loss_p_weighted
                            ), dict( log_sigma=log_sigma, y_dist=dist ,  pred = y_pred,
                            s_pred = s_pred)

    def step(self, batch, sequence_type= None):
    
        assert all(torch.isfinite(d).all() for d in batch)
        context_x, context_y, target_x, target_y = batch
        out, losses, extra = self(context_x, context_y, target_x, target_y)
        loss = losses['loss']
        error = (target_y.squeeze(1) - out)
        mae = error.abs().data.mean()
        return  loss, mae   

    def predict(self,  model, test_dataloader):
        
    
        
        net = model.model.eval()
        num_batches = len(test_dataloader)
        values = range(num_batches)
        
        predictions = []
        uncertainty = []
        dist_target = []
        
        self.training = False

        # In the beginning we suppose everything is OFF
        context_y = torch.zeros((1,self.num_context_point, self.output_size))

        with tqdm(total = len(values), file=sys.stdout) as pbar:
            with torch.no_grad():

                for batch_idx, batch in enumerate(test_dataloader):
                    
                    context_x ,target_x = batch

                    for i in range(context_x.shape[0]):

                        pred, losses, extra = net(
                            context_x[i,:,:].unsqueeze(0), 
                            context_y, 
                            target_x[i,:,:].unsqueeze(0), 
                            None)

                        predictions.append(pred)
                        dist_target.append(extra['y_dist'])
                        uncertainty.append(extra['log_sigma'])

                        # prepare the context_y of next sequence 
                        # filling the contextual informtation with new predictions
                        context_y[0,:-self.num_target_point,:] = context_y[0,self.num_target_point:,:].clone()
                        context_y[0,-self.num_target_point:,:] = pred.squeeze().unsqueeze(dim=1)
                    
                    del  batch    
                    pbar.set_description('processed: %d' % (1 + batch_idx))
                    pbar.update(1)  
                
                pbar.close()
        
        pred = torch.cat(predictions, 0)
        uncertainty = torch.cat( uncertainty, 0)

        pred = self.aggregate_seqs_median(pred)
        # Denormalise the output 
        if self.target_norm == 'z-norm':
            
            pred = self.mean + self.std * pred
            pred = torch.tensor(np.where(pred > 0, pred, 0))
        else:
            pred = pred.expm1()
        uncertainty = torch.tensor(self.aggregate_seqs(uncertainty))
        
        results = {"pred":pred, 'uncertanity':uncertainty}
        return results

    
    def aggregate_seqs(self, prediction):
        """
        Aggregate the overleapping sequences using the mean

        Args:
            prediction (tensor[n_samples + window_size +1 1,window_size]): test predictions of the current model

        Returns:
            [type]: [description]
        """
        l = self.num_target_point
        n = prediction.shape[0] + l - 1
        sum_arr = np.zeros((n))
        counts_arr = np.zeros((n))
        o = len(sum_arr)
        for i in range(prediction.shape[0]):
            sum_arr[i:i + l] += prediction[i].reshape(-1).numpy()
            counts_arr[i:i + l] += 1
        for i in range(len(sum_arr)):
            sum_arr[i] = sum_arr[i] / counts_arr[i]
        
        return sum_arr

    

    def aggregate_seqs_median( self, prediction):
        """
        Aggregate the overleapping sequences using the mean

        Args:
            prediction (tensor[n_samples + window_size +1 1,window_size]): test predictions of the current model

        Returns:
            [type]: [description]
        """
        l = self.num_target_point
        n = prediction.shape[0] + l - 1
        sum_arr = np.zeros(n)
        o = len(sum_arr)
        for i in range(prediction.shape[0]):
          seq = []
          j =0
          while ((i-j)>=0 and j<l):
            seq.append(prediction[i-j,j])
            j+=1
          sum_arr[i] = np.median(np.array(seq))

          if i == prediction.shape[0] -1 :
            seq= []
            for j in range( l - 1):
              if j == l-2:
                sum_arr [i+j+1] = prediction[prediction.shape[0]-1, prediction.shape[1]-1]
              else:
                k = j + 1
                seq =[]
                while k<l  :
                  seq.append(prediction[i-k+1,k])
                  k+=1
                sum_arr [i+j+1] = np.median(np.array(seq))

        return sum_arr
    
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
            'use_rnn': trial.suggest_categorical('use_rnn',[True, False]),
            'hidden_dim': trial.suggest_int('hidden_dim', 32, 64 , step=32),
            'use_latent':trial.suggest_categorical('use_latent',[True, False]),
            }

 
