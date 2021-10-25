# The original code was taken from the 
# the repo: https://github.com/Yueeeeeeee/BERT4NILM/
# Paper Link: https://dl.acm.org/doi/pdf/10.1145/3427771.3429390

import math
import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
import sys

from .layers import create_linear, create_conv1, create_deconv1

from torch.nn import TransformerEncoderLayer

import numpy as np
import mlflow



class GELU(nn.Module):
    """
    Gaussian Error Linear Units GLU
    """
    def forward(self, x):
        """
        """
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEmbedding(nn.Module):
    """
    Positional Embedding 
    
    :param max_len: maximum length of the input
    :type max_len: int
    :param d_model: dimension of the model
    :type d_model: int

    """
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class LayerNorm(nn.Module):
    """
    Normalization layer

    :param features: The number of input features
    :type features: int
    :param eps: Regularization factor, defaults to 1e-6
    :type eps: float, optional
    """
    def __init__(self, features, eps=1e-6):
   
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class Attention(nn.Module):
    """
    Attention layer

    :param query: Query values
    :type query: tensor
    :param key: Key values
    :type key: tensor
    :param value: Values
    :type value: tensor
    :param mask: Mask for a causal model, defaults to None
    :type mask: tensor, optional
    :param dropout: Dropout, defaults to None
    :type dropout: float, optional
    :return:  output of the attention layer and attention score
    :rtype: tuple(tensor, tensor)
    """
    def forward(self, query, key, value, mask=None, dropout=None):
     
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi headed attention layer

    :param h: The number of heads
    :type h: int
    :param d_model: The dimension of the model
    :type d_model: int
    :param dropout: Dropout, defaults to 0.1
    :type dropout: float, optional

    .. note:
       d_model should be multiple of h.
        
    """
    def __init__(self, h, d_model, dropout=0.1):
        
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([create_linear(d_model, d_model) for _ in range(3)])
        self.output_linear = create_linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    """
    Calculates the position wise feed forward

    :param d_model: The dimension of the model
    :type d_model: int
    :param d_ff: size of hidden layer
    :type d_ff: int
    """
    def __init__(self, d_model, d_ff):

        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = create_linear(d_model, d_ff)
        self.w_2 = create_linear(d_ff, d_model)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.activation(self.w_1(x)))


class SublayerConnection(nn.Module):
    """
    Performs the addition and layer normalisation
    More details can be found https://arxiv.org/pdf/1706.03762.pdf

    :param size: the size of teh input
    :type size: int
    :param dropout: Dropout
    :type dropout: float
    """
    def __init__(self, size, dropout):
      
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.layer_norm(x + self.dropout(sublayer(x)))


class TransformerBlock(nn.Module):
    """
    Tranformer decoder block.

    :param hidden: Dimension of the model 
    :type hidden: int
    :param attn_heads: The number of attention heads
    :type attn_heads: int
    :param feed_forward_hidden: The hidden size of feedforward layer
    :type feed_forward_hidden: int
    :param dropout: Dropout
    :type dropout: float
    """
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(
            h=attn_heads, d_model=hidden, dropout=dropout)
        
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden)
        
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BERT4NILM(nn.Module):
    """
    .. _bert:

    BERT4NILM implementation. 
    Original paper can be found here: https://dl.acm.org/doi/pdf/10.1145/3427771.3429390
    Original code can be found here: https://github.com/Yueeeeeeee/BERT4NILM


    The hyperparameter dictionnary is expected to include the following parameters
    
    :param threshold:  The threshold for states generation in the target power consumption, defaults to None
    :type threshold: List of floats
    :param cutoff: The cutoff for states generation in the target power consumption, defaults to None
    :type cutoff: List of floats
    :param min_on: The min on duration for states generation in the target power consumption, defaults to None
    :type min_on: List of floats
    :param min_off: The min off duration for states generation in the target power consumption, defaults to None
    :type min_off: List of floats
    :param in_size: The length of the input sequence, defaults to 488.
    :type in_size: int
    :param stride: The distance between two consecutive sequences, defaults to 1.
    :type stride: int
    :param hidden: The hidden size, defaults to 256
    :type hidden: int
    :param heads: The number of attention heads in each transformer block, defaults to 2
    :type heads: int
    :param n_layers: the number of transformer blocks in the model, defaults to 2
    :type n_layers: int
    :params dropout: The dropout, defaults to 0.2
    :type dropout: float

    it can be used as follow:

    .. code-block:: python

        'Bert4NILM': NILMExperiment({
                  "model_name": 'BERT4NILM', 
                  'in_size': 480, 
                  'feature_type':'main',
                  'stride':10,
                  'max_nb_epochs':1,
                  'cutoff':{
                      'aggregate': 6000,
                      'kettle': 3100,
                      'fridge': 300,
                      'washing machine': 2500,
                      'microwave': 3000,
                      'dishwasher': 2500
                  },
                  'threshold':{
                     'kettle': 2000,
                     'fridge': 50,
                     'washing machine': 20,
                     'microwave': 200,
                     'dishwasher': 10
                  },
                  'min_on':{
                    'kettle': 2,
                    'fridge': 10,
                    'washing machine': 300,
                    'microwave': 2,
                    'dishwasher': 300
                  },
                  'min_off':{
                      'kettle': 0,
                      'fridge': 2,
                      'washing machine': 26,
                      'microwave': 5,
                      'dishwasher': 300
                  },
                })

    """
    def __init__(self, params):

        super().__init__()
       

        self.original_len = params['in_size'] if 'in_size' in params else 99
        self.output_size = len(params['appliances']) if params['multi_appliance'] else 1
        self.stride = params['stride'] if 'stride' in params else 1

        # The original mode was proposed for several appliances
        if params['multi_appliance']:
            self.threshold = [params['threshold'][app] for app in params['appliances']] if 'threshold' in params else None

            self.cutoff = [params['cutoff'][app] for app in params['appliances']] if 'cutoff' in params else None

            self.min_on = [params['min_on'][app] for app in params['appliances']] if 'min_on' in params else None
            self.min_off = [params['min_off'][app] for app in params['appliances']] if 'min_off' in params else None
        else:
            self.threshold = [params['threshold'][params['appliances'][0]]] if 'threshold' in params else None

            self.cutoff = [params['cutoff'][params['appliances'][0]]] if 'cutoff' in params else None

            self.min_on = [params['min_on'][params['appliances'][0]]] if 'min_on' in params else None
            self.min_off = [params['min_off'][params['appliances'][0]]] if 'min_off' in params else None
        
        # self.C0 = [params['lambda'][params['appliances'][0]]] if 'lambda' in params else [1e-6]
        
        self.main_mu = params['main_mu']
        self.main_std = params['main_std']
        
        self.set_hpramas(self.cutoff, self.threshold, self.min_on, self.min_off)

        self.C0 = torch.tensor([params['c0'] if 'c0' in params else 1.])

        self.latent_len = int(self.original_len / 2)
        self.dropout_rate = params['dropout'] if 'dropout' in params else 0.2

        self.hidden = params['hidden'] if 'hidden' in params else 32
        self.heads = params['heads'] if 'heads' in params else 2
        self.n_layers = params['n_layers'] if 'n_layers' in params else 2


        self.conv = create_conv1(in_channels=1, out_channels=self.hidden,
                               kernel_size=5, stride=1, padding=2, padding_mode='replicate')

        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)

        self.position = PositionalEmbedding(
            max_len=self.latent_len, d_model=self.hidden)
        self.layer_norm = LayerNorm(self.hidden)
        self.dropout = nn.Dropout(p=self.dropout_rate)

#         self.transformer_blocks = nn.ModuleList([TransformerBlock(
#             self.hidden, self.heads, self.hidden * 4, self.dropout_rate) for _ in range(self.n_layers)])
        
        self.transformer_blocks = nn.ModuleList([TransformerEncoderLayer(
            self.hidden, self.heads, self.hidden * 4, self.dropout_rate) for _ in range(self.n_layers)])

        self.deconv = create_deconv1(
            in_channels=self.hidden, out_channels=self.hidden, kernel_size=4, stride=2, padding= 1, output_padding= 0 if self.original_len % 2 ==0 else 1)
        self.linear1 = create_linear(self.hidden, 128)
        self.linear2 = create_linear(128, self.output_size)

        self.activation = nn.Sigmoid()



        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss(reduction='mean')
        self.margin = nn.SoftMarginLoss()
        self.l1_on = nn.L1Loss(reduction='sum')



    

    def forward(self, sequence):
        x_token = self.pool(self.conv(sequence.unsqueeze(1))).permute(0, 2, 1)

        embedding = x_token + self.position(sequence)
        x = self.dropout(self.layer_norm(embedding))
       
        mask = None
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        x = self.deconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)
        
        return x
        

    def step(self, batch, seq_type=None):
        """Disaggregates a batch of data

        :param batch: A batch of data.
        :type batch: Tensor
        :return: loss function as returned form the model and MAE as returned from the model.
        :rtype: tuple(float,float)
        """
        seqs, labels_energy, status = batch
        
        batch_shape = status.shape
        
        logits = self.forward(seqs)
            
        labels = labels_energy / self.cutoff.to(seqs.device) # This the normalization of the output data ?!!!
        
        
        logits_energy = self.cutoff_energy(logits * self.cutoff.to(seqs.device))
        logits_status = self.compute_status(logits_energy)
            
        mask = (status > 0).to(seqs.device)
       
        labels_masked = torch.masked_select(labels, mask).view((-1, batch_shape[-1])).float()
        logits_masked = torch.masked_select(logits, mask).view((-1, batch_shape[-1])).float()
        status_masked = torch.masked_select(status, mask).view((-1, batch_shape[-1])).float()
        logits_status_masked = torch.masked_select(logits_status, mask).view((-1, batch_shape[-1])).float()

        # Calculating the Loss function 
        kl_loss = self.kl(torch.log(F.softmax(logits_masked.squeeze() / 0.1, dim=-1) + 1e-9), F.softmax(labels_masked.squeeze() / 0.1, dim=-1))
        mse_loss = self.mse(logits_masked.contiguous().view(-1),
            labels_masked.contiguous().view(-1))
            
        margin_loss = self.margin((logits_status_masked * 2 - 1).contiguous().view(-1), 
            (status_masked * 2 - 1).contiguous().view(-1))

        total_loss = kl_loss + mse_loss + margin_loss
        on_mask = (status >= 0) * (((status == 1) + (status != logits_status.reshape(status.shape))) >= 1)
        if on_mask.sum() > 0:
            total_size = torch.tensor(on_mask.shape).prod()
            logits_on = torch.masked_select(logits.reshape(on_mask.shape), on_mask)
            labels_on = torch.masked_select(labels.reshape(on_mask.shape), on_mask)
            loss_l1_on = self.l1_on(logits_on.contiguous().view(-1), 
                labels_on.contiguous().view(-1))
            total_loss += self.C0.to(seqs.device)[0]  * loss_l1_on / total_size
            
        mae = self.mae(logits_masked.contiguous().view(-1),
            labels_masked.contiguous().view(-1))
        
        return  total_loss, mae
    
    def set_hpramas(self,cutoff, threshold, min_on, min_off):
        """
        Setter for the hyper-parameters related to appliance state generation

        :param cutoff: The power cutoff
        :type cutoff: float
        :param threshold: Threshold of target power consumption
        :type threshold: float
        :param min_on: Minimum on duration
        :type min_on: float
        :param min_off: Minimum off duration
        :type min_off: float
        """
        if cutoff is not None:
            self.cutoff = torch.tensor(cutoff)
        if threshold is not None:
            self.threshold = torch.tensor(threshold)
        if min_on is not None:
            self.min_on = torch.tensor(min_on)
        if min_off is not None:
            self.min_off = torch.tensor(min_off)

    def cutoff_energy(self, data):
        """
        Removes the spikes from the data

        :param data: Power consumption
        :type data: tesnor
        :return: Updated ower consumption 
        :rtype: tensor
        """
        columns = data.squeeze().shape[-1]

        if self.cutoff.size(0) == 0:
            self.cutoff = torch.tensor(
                [3100 for i in range(columns)])

        data[data < 5] = 0
    
        data = torch.min(data, self.cutoff.to(data.device))
        return data

    def compute_status(self, data):
        """
        Calculates teh states for the  target data based on the threshold

        :param data: The target data
        :type data: tensor
        :return: The operational states
        :rtype: tensor
        """
        data_shape = data.shape
        columns = data.shape[-1]

        if self.threshold.size(0) == 0:
            self.threshold = torch.tensor(
                [10 for i in range(columns)])
        
        status = (data >= self.threshold.to(data.device)) * 1

        return status  
    

    
    def predict(self,  model, test_dataloader):
        """Generates predictions for the test data loader

        :param model: Pre-trained model
        :type model: nn.Module
        :param test_dataloader: The test data
        :type test_dataloader: dataLoader
        :return: Generated predictions
        :rtype: dict
        """

        net = model.model.eval()
        num_batches = len(test_dataloader)
        values = range(num_batches)
        
        pred = []
        
        e_pred_curve = []
        s_pred_curve = []

        true = []

        with tqdm(total = len(values), file=sys.stdout) as pbar:
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    seqs = batch

                    logits = self.forward(seqs)

                    true.append(seqs)

                    logits_energy = self.cutoff_energy(logits * self.cutoff.to(seqs.device)) # Denormalization
                    logits_status = self.compute_status(logits_energy)
                    
                    status = (logits_status > 0) * 1
                    
                    s_pred_curve.append(status )
                    e_pred_curve.append(logits_energy * status) 
                    
                    del  batch    
                    pbar.set_description('processed: %d' % (1 + batch_idx))
                    pbar.update(1)  
                
                pbar.close()

        # TODO: Denormalisation !!! Previously done ?
        e_pred_curve = torch.cat(e_pred_curve, 0)
        s_pred_curve = torch.cat(s_pred_curve, 0)

        true = torch.cat(true, 0)
        true = true * self.main_std +self.main_mu

        e_pred_curve = self.aggregate_seqs(e_pred_curve.squeeze())
        s_pred_curve = self.aggregate_seqs(s_pred_curve.squeeze())
        true = self.aggregate_seqs(true.squeeze())

        e_pred_curve[e_pred_curve>true] = 0
        s_pred_curve[e_pred_curve>true] = 0
        
        results = {
            "pred":e_pred_curve, 
            "s_pred_curve":s_pred_curve
            }
        
        return results
    


    def aggregate_seqs(self, prediction):
        """ Aggregate the overleapping sequences using the mean
        taking into consideration the stride size

        :param prediction: test predictions of the current model
        :type prediction: tensor
        :return: Aggregted sequence
        :rtype: tensor
        """

        l = self.original_len
        s = self.stride
        n = (prediction.shape[0] -1) * self.stride + l # this is yo take into consideration the stride
        
        sum_arr = np.zeros((n, self.output_size))
        counts_arr = np.zeros((n, self.output_size))
        o = len(sum_arr)

        if len(prediction.shape) == 2:
            prediction = prediction.unsqueeze(dim=2)

        for i in range(prediction.shape[0]):
            sum_arr[i*s:i*s + l,:] += prediction[i,:,:].numpy()
            counts_arr[i*s:i*s + l,:] += 1
            
        for i in range(len(sum_arr)):
            sum_arr[i,:] = sum_arr[i,:] / counts_arr[i,:]
        
        return torch.tensor(sum_arr)
        
        
