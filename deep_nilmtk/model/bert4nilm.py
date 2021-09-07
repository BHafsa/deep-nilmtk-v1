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

from sklearn.metrics import confusion_matrix

def acc_precision_recall_f1_score(pred, status):
    assert pred.shape == status.shape

    pred = pred.reshape(-1, pred.shape[-1])
    status = status.reshape(-1, status.shape[-1])
    accs, precisions, recalls, f1_scores = [], [], [], []

    for i in range(status.shape[-1]):
        tn, fp, fn, tp = confusion_matrix(status[:, i], pred[:, i], labels=[
                                          0, 1]).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / np.max((tp + fp, 1e-9))
        recall = tp / np.max((tp + fn, 1e-9))
        f1_score = 2 * (precision * recall) / \
            np.max((precision + recall, 1e-9))

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return np.array(accs).mean(), np.array(precisions).mean(), np.array(recalls).mean(), np.array(f1_scores).mean()

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class LayerNorm(nn.Module):
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
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
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
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = create_linear(d_model, d_ff)
        self.w_2 = create_linear(d_ff, d_model)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.activation(self.w_1(x)))


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.layer_norm(x + self.dropout(sublayer(x)))


class TransformerBlock(nn.Module):
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
    
    BERT4NILM implementation. 
    Original paper can be found here: https://dl.acm.org/doi/pdf/10.1145/3427771.3429390
    Original code can be found here: https://github.com/Yueeeeeeee/BERT4NILM

    .. _bert:

    """
    def __init__(self, params):

        super().__init__()
       

        self.original_len = params['in_size'] if 'in_size' in params else 99
        self.output_size = len(params['appliances']) if 'appliances' in params else 1
        self.stride = params['stride'] if 'stride' in params else 1

        # this is only valide for a single appliance usage
        # The original mode was proposed for several appliances
        self.cutoff = [params['cutoff'][params['appliances'][0]]] if 'cutoff' in params else None
        self.threshold = [params['threshold'][params['appliances'][0]]] if 'threshold' in params else None
        self.min_on = [params['min_on'][params['appliances'][0]]] if 'min_on' in params else None
        self.min_off = [params['min_off'][params['appliances'][0]]] if 'min_off' in params else None
        
        self.C0 = [params['lambda'][params['appliances'][0]]] if 'lambda' in params else [1e-6]
        
      
        
        self.set_hpramas(self.cutoff, self.threshold, self.min_on, self.min_off, self.C0)

#         self.C0 = torch.tensor([params['c0'] if 'c0' in params else 1.]).float()

        self.latent_len = int(self.original_len / 2)
        self.dropout_rate = params['dropout']

        self.hidden = params['hidden'] if 'hidden' in params else 256
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
            in_channels=self.hidden, out_channels=self.hidden, kernel_size=4, stride=2, padding=1)
        self.linear1 = create_linear(self.hidden, 128)
        self.linear2 = create_linear(128, self.output_size)

        self.activation = nn.Sigmoid()

#         self.truncated_normal_init()

        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss(reduction='mean')
        self.margin = nn.SoftMarginLoss()
        self.l1_on = nn.L1Loss(reduction='sum')


    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        params = list(self.named_parameters())
        for n, p in params:
            if 'layer_norm' in n:
                continue
            else:
                with torch.no_grad():
                    l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
                    u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)

    def set_hpramas(self,cutoff, threshold, min_on, min_off, C0):
        if cutoff is not None:
            self.cutoff = torch.tensor(cutoff).float()
        if threshold is not None:
            self.threshold = torch.tensor(threshold).float()
        if min_on is not None:
            self.min_on = torch.tensor(min_on).float()
        if min_off is not None:
            self.min_off = torch.tensor(min_off).float()
            
        if C0 is not None:
            self.C0 = torch.tensor(C0).float()

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
        seqs, labels_energy, status = batch
        
#         print(labels_energy.mean())
#         print(labels_energy.min())
#         print(labels_energy.max())
#         print('==================')
        
        batch_shape = status.shape
        logits = self.forward(seqs.float())
            
        labels = labels_energy / self.cutoff.to(seqs.device) # This the normalisation of the output data ?!!!
        
#         print((logits * self.cutoff.to(seqs.device)).mean())
#         print((logits * self.cutoff.to(seqs.device)).min())
#         print((logits * self.cutoff.to(seqs.device)).max())
#         print('==================')
        
        logits_energy = self.cutoff_energy(logits * self.cutoff.to(seqs.device))
        logits_status = self.compute_status(logits_energy)
            
        mask = (status >= 0).to(seqs.device)
        labels_masked = torch.masked_select(labels, mask).view((-1, batch_shape[-1]))
        logits_masked = torch.masked_select(logits, mask).view((-1, batch_shape[-1]))
        status_masked = torch.masked_select(status, mask).view((-1, batch_shape[-1]))
        logits_status_masked = torch.masked_select(logits_status, mask).view((-1, batch_shape[-1]))

        # Calculating the Loss function 
        kl_loss = self.kl(torch.log(F.softmax(logits_masked.squeeze() / 0.1, dim=-1) + 1e-9), F.softmax(labels_masked.squeeze() / 0.1, dim=-1))
        mse_loss = self.mse(logits_masked.contiguous().view(-1).float(),
            labels_masked.contiguous().view(-1).float())
        margin_loss = self.margin((logits_status_masked * 2 - 1).contiguous().view(-1).float(), 
            (status_masked * 2 - 1).contiguous().view(-1).float())
        total_loss = kl_loss + mse_loss + margin_loss
        #  
        on_mask = (status >= 0) * (((status == 1) + (status != logits_status.reshape(status.shape))) >= 1)
        if on_mask.sum() > 0:
            total_size = torch.tensor(on_mask.shape).prod()
            logits_on = torch.masked_select(logits.reshape(on_mask.shape), on_mask)
            labels_on = torch.masked_select(labels.reshape(on_mask.shape), on_mask)
            loss_l1_on = self.l1_on(logits_on.contiguous().view(-1), 
                labels_on.contiguous().view(-1))
            total_loss += self.C0.to(seqs.device)[0]  * loss_l1_on / total_size
            
        mae = self.mae(logits_masked.contiguous().view(-1).float(),
            labels_masked.contiguous().view(-1).float())
        
        

               
        return  total_loss, mae

    def cutoff_energy(self, data):
        columns = data.squeeze().shape[-1]

        if self.cutoff.size(0) == 0:
            self.cutoff = torch.tensor(
                [3100 for i in range(columns)])

        data[data < 5] = 0
    
        data = torch.min(data, self.cutoff.float().to(data.device))
        return data

    def compute_status(self, data):
        data_shape = data.shape
        columns = data.shape[-1]

        if self.threshold.size(0) == 0:
            self.threshold = torch.tensor(
                [10 for i in range(columns)])
        
        status = (data >= self.threshold.to(data.device)) * 1

        return status  
    

    
    def predict(self,  model, test_dataloader):

        net = model.model.eval()
        num_batches = len(test_dataloader)
        values = range(num_batches)
        
        pred = []
        
        e_pred_curve = []
        s_pred_curve = []

        
        with tqdm(total = len(values), file=sys.stdout) as pbar:
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    seqs = batch

                    logits = self.forward(seqs.float())

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
        e_pred_curve = torch.cat(e_pred_curve, 0).float()
        s_pred_curve = torch.cat(s_pred_curve, 0).float()
        
        e_pred_curve = self.aggregate_seqs(e_pred_curve.squeeze())
        s_pred_curve = self.aggregate_seqs(s_pred_curve.squeeze())
        
        results = {
            "pred":e_pred_curve, 
            "s_pred_curve":s_pred_curve
            }
        
        return results
    
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

    def aggregate_seqs(self, prediction):
        """
        Aggregate the overleapping sequences using the mean
        taking into consideration the stride size

        Args:
            prediction (tensor[n_samples + window_size -1 ,window_size]): test predictions of the current model

        Returns:
            [type]: [description]
        """
        l = self.original_len
        s = self.stride
        n = (prediction.shape[0] -1) * self.stride + l # this is yo take into consideration the stride
        
        sum_arr = np.zeros((n))
        counts_arr = np.zeros((n))
        o = len(sum_arr)
        
        for i in range(prediction.shape[0]):
            sum_arr[i*s:i*s + l] += prediction[i].reshape(-1).numpy()
            counts_arr[i*s:i*s + l] += 1
            
        for i in range(len(sum_arr)):
            sum_arr[i] = sum_arr[i] / counts_arr[i]
        
        return torch.tensor(sum_arr)
        
        
