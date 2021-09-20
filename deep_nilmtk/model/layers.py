import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import math


def elu_plus_one_plus_epsilon(x, eps=1e-8, max_value=6.0):
    """ELU activation with a very small addition to help prevent NaN in loss."""
    return torch.clamp((F.elu(x) + 1 + eps), max=max_value)

class create_attn_linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        nn.init.normal_(self.linear.weight, std=in_channels ** -0.5)

    def forward(self, x):
        x = self.linear(x)
        return x


def create_linear(in_channels, out_channels):
    m = nn.Linear(in_channels,out_channels)
    nn.init.kaiming_normal_(m.weight.data)
    return m

def create_conv1(in_channels, out_channels, 
                 kernel_size, bias=True, 
                 stride=2,
                 padding= 0, 
                 padding_mode='zeros'):
    m = nn.Conv1d(in_channels,out_channels, 
                  kernel_size, 
                  bias=bias, 
                  stride=stride,
                  padding=padding,
                  padding_mode=padding_mode)
    nn.init.xavier_normal_(m.weight.data)
    
    return m

def create_deconv1(in_channels, out_channels,
                   kernel_size, bias=True,
                   stride=2, padding = 0, output_padding=0):
    m = nn.ConvTranspose1d(in_channels,
                           out_channels, 
                           kernel_size, 
                           bias=bias, 
                           stride=stride,
                           padding=padding,
                           output_padding=output_padding)
    nn.init.xavier_normal_(m.weight.data)
    return m


    
class ConvLayer(nn.Module):
    def __init__(self, in_size=32, out_channels=[32, 64, 128], 
                 kernel_size=[3,3,3],  bias=True, 
                 bn=False, activation=nn.ReLU(),
                 strides=[1,1,1], pool_filter=16):
        super().__init__()
        self.layers = []
        for i in range(len(out_channels)):
            if i==0:
                layer=create_conv1(in_size, out_channels[i], kernel_size[i], bias=bias, stride=strides[i])
            else:
                layer=create_conv1(out_channels[i-1], out_channels[i], kernel_size[i], bias=bias, stride=strides[i])
            
            self.layers.append(layer)
            self.add_module("cnn_layer_"+str(i+1), layer)  
            if bn: 
                bn = nn.BatchNorm1d(out_channels[i])
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)
            
        self.cnn_network =  nn.Sequential(*self.layers)
        self.pool_filter = pool_filter
        
    def forward(self, x):
        x = F.adaptive_avg_pool1d(self.cnn_network(x), self.pool_filter) 
        return x
    
class MLPLayer(nn.Module):
    def __init__(self, in_size, 
                 out_channels=[32, 64, 128], 
                 output_size=None, 
                 activation=nn.ReLU(),
                 bn=True):
        
        super().__init__()
        self.in_size = in_size
        self.output_size = output_size
        self.layers = []
        for i in range(len(out_channels)):
            if i==0:
                layer=create_linear(in_size, out_channels[i])
            else:
                layer=create_linear(out_channels[i-1], out_channels[i])
        
            self.layers.append(layer)
            self.add_module("mlp_layer_"+str(i+1), layer)  
            if bn: 
                bn = nn.BatchNorm1d(out_channels[i])
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)
        
        self.mlp_network =  nn.Sequential(*self.layers)
        
    def forward(self, x):
        x = self.mlp_network(x)
        return x    
        

class MDGMM(nn.Module):
    def __init__(self, in_dims=1,out_dims=1,
                 kmix=5, dist_type="lognormal", activation=nn.SiLU()):
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dims
        self.kmix = kmix
        self.activation = activation
        self.dist_type=dist_type
        self.lin_feats   = create_linear(self.in_dims,self.in_dims*2)
        self._pi = nn.Linear(self.in_dims*2,self.kmix)
        self._mu = nn.Linear(self.in_dims*2,self.kmix*self.out_dim)
        self._sigma = nn.Linear(self.in_dims*2,self.kmix*self.out_dim)
        self.min_std = 0.001
        
                                     
    def forward(self, x):
        feats = self.activation(self.lin_feats(x))
        pi = torch.softmax(self._pi(feats), -1)
        mu = F.relu(self._mu( feats).reshape(-1,self.kmix, self.out_dim))
        #mu = torch.clamp(mu.exp(), max=7.8)
        #mu = F.softplus(mu)+ 1e-8
        log_var = self._sigma( feats).reshape(-1,self.kmix, self.out_dim)
        log_var = F.logsigmoid(log_var)
        log_var = torch.clamp(log_var, math.log(self.min_std), -math.log(self.min_std))
        sigma = torch.exp(0.5 * log_var)
        #sigma   = F.softplus(log_var)+ 1e-8
        
        
        mix = dist.Categorical(pi)
        if self.dist_type =="laplace":
            comp = dist.Independent(dist.Laplace(mu, sigma), 1)
        elif self.dist_type =="studentT":
            comp = dist.Independent(dist.StudentT(mu, sigma), 1)
        elif self.dist_type =="lognormal":
            comp = dist.Independent(dist.LogNormal(mu, sigma), 1)
        else:
            comp = dist.Independent(dist.Normal(mu, sigma), 1)  

        gmm = dist.MixtureSameFamily(mix, comp)

        return pi, mu, sigma, gmm

class Up(nn.Module):
       
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride=1, activation=nn.ReLU(), pad_out=False):
        super().__init__()
        
        self.upsample = nn.Sequential(create_deconv1(in_channels=in_ch, 
                                       out_channels=out_ch, 
                                       kernel_size=kernel_size, 
                                       stride=stride),
                                      activation)
        
        self.conv = nn.Sequential(create_conv1(in_channels=out_ch+in_ch,
                            out_channels=out_ch, 
                            kernel_size=kernel_size, 
                            padding= kernel_size //2 if pad_out else 0,
                            stride=stride),
                            activation)
        

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        # Pad x1 to the size of x2
        diff = x2.shape[2] - x1.shape[2]
        x1 = F.pad(x1, [diff// 2, diff - diff // 2])
        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)     
        return  x     


class BatchNormSequence(nn.Module):
    """Applies batch norm on features of a batch first sequence."""
    def __init__(
        self, out_channels, **kwargs
    ):
        super().__init__()
        self.norm = nn.BatchNorm1d(out_channels, **kwargs)

    def forward(self, x):
        # x.shape is (Batch, Sequence, Channels)
        # Now we want to apply batchnorm and dropout to the channels. So we put it in shape
        # (Batch, Channels, Sequence) which is what BatchNorm1d expects
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        return x.permute(0, 2, 1)

# Reccurent blocks

class LSTMBlock(nn.Module):
    def __init__(self, 
        in_channels, 
        out_channels, 
        dropout=0.5, 
        batchnorm=True, 
        bias=True, 
        num_layers=1
    ):
        super().__init__()
        self._lstm = nn.LSTM(
                input_size=in_channels,
                hidden_size=out_channels,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bias=bias
        )

    def forward(self, x):
        return self._lstm(x)[0]

# Linear blocks

class NPBlockRelu2d(nn.Module):
    """Block for Neural Processes."""

    def __init__(
        self, in_channels, out_channels, dropout=0, batchnorm=False, bias=True
    ):
        super().__init__()
        self.linear = create_linear(in_channels, out_channels)
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.norm = nn.BatchNorm2d(out_channels) if batchnorm else False

    def forward(self, x):
        # x.shape is (Batch, Sequence, Channels)
        # We pass a linear over it which operates on the Channels
        x = self.act(self.linear(x))

        # Now we want to apply batchnorm and dropout to the channels. So we put it in shape
        # (Batch, Channels, Sequence, None) so we can use Dropout2d & BatchNorm2d
        x = x.permute(0, 2, 1)[:, :, :, None]

        if self.norm:
            x = self.norm(x)

        x = self.dropout(x)
        return x[:, :, :, 0].permute(0, 2, 1)


class BatchMLP(nn.Module):
    """
    Apply MLP to the final axis of a 3D tensor (reusing already defined MLPs).
    Args:
        input: input tensor of shape [B,n,d_in].
        output_sizes: An iterable containing the output sizes of the MLP as defined 
            in `basic.Linear`.
    Returns:
        tensor of shape [B,n,d_out] where d_out=output_size
    """

    def __init__(
        self, input_size, output_size, num_layers=3, dropout=0.7, batchnorm=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.initial = NPBlockRelu2d(
            input_size, output_size, dropout=dropout, batchnorm=batchnorm
        )
        
        self.encoder = nn.Sequential(
            *[
                NPBlockRelu2d(
                    output_size, output_size, dropout=dropout, batchnorm=batchnorm
                )
                for _ in range(num_layers - 2)
            ]
        )
        self.final = create_linear(output_size, output_size)

    def forward(self, x):
        x = self.initial(x)
        x = self.encoder(x)
        return self.final(x)





# Attention blocks


def batch_first_attention(module: nn.MultiheadAttention, k, v, q, **kwargs):
    """
    Batch first attention 
    [batch, seq, hidden] instead of [seq, batch, hidden]
    see https://pytorch.org/docs/stable/nn.html#torch.nn.MultiheadAttention
    """
    assert isinstance(
        module, nn.MultiheadAttention
    ), f"should be nn.MultiheadAttention not {type(module)}"
    q = q.permute(1, 0, 2)
    k = k.permute(1, 0, 2)
    v = v.permute(1, 0, 2)
    attn_output, attn_output_weights = module(query=q, key=k, value=v, **kwargs)
    return attn_output.permute(1, 0, 2).contiguous(), attn_output_weights


class AttnLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        torch.nn.init.normal_(self.linear.weight, std=in_channels ** -0.5)

    def forward(self, x):
        x = self.linear(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        attention_type,
        attention_layers=2,
        n_heads=8,
        x_dim=1,
        rep="mlp",
        dropout=0,
        batchnorm=False,
    ):
        super().__init__()
        self._rep = rep

        if self._rep == "mlp":
            self.batch_mlp_k = BatchMLP(
                x_dim,
                output_size = hidden_dim,
               
            )
            self.batch_mlp_q = BatchMLP(
                x_dim,
                output_size = hidden_dim,
                
            )
        elif self._rep == "lstm":
            self.batch_lstm_k = LSTMBlock(x_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout, num_layers=attention_layers)
            self.batch_lstm_q = LSTMBlock(x_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout, num_layers=attention_layers)

        if attention_type == "uniform":
            self._attention_func = self._uniform_attention
        elif attention_type == "laplace":
            self._attention_func = self._laplace_attention
        elif attention_type == "dot":
            self._attention_func = self._dot_attention
        elif attention_type == "multihead":
            self._W_k = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W_v = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W_q = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W = AttnLinear(n_heads * hidden_dim, hidden_dim)
            self._attention_func = self._multihead_attention
            self.n_heads = n_heads
        elif attention_type == "ptmultihead":
            self._W = torch.nn.MultiheadAttention(
                hidden_dim, n_heads, bias=False, dropout=dropout
            )
            self._attention_func = self._pytorch_multihead_attention
        else:
            raise NotImplementedError

    def forward(self, k, v, q):
        if self._rep == "mlp":
            k = self.batch_mlp_k(k)
            q = self.batch_mlp_q(q)
        elif self._rep == "lstm":
            k = self.batch_lstm_k(k)
            q = self.batch_lstm_q(q)
        rep = self._attention_func(k, v, q)
        return rep

    def _uniform_attention(self, k, v, q):
        total_points = q.shape[1]
        rep = torch.mean(v, dim=1, keepdim=True)
        rep = rep.repeat(1, total_points, 1)
        return rep

    def _laplace_attention(self, k, v, q, scale=0.5):
        k_ = k.unsqueeze(1)
        v_ = v.unsqueeze(2)
        unnorm_weights = torch.abs((k_ - v_) * scale)
        unnorm_weights = unnorm_weights.sum(dim=-1)
        weights = torch.softmax(unnorm_weights, dim=-1)
        rep = torch.einsum("bik,bkj->bij", weights, v)
        return rep

    def _dot_attention(self, k, v, q):
        scale = q.shape[-1] ** 0.5
        unnorm_weights = torch.einsum("bjk,bik->bij", k, q) / scale
        weights = torch.softmax(unnorm_weights, dim=-1)

        rep = torch.einsum("bik,bkj->bij", weights, v)
        return rep

    def _multihead_attention(self, k, v, q):
        outs = []
        for i in range(self.n_heads):
            k_ = self._W_k[i](k)
            v_ = self._W_v[i](v)
            q_ = self._W_q[i](q)
            out = self._dot_attention(k_, v_, q_)
            outs.append(out)
        outs = torch.stack(outs, dim=-1)
        outs = outs.view(outs.shape[0], outs.shape[1], -1)
        rep = self._W(outs)
        return rep

    def _pytorch_multihead_attention(self, k, v, q):
        # Pytorch multiheaded attention takes inputs if diff order and permutation
        return batch_first_attention(self._W, q=q, k=k, v=v)[0]
