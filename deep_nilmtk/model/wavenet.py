import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .baselines import S2S

# Original script https://github.com/jiejiang-jojo/fast-seq2point/


def init_layer_2(layer):
    """Initialize a Linear or Convolutional layer. """

    #nn.init.xavier_uniform_(layer.weight)
    layer.weight.data.uniform_(-0.1, 0.1)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """

    #nn.init.xavier_uniform_(layer.weight)

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 3:
        (n_out, n_in, width) = layer.weight.size()
        n = n_in * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)


class CNN3(nn.Module):

    def __init__(self, seq_len=41, to_binary=False):

        super(CNN3, self).__init__()
        assert (seq_len - 1) % 2 == 0, f'seq_len ({seq_len}) must be odd'
        self.seq_len = seq_len
        self.to_binary = to_binary
        self.kernel_size = (seq_len - 1) // 2 + 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(self.kernel_size, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(self.kernel_size, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.conv_final = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv_final)

    def forward(self, input):

        x = input
        x = x.view(x.shape[0], 1, x.shape[1], 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.conv_final(x)
        x = x.view(x.shape[0], x.shape[2])
        if self.to_binary:
            return torch.sigmoid(x)
        return x


class CNN7(nn.Module):

    def __init__(self, seq_len=253, to_binary=False):

        super(CNN7, self).__init__()
        self.seq_len = seq_len
        self.to_binary = to_binary
        assert (seq_len - 1) % 6 == 0, f'seq_len ({seq_len}) - 1 must be divisible by 6'
        self.kernel_size = (seq_len - 1) // 6 + 1

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(self.kernel_size, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(self.kernel_size, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(self.kernel_size, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(self.kernel_size, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(self.kernel_size, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(self.kernel_size, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.conv_final = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.conv5)
        init_layer(self.conv6)
        init_layer(self.conv_final)

    def forward(self, input):

        x = input
        x = x.view(x.shape[0], 1, x.shape[1], 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = self.conv_final(x)
        x = x.view(x.shape[0], x.shape[2])
        if self.to_binary:
            return torch.sigmoid(x)

        return x


class CNN5(nn.Module):

    def __init__(self, seq_len=15, to_binary=False):

        super(CNN5, self).__init__()
        self.seq_len = seq_len
        self.to_binary = to_binary
        assert (seq_len - 3) % 4 == 0, f'seq_len ({seq_len}) - 3 should be divisible by 4'
        self.kernel_size = (seq_len - 3) // 4

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(self.kernel_size, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(self.kernel_size, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(self.kernel_size, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(self.kernel_size, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(7, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.conv_final = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.conv5)
        init_layer(self.conv_final)

    def forward(self, input):

        x = input
        x = x.view(x.shape[0], 1, x.shape[1], 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = self.conv_final(x)
        x = x.view(x.shape[0], x.shape[2])
        if self.to_binary:
            return torch.sigmoid(x)

        return x


class DilatedResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation_channels, skip_channels, kernel_size, dilation, bias):
        super(DilatedResidualBlock, self).__init__()
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.dilation = dilation
        self.dilated_conv = nn.Conv1d(residual_channels, 2 * dilation_channels, kernel_size=kernel_size, dilation=dilation, padding=dilation)

        self.mixing_conv = nn.Conv1d(dilation_channels, residual_channels + skip_channels, kernel_size=1, bias=False)
        self.init_weights()

    def init_weights(self):
        init_layer(self.dilated_conv)
        init_layer(self.mixing_conv)

    def forward(self, data_in):
      
        out = self.dilated_conv(data_in)
       
        out1 = out.narrow(-2, 0, self.dilation_channels)
        out2 = out.narrow(-2, self.dilation_channels, self.dilation_channels)
        tanh_out = torch.tanh(out1)
        sigm_out = torch.sigmoid(out2)
        data = torch.mul(tanh_out, sigm_out)
        data = self.mixing_conv(data)
        res = data.narrow(-2, 0, self.residual_channels)
        skip = data.narrow(-2, self.residual_channels, self.skip_channels)
        
        res = res + data_in
        
        return res, skip
        
        
class DilatedResidualBlock2(nn.Module):
    def __init__(self, residual_channels, dilation_channels, skip_channels, kernel_size, dilation, bias):
        super(DilatedResidualBlock2, self).__init__()
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        
        self.dilated_conv_tanh = nn.Conv1d(residual_channels, dilation_channels, kernel_size=kernel_size, dilation=dilation, padding=dilation, bias=bias)
        self.dilated_conv_sigmoid = nn.Conv1d(residual_channels, dilation_channels, kernel_size=kernel_size, dilation=dilation, padding=dilation, bias=bias)
        self.res = nn.Conv1d(dilation_channels, residual_channels, kernel_size=1, bias=True)
        self.skip = nn.Conv1d(dilation_channels, skip_channels, kernel_size=1, bias=True)
        
        self.init_weights()

    def init_weights(self):
        init_layer(self.dilated_conv_tanh)
        init_layer(self.dilated_conv_sigmoid)
        init_layer(self.res)
        init_layer(self.skip)

    def forward(self, data_in):

        out1 = self.dilated_conv_tanh(data_in)
        out2 = self.dilated_conv_sigmoid(data_in)
        tanh_out = torch.tanh(out1)
        sigm_out = torch.sigmoid(out2)
        data = F.mul(tanh_out, sigm_out)
        skip = self.skip(data)
        res = self.res(data) + data_in
        return res, skip


class WaveNet(S2S):
    """
    .. _wavenilm:
    WaveNet model for load disaggregtion using residual and dilated convolutions.
    This model a sequence to subsequence model where:
    Output_sequence_length = Input_sequence_length - L
    L = (2 ** layers - 1) * (kernel_size - 1) + 1
    
    """

    def __init__(self, 
        params):
        
        super(WaveNet, self).__init__(params)
        
        self.kernel_size = params['kernel_size'] if 'kernel_size' in params else 3 # has to be odd integer, since even integer may break dilated conv output size
        self.layers = params['layers'] if 'layers' in params else 6
        
        assert self.kernel_size % 2 == 1, f'kernel_size ({self.kernel_size}) must be odd'

        self.to_binary = params['to_binary'] if 'to_binary' in params else False

        self.n_appliances = len(params['appliances']) if 'appliances' in params else 1
        
        self.in_features = 1 # TODO: fix this whith according to input features

        self.seq_len = (2 ** self.layers - 1) * (self.kernel_size - 1) + 1

        self.residual_channels = params['residual_channels'] if 'residual_channels' in params else 32
        self.dilation_channels = params['dilation_channels'] if 'dilation_channels' in params else 32
        self.skip_channels = params['skip_channels'] if 'skip_channels' in params else 32

        self.causal_conv = nn.Conv1d(1, self.residual_channels, kernel_size=1, bias=True)
        self.blocks = [DilatedResidualBlock(self.residual_channels, self.dilation_channels, self.skip_channels, self.kernel_size, 2**i, True)
                       for i in range(self.layers)]
        for i, block in enumerate(self.blocks):
            self.add_module(f"dilatedConv{i}", block)
        self.penultimate_conv = nn.Conv1d(self.skip_channels, self.skip_channels, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, bias=True)
        self.final_conv = nn.Conv1d(self.skip_channels, self.n_appliances, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.causal_conv)
        init_layer(self.penultimate_conv)
        init_layer(self.final_conv)

    def forward(self, data_in):
        
        data_in = data_in.view(data_in.shape[0], self.in_features, data_in.shape[1])
        data_out = self.causal_conv(data_in)
        skip_connections = []
        for block in self.blocks:
            data_out, skip_out = block(data_out)
            skip_connections.append(skip_out)
            
        skip_out = skip_connections[0]
        for skip_other in skip_connections[1:]:
            skip_out = skip_out + skip_other
        data_out = F.relu(skip_out)
        
        data_out = self.penultimate_conv(data_out)
        #data_out = F.relu(data_out)
        data_out = self.final_conv(data_out)
        
        # data_out = data_out.narrow(-1, self.seq_len//2, data_out.size()[-1]-self.seq_len)
        # data_out = data_out.view(data_out.shape[0], data_out.shape[2])
        
        if self.to_binary:
            return torch.sigmoid(data_out)

        return data_out
    

class WaveNetBGRU(nn.Module):
    """
    WaveNet model with a GRU 
    :param nn: [description]
    :type nn: [type]
    :return: [description]
    :rtype: [type]
    """

    def __init__(self, layers=6, kernel_size=3, residual_channels=32, dilation_channels=32, skip_channels=32):
        super(WaveNetBGRU, self).__init__()
        assert kernel_size % 2 == 1, f'kernel_size ({kernel_size}) must be odd'
        self.kernel_size = kernel_size # has to be odd integer, since even integer may break dilated conv output size
        self.seq_len = (2 ** layers - 1) * (kernel_size - 1) + 1

        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels

        self.causal_conv = nn.Conv1d(1, residual_channels, kernel_size=1, bias=False)
        self.blocks = [DilatedResidualBlock(residual_channels, dilation_channels, skip_channels, kernel_size, 2**i, True)
                       for i in range(layers)]
        for i, block in enumerate(self.blocks):
            self.add_module(f"dilatedConv{i}", block)
        self.penultimate_conv = nn.Conv1d(skip_channels, skip_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True)
        self.final_conv = nn.Conv1d(skip_channels, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True)

        self.bgru = nn.GRU(input_size=skip_channels, hidden_size=skip_channels, num_layers=2, bias=True, batch_first=True, dropout=0., bidirectional=True)

        self.fc_final = nn.Linear(2 * skip_channels, 1)

        self.init_weights()

    def _init_param(self, param):

        if param.ndimension() == 1:
            param.data.fill_(0.)

        elif param.ndimension() == 2:
            n = param.size(-1)
            std = math.sqrt(2. / n)
            scale = std * math.sqrt(3.)
            param.data.uniform_(-scale, scale)

    def init_weights(self):
        init_layer(self.causal_conv)
        init_layer(self.penultimate_conv)

        for param in self.bgru.parameters():
            self._init_param(param)

        init_layer(self.fc_final)


    def forward(self, data_in):
        data_in = data_in.view(data_in.shape[0], 1, data_in.shape[1])

        data_out = self.causal_conv(data_in)
        skip_connections = []
        for block in self.blocks:
            data_out, skip_out = block(data_out)
            skip_connections.append(skip_out)
        skip_out = skip_connections[0]
        for skip_other in skip_connections[1:]:
            skip_out = skip_out + skip_other
        data_out = F.relu(skip_out)
        data_out = self.penultimate_conv(data_out)
        '''(batch_size, feature_maps, time_steps)'''

        data_out = data_out.transpose(1, 2)
        '''(batch_size, time_steps, feature_maps)'''

        (data_out, h) = self.bgru(data_out)

        data_out = self.fc_final(data_out)
        '''(batch_size, time_steps, 1)'''

        data_out = data_out.view(data_out.shape[0 : 2])
        '''(batch_size, time_steps)'''

        seq_len = self.seq_len
        width = data_out.shape[1] - seq_len + 1
        output = data_out[:, seq_len // 2 : seq_len // 2 + width]
        '''(batch_size, width)'''

        return output
        # return data_out.view(data_out.shape[0], data_out.shape[2])


class WaveNetBGRU_speedup(nn.Module):
    """
    WaveNet with a GRU and faster generation fo predictions
    :param nn: [description]
    :type nn: [type]
    :return: [description]
    :rtype: [type]
    """

    def __init__(self, layers=6, kernel_size=3, residual_channels=32, dilation_channels=32, skip_channels=32):
        super(WaveNetBGRU_speedup, self).__init__()
        assert kernel_size % 2 == 1, f'kernel_size ({kernel_size}) must be odd'
        self.kernel_size = kernel_size # has to be odd integer, since even integer may break dilated conv output size
        self.seq_len = (2 ** layers - 1) * (kernel_size - 1) + 1

        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels

        self.causal_conv = nn.Conv1d(1, residual_channels, kernel_size=1, bias=False)
        self.blocks = [DilatedResidualBlock(residual_channels, dilation_channels, skip_channels, kernel_size, 2**i, True)
                       for i in range(layers)]
        for i, block in enumerate(self.blocks):
            self.add_module(f"dilatedConv{i}", block)
        self.penultimate_conv = nn.Conv1d(skip_channels, skip_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True)
        self.final_conv = nn.Conv1d(skip_channels, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True)

        self.bgru = nn.GRU(input_size=skip_channels, hidden_size=skip_channels, num_layers=2, bias=True, batch_first=True, dropout=0., bidirectional=True)

        self.fc_final = nn.Linear(2 * skip_channels, 1)

        self.init_weights()

    def _init_param(self, param):

        if param.ndimension() == 1:
            param.data.fill_(0.)

        elif param.ndimension() == 2:
            n = param.size(-1)
            std = math.sqrt(2. / n)
            scale = std * math.sqrt(3.)
            param.data.uniform_(-scale, scale)

    def init_weights(self):
        init_layer(self.causal_conv)
        init_layer(self.penultimate_conv)

        for param in self.bgru.parameters():
            self._init_param(param)

        init_layer(self.fc_final)


    def forward(self, data_in):
        data_in = data_in.view(data_in.shape[0], 1, data_in.shape[1])

        data_out = self.causal_conv(data_in)
        skip_connections = []
        for block in self.blocks:
            data_out, skip_out = block(data_out)
            skip_connections.append(skip_out)
        skip_out = skip_connections[0]
        for skip_other in skip_connections[1:]:
            skip_out = skip_out + skip_other
        data_out = F.relu(skip_out)
        data_out = self.penultimate_conv(data_out)
        '''(batch_size, feature_maps, time_steps)'''

        data_out = data_out.transpose(1, 2)
        '''(batch_size, time_steps, feature_maps)'''

        seq_len = self.seq_len
        width = data_out.shape[1] - seq_len + 1
        data_out = data_out[:, seq_len // 2 : seq_len // 2 + width]
        '''(batch_size, width)'''

        (data_out, h) = self.bgru(data_out)

        data_out = self.fc_final(data_out)
        '''(batch_size, time_steps, 1)'''

        output = data_out.view(data_out.shape[0 : 2])
        '''(batch_size, time_steps)'''

        return output
        # return data_out.view(data_out.shape[0], data_out.shape[2])