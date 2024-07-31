import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, knn_graph
import math
import torch
import torch.nn as nn
# import torch.nn.functional as F
from scipy import sparse
import torch.nn.parallel
import torch.utils.data
from spatial_utils import *
import time

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, device):
        super().__init__()
        
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
        self.attn = MultiHeadAttention(d_model, nhead, dropout, device)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        
    def forward(self, x, mask, input_keep):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask, input_keep))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        self.eps = eps
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

    
def ori_attention(q, k, v, d_k, mask, dropout, rm):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # scores = scores + rm
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, device):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // nhead
        self.h = nhead

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # self.rbf = Multi_ch_RBF(2, device)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask, input_keep):
        bs = q.size(0)
        
        # rm = self.rbf(input_keep, input_keep)
        
        # perform linear operation and split into N heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        # scores = ori_attention(q, k, v, self.d_k, mask, self.dropout, rm)
        scores = ori_attention(q, k, v, self.d_k, mask, self.dropout, None)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output
    
    
class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_feedforward, d_model)
    
    def forward(self, x):
        x = self.dropout(NewGELU()(self.linear_1(x)))
        x = self.linear_2(x)
        return x

import copy


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, device):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.layers = get_clones(EncoderLayer(d_model, nhead, dim_feedforward, dropout, device), num_encoder_layers)
        self.norm = Norm(d_model)
        
    def forward(self, src, mask, input_keep):
        for i in range(self.num_encoder_layers):
            src = self.layers[i](src, mask, input_keep)
        return self.norm(src)

    
class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, device):
        super().__init__()
        self.encoder = Encoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout, device)
        
    def forward(self, src, src_mask, input_keep):
        e_outputs = self.encoder(src, src_mask, input_keep)
        return e_outputs


def padded_seq_to_vectors(padded_seq, logger):
    # Get the actual lengths of each sequence in the batch
    actual_lengths = logger.int()
    # Step 1: Form the first tensor containing all actual elements from the batch
    mask = torch.arange(padded_seq.size(1), device=padded_seq.device) < actual_lengths.view(-1, 1)
    tensor1 = torch.masked_select(padded_seq, mask.unsqueeze(-1)).view(-1, padded_seq.size(-1))

    # Step 2: Form the second tensor to record which row each element comes from
    tensor2 = torch.repeat_interleave(torch.arange(padded_seq.size(0), device=padded_seq.device), actual_lengths)
    return tensor1, tensor2


def extract_first_element_per_batch(tensor1, tensor2):
    # Get the unique batch indices from tensor2
    unique_batch_indices = torch.unique(tensor2)
    # Initialize a list to store the first elements of each batch item
    first_elements = []

    # Iterate through each unique batch index
    for batch_idx in unique_batch_indices:
        # Find the first occurrence of the batch index in tensor2
        first_occurrence = torch.nonzero(tensor2 == batch_idx, as_tuple=False)[0, 0]
        # Extract the first element from tensor1 and append it to the list
        first_element = tensor1[first_occurrence]
        first_elements.append(first_element)
    # Convert the list to a tensor
    result = torch.stack(first_elements, dim=0)
    return result


class LayerNorm(nn.Module):
    """
    layer normalization
    Simple layer norm object optionally used with the convolutional encoder.
    """

    # Batch_norm: same dimension, different features, different examples
    # layer_norm: same features, same examples, different dim


    def __init__(self, feature_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones((feature_dim,)))
        self.register_parameter("gamma", self.gamma)
        self.beta = nn.Parameter(torch.zeros((feature_dim,)))
        self.register_parameter("beta", self.beta)
        self.eps = eps

    def forward(self, x):
        # x: [batch_size, embed_dim]

        # yes, it's layer normalization

        # normalize for each embedding
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # output shape is the same as x
        # Type not match for self.gamma and self.beta??????????????????????
        # output: [batch_size, embed_dim]
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def get_activation_function(activation, context_str):
    if activation == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise Exception("{} activation not recognized.".format(context_str))


class SingleFeedForwardNN(nn.Module):

    """
        Creates a single layer fully connected feed forward neural network.
        this will use non-linearity, layer normalization, dropout
        this is for the hidden layer, not the last layer of the feed forard NN
    """

    def __init__(self, input_dim,
                 output_dim,
                 dropout_rate=None,
                 activation="sigmoid",
                 use_layernormalize=False,
                 skip_connection=False,
                 context_str=''):
        '''
        Args:
            input_dim (int32): the input embedding dim
            output_dim (int32): dimension of the output of the network.
            dropout_rate (scalar tensor or float): Dropout keep prob.
            activation (string): tanh or relu or leakyrelu or sigmoid
            use_layernormalize (bool): do layer normalization or not
            skip_connection (bool): do skip connection or not
            context_str (string): indicate which spatial relation encoder is using the current FFN
        '''
        super(SingleFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        self.act = get_activation_function(activation, context_str)

        if use_layernormalize:
            # the layer normalization is only used in the hidden layer, not the last layer
            self.layernorm = nn.LayerNorm(self.output_dim)
        else:
            self.layernorm = None

        # the skip connection is only possible, if the input and out dimention is the same
        if self.input_dim == self.output_dim:
            self.skip_connection = skip_connection
        else:
            self.skip_connection = False

        self.linear = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size,..., output_dim]
            note there is no non-linearity applied to the output.
        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        # Linear layer
        output = self.linear(input_tensor)
        # non-linearity
        output = self.act(output)
        # dropout
        if self.dropout is not None:
            output = self.dropout(output)

        # skip connection
        if self.skip_connection:
            output = output + input_tensor

        # layer normalization
        if self.layernorm is not None:
            output = self.layernorm(output)

        return output


class MultiLayerFeedForwardNN(nn.Module):
    """
        Creates a fully connected feed forward neural network.
        N fully connected feed forward NN, each hidden layer will use non-linearity, layer normalization, dropout
        The last layer do not have any of these
    """

    def __init__(self, input_dim,
                 output_dim,
                 num_hidden_layers=0,
                 dropout_rate=0.5,
                 hidden_dim=-1,
                 activation="relu",
                 use_layernormalize=True,
                 skip_connection=False,
                 context_str=None):
        '''
        Args:
            input_dim (int32): the input embedding dim
            num_hidden_layers (int32): number of hidden layers in the network, set to 0 for a linear network.
            output_dim (int32): dimension of the output of the network.
            dropout (scalar tensor or float): Dropout keep prob.
            hidden_dim (int32): size of the hidden layers
            activation (string): tanh or relu
            use_layernormalize (bool): do layer normalization or not
            context_str (string): indicate which spatial relation encoder is using the current FFN
        '''
        super(MultiLayerFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_layernormalize = use_layernormalize
        self.skip_connection = skip_connection
        self.context_str = context_str

        self.layers = nn.ModuleList()
        if self.num_hidden_layers <= 0:
            self.layers.append(SingleFeedForwardNN(input_dim=self.input_dim,
                                                   output_dim=self.output_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=False,
                                                   skip_connection=False,
                                                   context_str=self.context_str))
        else:
            self.layers.append(SingleFeedForwardNN(input_dim=self.input_dim,
                                                   output_dim=self.hidden_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=self.use_layernormalize,
                                                   skip_connection=self.skip_connection,
                                                   context_str=self.context_str))

            for i in range(self.num_hidden_layers - 1):
                self.layers.append(SingleFeedForwardNN(input_dim=self.hidden_dim,
                                                       output_dim=self.hidden_dim,
                                                       dropout_rate=self.dropout_rate,
                                                       activation=self.activation,
                                                       use_layernormalize=self.use_layernormalize,
                                                       skip_connection=self.skip_connection,
                                                       context_str=self.context_str))

            self.layers.append(SingleFeedForwardNN(input_dim=self.hidden_dim,
                                                   output_dim=self.output_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=False,
                                                   skip_connection=False,
                                                   context_str=self.context_str))

    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size, ..., output_dim]
            note there is no non-linearity applied to the output.
        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        
        output = input_tensor
        for layer in self.layers:
            # applied in each layer
            output = layer(output)

        return output

def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    freq_list = None
    if freq_init == "random":
        freq_list = torch.rand(frequency_num) * max_radius
    elif freq_init == "geometric":
        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) / (frequency_num * 1.0 - 1))
        timescales = min_radius * torch.exp(torch.arange(frequency_num, dtype=torch.float32) * log_timescale_increment)
        freq_list = 1.0 / timescales
    return freq_list

class GridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """

    def __init__(self, spa_embed_dim, coord_dim=2, frequency_num=16,
                 max_radius=0.01, min_radius=0.00001,
                 freq_init="geometric",
                 ffn=None):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(GridCellSpatialRelationEncoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.ffn = ffn
        self.cal_freq_list()
        self.cal_freq_mat()
        self.input_embed_dim = self.cal_input_dim()

        if self.ffn is not None:
            self.ffn = MultiLayerFeedForwardNN(2 * frequency_num * 2, spa_embed_dim)

    def cal_input_dim(self):
        return int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)

    def cal_freq_mat(self):
        freq_mat = torch.unsqueeze(self.freq_list, 1)
        self.freq_mat = freq_mat.repeat(1, 2)


    def make_input_embeds(self, coords):
        # coords: shape (batch_size, num_context_pt, 2)
        batch_size, num_context_pt, _ = coords.shape
        # coords: shape (batch_size, num_context_pt, 2, 1, 1)
        coords = coords.unsqueeze(-1).unsqueeze(-1)
        # coords: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords = coords.repeat(1, 1, 1, self.frequency_num, 2)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords * self.freq_mat.to(self.device)
        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=input_embed_dim)
        spr_embeds[:, :, :, :, 0::2] = torch.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, :, 1::2] = torch.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1
        # (batch_size, num_context_pt, 2*frequency_num*2)
        spr_embeds = torch.reshape(spr_embeds, (batch_size, num_context_pt, -1))
        return spr_embeds

    def forward(self, coords):

        # embedding function
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)

        # Feed Forward Network
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds

# class SpatialEncoder(nn.Module):
#     """
#     Given a list of (deltaX,deltaY), encode them using the position encoding
#     """

#     def __init__(self, spa_embed_dim, coord_dim=2, settings=None, ffn=None):
#         """
#         Args:
#             spa_embed_dim: the output spatial relation embedding dimension
#             coord_dim: the dimension of space
#         """
#         super(SpatialEncoder, self).__init__()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.spa_embed_dim = spa_embed_dim
#         self.coord_dim = coord_dim
#         self.ffn = ffn
#         # input_dim:2
#         self.input_embed_dim = self.coord_dim
#         self.nn_length = settings['nn_length']
#         self.nn_hidden_dim = settings['nn_hidden_dim']
#         if self.ffn is not None:
#             # by creating the ffn, the weights are initialized use kaiming_init
#             self.ffn = MultiLayerFeedForwardNN(self.input_embed_dim, spa_embed_dim,
#                                                num_hidden_layers=settings['nn_length'],
#                                                hidden_dim=settings['nn_hidden_dim'],
#                                                dropout_rate=settings['dropout_rate'])

#     def forward(self, coords):
#         """
#         Given a list of coords (deltaX, deltaY), give their spatial relation embedding
#         Args:
#             coords: a python list with shape (batch_size, num_context_pt, coord_dim)
#         Return:
#             sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
#         """
#         spr_embeds = coords
#         # Feed Forward Network
#         if self.ffn is not None:
#             return self.ffn(spr_embeds)
#         else:
#             return spr_embeds

def length_to_mask(lengths, total_len, device):
    max_len = total_len
    mask = torch.arange(max_len).expand(lengths.shape[0], max_len).to(device) < lengths.unsqueeze(1)
    return mask.unsqueeze(-2)


class PEGCN(nn.Module):

    # Hole right part of PEGCN consider the figure 1 in the paper: GCNCONV layers?

    """
        GCN with positional encoder and enviromental encoder
    """

    # default parameters
    def __init__(self, num_features_in=3, num_features_out=1, emb_hidden_dim=128, emb_dim=16, k=20, conv_dim=64, settings=None):
        super(PEGCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features_in = num_features_in
        self.emb_hidden_dim = emb_hidden_dim
        self.emb_dim = emb_dim
        self.k = k


        # -----------------Postional Encoder-----------------
        self.spenc = GridCellSpatialRelationEncoder(
            spa_embed_dim=emb_hidden_dim, ffn=True, min_radius=1e-06, max_radius=360
        )

        # self.spenc = SpatialEncoder(
        #     spa_embed_dim=emb_hidden_dim, ffn=True, settings=settings
        # )

        self.spdec = nn.Sequential(
            nn.Linear(emb_hidden_dim, emb_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 2, emb_hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 4, emb_dim)
        )

        # -----------------Transformer Encoder-----------------
        self.Q = torch.nn.Parameter(torch.randn((1, 1, settings['d_model']), requires_grad=True, device=self.device))
        
        self.transformer_inc = nn.Linear(in_features=settings['env_features_in'], out_features=settings['d_model'])
        
        # self.transformer_encoder = TransformerEncoder(encoder_layers, settings['num_encoder_layers'])

        # self.transformer_dec = MultiLayerFeedForwardNN(input_dim=settings['d_model'], output_dim=settings['transformer_dec_output'],
        #                                                num_hidden_layers=1,
        #                                                hidden_dim=settings['nn_hidden_dim'],
        #                                                dropout_rate=settings['dropout_rate'], )
        
        self.transformer_encoder = Transformer(d_model=settings['d_model'], nhead=settings['nhead'],
                                               num_encoder_layers=settings['num_encoder_layers'],
                                               dim_feedforward=settings['dim_feedforward'],
                                               dropout=settings['transformer_dropout'], device=self.device)


        self.transformer_dec = MultiLayerFeedForwardNN(input_dim=settings['d_model'],
                                                       output_dim=settings['transformer_dec_output'],
                                                       num_hidden_layers=1,
                                                       hidden_dim=settings['nn_hidden_dim'],
                                                       dropout_rate=settings['dropout_rate'], )
        
         # -----------------Conv layers-----------------
        self.conv1 = GCNConv(num_features_in + emb_dim + settings['transformer_dec_output'] , conv_dim)
        self.conv2 = GCNConv(conv_dim, conv_dim)
        self.fc = nn.Linear(conv_dim, num_features_out)
        self.noise_sigma = torch.nn.Parameter(torch.tensor([0.1, ], device=self.device))
        
        # init weights
        for p in self.spdec.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)
        for p in self.conv1.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)
        for p in self.conv2.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)
        torch.nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, inputs, targets, coords, input_lenths, env_features, auxil_lenths,  head_only):

        # inputs.shape x: torch.Size([32, 43, N])
        # input_lenths.shape: torch.Size([32])
        # targets.shape: torch.Size([32, 86, 1])
        # coords.shape: torch.Size([32, 86, 2])
        # env_features.shape: torch.Size([32, 184, 11])
        
        # ________________________________________postional encoding_______________________________________________________
        pe = self.spenc(coords)
        pe = self.spdec(pe)
        # _________________________________________Env_features Transformer_________________________________________________
        
        q_token_emb = self.Q.repeat(env_features.size(0),1,1)
        # q_token.shape:torch.Size([16, 1, 32])
        env_feature_emb = self.transformer_inc(env_features)
        # env_feature_emb torch.Size([16, 192, 32])
        
        k_transformer_input = torch.cat([q_token_emb, env_feature_emb], dim=1)

        k_attention_mask = length_to_mask(auxil_lenths+1, k_transformer_input.size(1), self.device)
        
        token_k = self.transformer_encoder(k_transformer_input, k_attention_mask, None)[:, 0, :]
        
        Q_feature_emb = self.transformer_dec(token_k)

        # _______________________broadcast to PE________________________________
        # pe.shape: torch.Size([32, 87, 32])

        # Q_feature_emb.shape: torch.Size([32, 32])

        x_l, _ = padded_seq_to_vectors(inputs, input_lenths)
        if self.num_features_in == 2:
            first_element = x_l[:, 0].unsqueeze(-1)
            last_element = x_l[:, -1].unsqueeze(-1)
            x_l = torch.cat([first_element, last_element], dim=-1)

        y_l, _ = padded_seq_to_vectors(targets, input_lenths)
        c_l, _ = padded_seq_to_vectors(coords, input_lenths)

        env_emb = Q_feature_emb.unsqueeze(1).expand(-1, pe.size(1), -1)
        env_emb = torch.cat([env_emb, pe], dim=-1)

        env_l, indexer = padded_seq_to_vectors(env_emb, input_lenths)
        # ____________________knn___________________________________
        # Diff between PEGNN: use c_l or pe to make graph(since pe concat env to make graph is same for only use pe)
        edge_index = knn_graph(c_l, k=self.k, batch=indexer)
        edge_weight = makeEdgeWeight(c_l, edge_index).to(self.device)

        # concat the embedding with the input
        x = torch.cat((x_l, env_l), dim=1)
        # x -> [batch_size, num_features_in + fe_pe]

        h1 = F.relu(self.conv1(x, edge_index, edge_weight))
        h1 = F.dropout(h1, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        h2 = F.dropout(h2, training=self.training)
        output = self.fc(h2)

        # the result of output and y_l, in order to compare the BMC bzw. lost function
        if not head_only:
            return output, y_l
        else:
            output_head = extract_first_element_per_batch(output, indexer)
            target_head = extract_first_element_per_batch(y_l, indexer)
            return output_head, target_head

    # #     calculate the loss
    # def bmc_loss(self, pred, target):
    #     """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    #     Args:
    #       pred: A float tensor of size [batch, 1].
    #       target: A float tensor of size [batch, 1].
    #       noise_var: A float number or tensor.
    #     Returns:
    #       loss: A float tensor. Balanced MSE Loss.
    #     """
    #     noise_var = self.noise_sigma ** 2
    #     logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    #     loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(self.device))     # contrastive-like loss
    #     loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable
    #     return loss
