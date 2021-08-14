import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv, GATConv, RelGraphConv

class GCNLabelEncoder(nn.Module):
    def __init__(self, graph: DGLGraph, emb_dim=512, use_bias=True, num_hops=2, dropout=0.2):
        super().__init__()
        self.graph = graph
        self.emb_dim = emb_dim
        self.num_nodes = graph.num_nodes()
        self.use_bias = use_bias
        self.num_hops = num_hops
        self.dropout = dropout
        self.init_emb = nn.Parameter(torch.Tensor(self.num_nodes, 300))
        nn.init.kaiming_normal_(self.init_emb, a=math.sqrt(5))
        if num_hops == 1:
            self.gcn1 = GraphConv(300, emb_dim, weight=True, bias=use_bias)
        elif num_hops == 2:
            self.gcn1 = GraphConv(300, 400, weight=True, bias=use_bias)
            self.act1 = nn.LeakyReLU(0.2)
            self.drop1 = nn.Dropout(dropout)
            self.gcn2 = GraphConv(400, emb_dim, weight=True, bias=use_bias)
        elif num_hops == 3:
            self.gcn1 = GraphConv(300, 400, weight=True, bias=use_bias)
            self.act1 = nn.LeakyReLU(0.2)
            self.drop1 = nn.Dropout(dropout)
            self.gcn2 = GraphConv(400, emb_dim, weight=True, bias=use_bias)
            self.act2 = nn.LeakyReLU(0.2)
            self.drop2 = nn.Dropout(dropout)
            self.gcn3 = GraphConv(emb_dim, emb_dim, weight=True, bias=use_bias)
    def forward(self):
        if self.num_hops == 1:
            x = self.gcn1(self.graph, self.init_emb)
        elif self.num_hops == 2:
            x = self.gcn1(self.graph, self.init_emb)
            x = self.act1(x)
            x = self.drop1(x)
            x = self.gcn2(self.graph, x)
        elif self.num_hops == 3:
            x = self.gcn1(self.graph, self.init_emb)
            x = self.act1(x)
            x = self.drop1(x)
            x = self.gcn2(self.graph, x)
            x = self.act2(x)
            x = self.drop2(x)
            x = self.gcn3(self.graph, x)
        return x                             # [num_nodes, emb_dim]

class GATLabelEncoder(nn.Module):
    def __init__(self, graph: DGLGraph, emb_dim=512, num_hops=2, num_heads=4):
        super().__init__()
        self.graph = graph
        self.emb_dim = emb_dim
        self.num_nodes = graph.num_nodes()
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.init_emb = nn.Parameter(torch.Tensor(self.num_nodes, 300))
        nn.init.kaiming_normal_(self.init_emb, a=math.sqrt(5))
        self.mid_dim = 400
        assert emb_dim % num_heads == 0 and self.mid_dim % num_heads == 0
        if num_hops == 1:
            self.gat1 = GATConv(300, emb_dim // num_heads, num_heads)
        elif num_hops == 2:
            self.gat1 = GATConv(300, self.mid_dim // num_heads, num_heads)
            self.act1 = nn.LeakyReLU(0.2)
            self.gat2 = GATConv(self.mid_dim, emb_dim // num_heads, num_heads)
        elif num_hops == 3:
            self.gat1 = GATConv(300, self.mid_dim // num_heads)
            self.act1 = nn.LeakyReLU(0.2)
            self.gat2 = GATConv(self.mid_dim, emb_dim // num_heads, num_heads)
            self.act2 = nn.LeakyReLU(0.2)
            self.gat3 = GATConv(emb_dim, emb_dim // num_heads, num_heads)
    def forward(self):
        if self.num_hops == 1:
            x = self.gat1(self.graph, self.init_emb).view(-1, self.emb_dim)
        elif self.num_hops == 2:
            x = self.gat1(self.graph, self.init_emb).view(-1, self.mid_dim)
            x = self.act1(x)
            x = self.gat2(self.graph, x).view(-1, self.emb_dim)
        elif self.num_hops == 3:
            x = self.gat1(self.graph, self.init_emb).view(-1, self.mid_dim)
            x = self.act1(x)
            x = self.gat2(self.graph, x).view(-1, self.emb_dim)
            x = self.act2(x)
            x = self.gat3(self.graph, x).view(-1, self.emb_dim)
        return x                             # [num_nodes, emb_dim]


class RGCNLabelEncoder(nn.Module):
    def __init__(self, graph: DGLGraph, emb_dim=512, use_bias=True, num_hops=2, dropout=0.):
        super().__init__()
        self.graph = graph
        assert 'etype' in graph.edata
        self.num_rels = len(torch.unique(graph.edata['etype']))
        self.emb_dim = emb_dim
        self.num_nodes = graph.num_nodes()
        self.use_bias = use_bias
        self.num_hops = num_hops
        self.dropout = dropout
        self.init_emb = nn.Parameter(torch.Tensor(self.num_nodes, 300))
        nn.init.kaiming_normal_(self.init_emb, a=math.sqrt(5))
        if num_hops == 1:
            self.gcn1 = RelGraphConv(300, emb_dim, self.num_rels, regularizer="bdd", bias=use_bias)
        elif num_hops == 2:
            self.gcn1 = RelGraphConv(300, 400, self.num_rels, regularizer="bdd", bias=use_bias)
            self.act1 = nn.LeakyReLU(0.2)
            self.drop1 = nn.Dropout(dropout)
            self.gcn2 = RelGraphConv(400, emb_dim, self.num_rels, regularizer="bdd", bias=use_bias)
    def forward(self):
        if self.num_hops == 1:
            x = self.gcn1(self.graph, self.init_emb, self.graph.edata['etype'])
        elif self.num_hops == 2:
            x = self.gcn1(self.graph, self.init_emb, self.graph.edata['etype'])
            x = self.act1(x)
            x = self.drop1(x)
            x = self.gcn2(self.graph, x, self.graph.edata['etype'])
        return x                             # [num_nodes, emb_dim]

class GCNLabelEncoderInter(nn.Module):
    def __init__(self, graph1: DGLGraph, graph2: DGLGraph, emb_dim=512, use_bias=True, num_hops=2, dropout=0.):
        super().__init__()
        self.graph1 = graph1
        self.graph2 = graph2
        # if "_edge_weight" in graph.edata:
        #     self.edge_weight = nn.Parameter(torch.tensor(graph.edata["_edge_weight"], dtype=torch.float32), requires_grad=False)
        # else:
        #     self.edge_weight = None
        self.emb_dim = emb_dim
        self.num_nodes = graph1.num_nodes()
        self.use_bias = use_bias
        self.num_hops = num_hops
        self.dropout = dropout
        self.init_emb = nn.Parameter(torch.Tensor(self.num_nodes, 300))
        nn.init.kaiming_normal_(self.init_emb, a=math.sqrt(5))
        if num_hops == 1:
            self.gcn_a1 = GraphConv(300, emb_dim, weight=True, bias=use_bias)
            self.gcn_b1 = GraphConv(300, emb_dim, weight=True, bias=use_bias)
        elif num_hops == 2:
            self.gcn_a1 = GraphConv(300, 400, weight=True, bias=use_bias)
            self.act1 = nn.LeakyReLU(0.2)
            self.drop1 = nn.Dropout(dropout)
            self.gcn_a2 = GraphConv(400, emb_dim, weight=True, bias=use_bias)
            self.gcn_b1 = GraphConv(300, 400, weight=True, bias=use_bias)
            self.act2 = nn.LeakyReLU(0.2)
            self.drop2 = nn.Dropout(dropout)
            self.gcn_b2 = GraphConv(400, emb_dim, weight=True, bias=use_bias)
    def forward(self):
        if self.num_hops == 1:
            x = self.gcn_a1(self.graph1, self.init_emb)
            x += self.gcn_b1(self.graph2, self.init_emb)
        elif self.num_hops == 2:
            x1 = self.gcn_a1(self.graph1, self.init_emb)
            x1 = self.act1(x1)
            x1 = self.drop1(x1)
            x2 = self.gcn_b1(self.graph2, self.init_emb)
            x2 = self.act2(x2)
            x2 = self.drop2(x2)
            x = x1 + x2
            x1 = self.gcn_a2(self.graph1, x)
            x2 = self.gcn_b2(self.graph2, x)
            x = x1 + x2
        return x                             # [num_nodes, emb_dim]


def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)
    
    
class Cnn_5layers_AvgPooling(nn.Module):
    
    def __init__(self, classes_num):
        super(Cnn_5layers_AvgPooling, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.fc)
        
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn4(self.conv4(x)))
        x = F.avg_pool2d(x, kernel_size=(1, 1))
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        
        output = torch.sigmoid(self.fc(x))
        
        return output
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x
    
    
class Cnn_9layers_MaxPooling(nn.Module):
    def __init__(self, classes_num):
        
        super(Cnn_9layers_MaxPooling, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='max')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        
        output = torch.sigmoid(self.fc(x))
        
        return output
        
        
class Cnn_9layers_AvgPooling(nn.Module):
    def __init__(self, classes_num):
        
        super(Cnn_9layers_AvgPooling, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        
        output = torch.sigmoid(self.fc(x))
        
        return output

class Cnn_9layers_AvgPooling_Emb(nn.Module):
    def __init__(self, classes_num):
        
        super(Cnn_9layers_AvgPooling_Emb, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        # self.fc = nn.Linear(512, classes_num, bias=True)
        self.label_emb = nn.Parameter(torch.Tensor(512, classes_num))
        

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.label_emb, a=math.sqrt(5))

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        
        output = torch.sigmoid(x.mm(self.label_emb))
        
        return output

class Cnn_9layers_AvgPooling_GCNEmb(nn.Module):
    def __init__(self, classes_num, graph, class_indices):
        
        super(Cnn_9layers_AvgPooling_GCNEmb, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.graph_encoder = GCNLabelEncoder(graph)
        self.class_indices = class_indices
        # workaround to easily get a model's device
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        label_emb = self.graph_encoder()   # (num_nodes, feature_maps)
        label_emb = label_emb[self.class_indices]   # (num_classes, feature_maps)
        output = torch.sigmoid(x.mm(label_emb.T)) # (batch_size, num_classes)
        
        return output

class Cnn_9layers_AvgPooling_DoubleGCNInterEmb(nn.Module):
    def __init__(self, classes_num, graph1, graph2, class_indices):
        
        super(Cnn_9layers_AvgPooling_DoubleGCNInterEmb, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.graph_encoder = GCNLabelEncoderInter(graph1, graph2)
        self.class_indices = class_indices
        self.fc = nn.Linear(512, 512)
        # workaround to easily get a model's device
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc(x))
        label_emb = self.graph_encoder()   # (num_nodes, feature_maps)
        label_emb = label_emb[self.class_indices]   # (num_classes, feature_maps)
        output = torch.sigmoid(x.mm(label_emb.T)) # (batch_size, num_classes)
        
        return output


class Cnn_9layers_AvgPooling_GATEmb(nn.Module):
    def __init__(self, classes_num, graph, class_indices):
        
        super(Cnn_9layers_AvgPooling_GATEmb, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.graph_encoder = GATLabelEncoder(graph)
        self.class_indices = class_indices
        # workaround to easily get a model's device
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        label_emb = self.graph_encoder()   # (num_nodes, feature_maps)
        label_emb = label_emb[self.class_indices]   # (num_classes, feature_maps)
        output = torch.sigmoid(x.mm(label_emb.T)) # (batch_size, num_classes)
        
        return output

class Cnn_9layers_AvgPooling_RGCNEmb(nn.Module):
    def __init__(self, classes_num, graph, class_indices):
        
        super(Cnn_9layers_AvgPooling_RGCNEmb, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.graph_encoder = RGCNLabelEncoder(graph)
        self.class_indices = class_indices
        # workaround to easily get a model's device
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        label_emb = self.graph_encoder()   # (num_nodes, feature_maps)
        label_emb = label_emb[self.class_indices]   # (num_classes, feature_maps)
        output = torch.sigmoid(x.mm(label_emb.T)) # (batch_size, num_classes)
        
        return output

class Cnn_13layers_AvgPooling(nn.Module):
    def __init__(self, classes_num):
        
        super(Cnn_13layers_AvgPooling, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)
        (x, _) = torch.max(x, dim=2)
        
        output = torch.sigmoid(self.fc(x))
        
        return output