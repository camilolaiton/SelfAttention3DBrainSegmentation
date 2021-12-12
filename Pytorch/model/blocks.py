from numpy.core.numeric import normalize_axis_tuple
from torch import nn
import torch

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class ConvolutionalBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding):

        conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
        )
        
        activation = nn.LeakyReLU()
        bn = nn.BatchNorm3d(out_channels)

        super(ConvolutionalBlock, self).__init__(
            conv, 
            activation, 
            bn
        )

# In tensorflow this would be the PatchEncoder Class
class PositionalEmbedding(nn.Module):
    def __init__(self, flatten_dim=512, feature_maps_size=64):
        super(PositionalEmbedding, self).__init__()

        self.projection = nn.Linear(in_features=feature_maps_size, out_features=feature_maps_size)
        self.positional_embedding = nn.Parameter(torch.zeros(1, flatten_dim, feature_maps_size))
        # self.positional_embedding = nn.Embedding(8, 64)
        # self.positions = torch.arange(start=0, end=num_patches)

    def forward(self, x):
        proj = self.projection(x)
        emb = self.positional_embedding
        # print("proj: ", proj.shape, " emb: ", emb.shape)
        return proj + emb

class TransformerBlock(nn.Module):
    def __init__(self, num_heads, embedding_dim, dropout_rate, norm_rate):
        super(TransformerBlock, self).__init__()

        self.layer_norm_1 = nn.LayerNorm(embedding_dim, eps=norm_rate)
        self.attention_layer = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate)
        self.mlp_block = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*3),
            nn.Dropout(dropout_rate),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim*3, embedding_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(inplace=True),
        )
        self.layer_norm_2 = nn.LayerNorm(embedding_dim, eps=norm_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        
        x = self.layer_norm_1(inputs)
        x, weights = self.attention_layer(x, x, x)
        x2 = x + inputs
        x3 = self.layer_norm_2(x2)
        x3 = self.mlp_block(x3)
        x3 = x3 + x2
        return x3

class DecoderBlockCup(nn.Module):
    def __init__(self, in_channels, out_channels, norm_rate, kernel_size=3):
        super(DecoderBlockCup, self).__init__()

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
        )

        self.batch_norm = nn.BatchNorm3d(out_channels, eps=norm_rate)
        self.activation = nn.LeakyReLU()
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, upsample=True):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        if upsample:
            x = self.upsample(x)

        return x

class DecoderUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_rate, kernel_size=3, strides=1):
        super(DecoderUpsampleBlock, self).__init__()

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding='same'
        )

        self.batch_norm = nn.BatchNorm3d(out_channels, eps=norm_rate)
        self.activation = nn.LeakyReLU()
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.upsample(x)
        return x

class ConnectionComponents(nn.Module):
    def __init__(self, in_channels, out_channels, norm_rate, kernel_size=3, strides=1):
        super(ConnectionComponents, self).__init__()

        self.conv_1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding='same'
        )

        self.conv_2 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding='same'
        )

        self.activation_1 = nn.LeakyReLU()
        self.activation_2 = nn.LeakyReLU()

        self.bach_norm_1 = nn.BatchNorm3d(1, eps=norm_rate)
        self.bach_norm_2 = nn.BatchNorm3d(out_channels, eps=norm_rate)
        self.bach_norm_3 = nn.BatchNorm3d(out_channels, eps=norm_rate)

    def forward(self, x):
        shortcut = x
        path_1 = self.conv_1(shortcut)
        path_1 = self.bach_norm_1(path_1)
        
        # conv 3x3
        path_2 = self.conv_2(x)
        path_2 = self.bach_norm_2(path_2)
        path_2 = self.activation_2(path_2)

        # add layer
        out = path_1 + path_2
        out = self.activation_1(out)
        out = self.bach_norm_3(out)
        return out

class EncoderDecoderConnections(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_rate=1e-4):
        super(EncoderDecoderConnections, self).__init__()

        self.con_comp_1 = ConnectionComponents(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_rate=norm_rate,
            kernel_size=kernel_size,
        )

        self.con_comp_2 = ConnectionComponents(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_rate=norm_rate,
            kernel_size=kernel_size,
        )

        self.con_comp_3 = ConnectionComponents(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_rate=norm_rate,
            kernel_size=kernel_size,
        )

        self.con_comp_4 = ConnectionComponents(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_rate=norm_rate,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        x = self.con_comp_1(x)
        x = self.con_comp_2(x)
        x = self.con_comp_3(x)
        x = self.con_comp_4(x)
        return x

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, strides=1):
        conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding='same'
        )

        activation = nn.Softmax(dim=1)
        super(SegmentationHead, self).__init__(
            conv, 
            activation
        )