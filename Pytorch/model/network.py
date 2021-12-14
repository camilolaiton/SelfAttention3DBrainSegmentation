import torch
from torch.autograd import Variable
from torch import nn
from blocks import *
from torchviz import make_dot

class EncoderPath(nn.Module):
    def __init__(self):
        super(EncoderPath, self).__init__()
        
        ### First block
        self.conv_1 = ConvolutionalBlock(
            in_channels=1,
            out_channels=16,
            strides=1,
            kernel_size=3,
            padding='same',
        )

        self.conv_2 = ConvolutionalBlock(
            in_channels=16,
            out_channels=16,
            strides=1,
            kernel_size=3,
            padding='same',
        )

        self.conv_3 = ConvolutionalBlock(
            in_channels=16,
            out_channels=16,
            strides=2,
            kernel_size=3,
            padding=1,
        )

        ### Second block
        self.conv_4 = ConvolutionalBlock(
            in_channels=16,
            out_channels=32,
            strides=1,
            kernel_size=3,
            padding='same',
        )

        self.conv_5 = ConvolutionalBlock(
            in_channels=32,
            out_channels=32,
            strides=1,
            kernel_size=3,
            padding='same',
        )

        self.conv_6 = ConvolutionalBlock(
            in_channels=32,
            out_channels=32,
            strides=2,
            kernel_size=3,
            padding=1,
        )

        ### Third block
        self.conv_7 = ConvolutionalBlock(
            in_channels=32,
            out_channels=64,
            strides=1,
            kernel_size=3,
            padding='same',
        )

        self.conv_8 = ConvolutionalBlock(
            in_channels=64,
            out_channels=64,
            strides=1,
            kernel_size=3,
            padding='same',
        )

        self.conv_9 = ConvolutionalBlock(
            in_channels=64,
            out_channels=64,
            strides=2,
            kernel_size=3,
            padding=1,
        )

        ### Positional embedding
        self.pos_embedding = PositionalEncoding(64)#PositionalEmbedding()

        ### Transformer block
        self.attention_1 = TransformerBlock(4, 64, 0.1, 1e-4)
        self.attention_2 = TransformerBlock(4, 64, 0.1, 1e-4)
        self.attention_3 = TransformerBlock(4, 64, 0.1, 1e-4)
        self.attention_4 = TransformerBlock(4, 64, 0.1, 1e-4)
        
        ### Layer norm
        self.layer_norm = nn.LayerNorm(64, 1e-4)

        self.skip_conns = []

    def _init_weights(self):
        nn.init.xavier_uniform(self.conv_1)

    def get_skip_conns(self):
        return self.skip_conns

    def forward(self, x):
        
        # Block 1
            # Input: N, 1, 64, 64, 64 | output: N, 16, 64, 64, 64 
        x = self.conv_1(x)
            # Input: N, 16, 64, 64, 64 | output: N, 16, 64, 64, 64
        skip_1 = self.conv_2(x)
            # Input: N, 16, 64, 64, 64 | output: N, 16, 32, 32, 32
        x = self.conv_3(skip_1)

        # Block 2
            # Input: N, 16, 32, 32, 32 | output: N, 32, 32, 32, 32
        x = self.conv_4(x)
            # Input: N, 32, 32, 32, 32 | output: N, 32, 32, 32, 32
        skip_2 = self.conv_5(x)
            # Input: N, 32, 32, 32, 32 | output: N, 32, 16, 16, 16
        x = self.conv_6(skip_2)
        
        # Block 3
            # Input: N, 32, 16, 16, 16 | output: N, 64, 16, 16, 16
        x = self.conv_7(x)
            # Input: N, 64, 16, 16, 16 | output: N, 64, 16, 16, 16
        skip_3 = self.conv_8(x)
            # Input: N, 64, 16, 16, 16 | output: N, 64, 8, 8, 8
        x = self.conv_9(skip_3)
        
        # Feature maps projection
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(x.size(0), 512, -1)

        x = self.pos_embedding(x)

        # Attention
        x = self.attention_1(x)
        x = self.attention_2(x)
        x = self.attention_3(x)
        x = self.attention_4(x)

        x = self.layer_norm(x)

        # Volume projection
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.size(0), x.size(1), 8, 8, 8)

        self.skip_conns = [skip_1, skip_2, skip_3]
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self) -> None:
        super(DecoderBlock, self).__init__()
        
        self.conv_1 = ConvolutionalBlock(
            in_channels=64,
            out_channels=64,
            strides=1,
            kernel_size=3,
            padding='same',
        )

        self.up_conv_1 = DecoderUpsampleBlock(
            in_channels=64, 
            out_channels=64, 
            norm_rate=1e-4,
            kernel_size=3,
            strides=1,
        )

        self.skip_con_1 = EncoderDecoderConnections(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            norm_rate=1e-4
        )

        self.conv_2 = ConvolutionalBlock(
            in_channels=128,
            out_channels=64,
            strides=1,
            kernel_size=3,
            padding='same',
        )

        self.conv_3 = ConvolutionalBlock(
            in_channels=64,
            out_channels=32,
            strides=1,
            kernel_size=3,
            padding='same',
        )

        self.up_conv_2 = DecoderUpsampleBlock(
            in_channels=32, 
            out_channels=32, 
            norm_rate=1e-4,
            kernel_size=3,
            strides=1,
        )

        self.skip_con_2 = EncoderDecoderConnections(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            norm_rate=1e-4
        )

        self.conv_4 = ConvolutionalBlock(
            in_channels=64,
            out_channels=32,
            strides=1,
            kernel_size=3,
            padding='same',
        )

        self.conv_5 = ConvolutionalBlock(
            in_channels=32,
            out_channels=16,
            strides=1,
            kernel_size=3,
            padding='same',
        )

        self.up_conv_3 = DecoderUpsampleBlock(
            in_channels=16, 
            out_channels=16, 
            norm_rate=1e-4,
            kernel_size=3,
            strides=1,
        )

        self.skip_con_3 = EncoderDecoderConnections(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            norm_rate=1e-4
        )

        self.segmentation_head = SegmentationHead(
            in_channels=32,
            out_channels=38, # N classes
            kernel_size=1,
            strides=1,
        )

    def forward(self, x, skips):
        # Block 1
            # Input: N, 64, 8, 8, 8 | output: N, 64, 8, 8, 8
        x = self.conv_1(x)
            # Input: N, 64, 8, 8, 8 | output: N, 64, 16, 16, 16
        x = self.up_conv_1(x)

            # Input: N, 64, 16, 16, 16 | output: N, 64, 16, 16, 16
        skip_lyr_1 = self.skip_con_1(skips[2])
        x = torch.cat([x, skip_lyr_1], dim=1)

        # Block 2
            # Input: N, 128, 16, 16, 16 | output: N, 64, 16, 16, 16
        x = self.conv_2(x)
            # Input: N, 64, 16, 16, 16 | output: N, 32, 16, 16, 16
        x = self.conv_3(x)
            # Input: N, 32, 16, 16, 16 | output: N, 32, 32, 32, 32
        x = self.up_conv_2(x)

            # Input: N, 32, 32, 32, 32 | output: N, 32, 32, 32, 32
        skip_lyr_2 = self.skip_con_2(skips[1])
        x = torch.cat([x, skip_lyr_2], dim=1)

        # Block 3
            # Input: N, 64, 16, 16, 16 | output: N, 32, 32, 32, 32
        x = self.conv_4(x)
            # Input: N, 32, 32, 32, 32 | output: N, 16, 32, 32, 32
        x = self.conv_5(x)
            # Input: N, 16, 16, 16, 16 | output: N, 16, 64, 64, 64
        x = self.up_conv_3(x)

        # Input: N, 16, 64, 64, 64 | output: N, 16, 64, 64, 64
        skip_lyr_3 = self.skip_con_3(skips[0])
        x = torch.cat([x, skip_lyr_3], dim=1)

        # End block
        x = self.segmentation_head(x)
        return x

class BrainSegmentationNetwork(nn.Module):
    def __init__(self):
        super(BrainSegmentationNetwork, self).__init__()
        self.encoder_path = EncoderPath()
        self.decoder_block_cup = DecoderBlockCup(64, 64, 1e-4)
        self.decoder_path = DecoderBlock()
        
    def forward(self, x):
        x = self.encoder_path(x)
        x = self.decoder_block_cup(x, upsample=False)
        skip_conns = self.encoder_path.get_skip_conns()
        x = self.decoder_path(x, skip_conns)
        return x

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())

def main():
    print(torch.__version__)
    x = Variable(torch.randn(1, 1, 64, 64, 64))
    # print(x.shape)
    model = BrainSegmentationNetwork()
    
    # for param in model.parameters():
    #     print(type(param.data), param.size())
    
    out = model(x)
    trainable_params, total_params = count_params(model)
    print(model)
    print("Trainable params: ", trainable_params, " total params: ", total_params)
    # print(model)
    # make_dot(out).render("brain_seg_net", format="png")

if __name__ == "__main__":
    main()
