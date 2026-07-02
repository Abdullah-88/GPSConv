import torch
from torch import nn, Tensor
import torch.nn.functional as F



class VecDyT(nn.Module):
    def __init__(self, input_shape):

        super().__init__()

        self.alpha = nn.Parameter(torch.randn(input_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x


class VecDyGeluSine(nn.Module):
    def __init__(self, input_shape):

        super().__init__()

        self.alpha = nn.Parameter(torch.randn(input_shape))
        self.beta = nn.Parameter(torch.randn(input_shape))
        self.gamma = nn.Parameter(torch.randn(1))
        self.etta = nn.Parameter(torch.randn(1))
        self.gelu = nn.GELU()

    def forward(self, x):



        x = self.gamma * self.gelu(self.alpha * x) + self.etta * torch.sin(self.beta * x)

        return x

class FFUnit(nn.Module):
    def __init__(self,dim):

        super().__init__()

        self.proj =  nn.Linear(dim,dim,bias=False)
        self.modulate = VecDyGeluSine(dim)


    def forward(self, x):

        u, v = x, x

        u = self.modulate(u)
        v = self.proj(v)
        g = u * v

        return g




class GatedProjectionShortConv(nn.Module):
    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.dim = dim
        
        self.short_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=dim 
        )
      
        self.proj =  nn.Linear(dim,dim,bias=False)
        self.modulate = VecDyGeluSine(dim)

    def forward(self, x):
      
        B, L, D = x.shape
        
       
        x_conv = x.transpose(1, 2)
        x_conv = self.short_conv(x_conv)[..., :L] 
        x_conv = x_conv.transpose(1, 2) 
        
       
        gate = self.modulate(x_conv)
        value = self.proj(x_conv)
        
        
        return gate * value




class GPSConvBlock(nn.Module):
    def __init__(self, dim):

        super().__init__()

        self.norm_1 =  VecDyT(dim)
        self.norm_2 =  VecDyT(dim)
        self.gpsConv = GatedProjectionShortConv(dim)
        self.feedforward = FFUnit(dim)


    def forward(self, x):
        
        residual = x

        x = self.norm_1(x)

        x = self.gpsConv(x)

        x = x + residual

        residual = x

        x = self.norm_2(x)

        x = self.feedforward(x)

        x = x + residual

        return x


class GPSConv(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()

        self.model = nn.Sequential(
            *[GPSConvBlock(d_model) for _ in range(num_layers)]
        )

    def forward(self, x):

        return self.model(x)




