import copy
from typing import Optional, Any

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
!pip install einops
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

#input ==> 32, 1, 1, 3000,, b==> batch, e==> embedding, s==> seq length
class Window_Embedding(nn.Module): 
    def __init__(self, in_channels: int = 1, window_size: int = 50, emb_size: int = 64):
        super(Window_Embedding, self).__init__()

        self.projection_1 =  nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains, in=>B,1,3000 out=>B,64,60
            nn.Conv1d(in_channels, emb_size//4, kernel_size = window_size, stride = window_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(emb_size//4),
            # Rearrange('b e s -> b s e'),
            )
        self.projection_2 =  nn.Sequential(#################
            # using a conv layer instead of a linear one -> performance gains, in=>B,1,3000 out=>B,64,60
            nn.Conv1d(in_channels, emb_size//8, kernel_size = 5, stride = 5),
            nn.LeakyReLU(),
            nn.Conv1d(emb_size//8, emb_size//4, kernel_size = 5, stride = 5),
            nn.LeakyReLU(),
            nn.Conv1d(emb_size//4, (emb_size-emb_size//4)//2, kernel_size = 2, stride = 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d((emb_size-emb_size//4)//2),
            # Rearrange('b e s -> b s e'),
            )
        
        self.projection_3 =  nn.Sequential(#################
            # using a conv layer instead of a linear one -> performance gains, in=>B,1,3000 out=>B,64,60
            nn.Conv1d(in_channels, emb_size//4, kernel_size = 25, stride = 25),
            nn.LeakyReLU(),
            nn.Conv1d(emb_size//4, (emb_size-emb_size//4)//2, kernel_size =2, stride = 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d((emb_size-emb_size//4)//2),
            # Rearrange('b e s -> b s e'),
            )
        
        
        self.projection_4 = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains, in=>B,1,3000 out=>B,64,60
            nn.Conv1d(emb_size, emb_size, kernel_size = 1, stride = 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(emb_size),
            Rearrange('b e s -> b s e'),)
            
        #in=>B,64,60 out=>B,64,61
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.arrange1 = Rearrange('b s e -> s b e')
        #in=>61,B,64 out=>61,B,64
        self.pos = PositionalEncoding(d_model=emb_size)
        #in=>61,B,64 out=>B,61,64
        self.arrange2 = Rearrange('s b e -> b s e ')

    def forward(self, x: Tensor) -> Tensor:
        x = x.squeeze().unsqueeze(dim = 1)
        # print(x.shape)
        b,_, _ = x.shape
        x_1 = self.projection_1(x)  ########################
        x_2 = self.projection_2(x) ###########
        x_3 = self.projection_3(x) 
        # print(x_local.shape,x_global.shape)
        x = torch.cat([x_1,x_2,x_3],dim = 1)##### 2)
        x = self.projection_4(x) 
        # print(x.shape)
        cls_tokens = repeat(self.cls_token, '() s e -> b s e', b=b)
        # print(cls_tokens.shape)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # print(x.shape)
        # add position embedding
        x = self.arrange1(x)
        # print(x.shape)
        x = self.pos(x)
        # print(x.shape)
        x = self.arrange2(x)
        # print(x.shape)
        return x

#input ==>(b,s,e)=>(32, 61, 64,) 
# b==> batch, s==> seq length, e==> embedding, 
class Intra_modal_atten(nn.Module): 
    def __init__(self, d_model=64, nhead=8, dropout=0.1,
                 layer_norm_eps=1e-5, window_size = 25, First = True,
                 device=None, dtype=None) -> None:
        super(Intra_modal_atten, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
       
        if First == True:
            self.window_embed = Window_Embedding(in_channels = 1, window_size = window_size, emb_size = d_model)
        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True,
                                            **factory_kwargs)
        self.dropout = Dropout(dropout) 
        self.First = First

    def forward(self, x: Tensor) -> Tensor:
        if self.First == True:
            src = self.window_embed(x)
        else:
            src = x
        # print(src.shape)
        # src = self.norm(src)  #####
        # print(src.shape)
        src2 = self.self_attn(src, src, src)[0]
        # print(src2.shape)
        out = src + self.dropout(src2)
        out = self.norm(out)   ########
        return out

##Cross Modal Attention
#input ==>(b,s,e)=>(32, 2, 64,) ==> Class tokens of EEG and EOG after intra modal attention
# b==> batch, s==> seq length, e==> embedding, 
class Cross_modal_atten(nn.Module): 
    def __init__(self, d_model=64, nhead=8, dropout=0.1,
                 layer_norm_eps=1e-5, First = False,
                 device=None, dtype=None) -> None:

        super(Cross_modal_atten, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        if First == True:
            self.cls_token = nn.Parameter(torch.randn(1,1, d_model)) ######
        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True,
                                            **factory_kwargs)
        self.dropout = Dropout(dropout) 
        self.First = First

    def forward(self, x1: Tensor,x2: Tensor) -> Tensor:
        # print(x1.shape,x2.shape)
        if len(x1.shape) == 2:
            x = torch.cat([x1.unsqueeze(dim=1), x2.unsqueeze(dim=1)], dim=1)
        else:
            x = torch.cat([x1, x2.unsqueeze(dim=1)], dim=1)
        # print(x.shape)
        b,_, _ = x.shape
        if self.First == True:
            cls_tokens = repeat(self.cls_token, '() s e -> b s e', b=b)  ######
            # print(cls_tokens.shape)
            # prepend the cls token to the input
            src = torch.cat([cls_tokens, x], dim=1)  #####
        else:
            src = x
        # print(src.shape)
        # src = self.norm(src)#####(src)
        # print(src.shape)
        src2 = self.cross_attn(src, src, src)[0]
        # print(src2.shape)
        out = src + self.dropout(src2)
        out = self.norm(out)
        return out

##Feed Forward Networks
#input ==>(b,s,e)=>(32, 61, 64,) 
# b==> batch, s==> seq length, e==> embedding, 
class Feed_forward(nn.Module): 
    def __init__(self, d_model=64,dropout=0.1,dim_feedforward=512,
                 layer_norm_eps=1e-5,
                 device=None, dtype=None) -> None:

        super(Feed_forward, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.relu = nn.ReLU()
        self.dropout1 = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout2 = Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        # print(x.shape)
        # src = self.norm(x)  ######
        src = x
        # print(src.shape)
        src2 = self.linear2(self.dropout1(self.relu(self.linear1(src))))
        # print(src2.shape)
        out = src + self.dropout2(src2)
        out = self.norm(out)
        return out

    # Best Model so far fine tuning
class Cross_Transformer_Network(nn.Module):
    def __init__(self,d_model = 64, dim_feedforward=512,window_size = 25): #  filt_ch = 4
        super(Cross_Transformer_Network, self).__init__()
        
        self.eeg_atten = Intra_modal_atten(d_model=d_model, nhead=8, dropout=0.1,
                                            window_size =window_size, First = True )
        self.eog_atten = Intra_modal_atten(d_model=d_model, nhead=8, dropout=0.1, 
                                            window_size =window_size, First = True )
        self.eeg2_atten = Intra_modal_atten(d_model=d_model, nhead=8, dropout=0.1, 
                                            window_size =window_size, First = True )
        
        self.cross_atten = Cross_modal_atten(d_model=d_model, nhead=8, dropout=0.1, First = True )
        
        self.eeg_ff = Feed_forward(d_model = d_model,dropout=0.1,dim_feedforward = dim_feedforward)
        self.eog_ff = Feed_forward(d_model = d_model,dropout=0.1,dim_feedforward = dim_feedforward)
        self.eeg2_ff = Feed_forward(d_model = d_model,dropout=0.1,dim_feedforward = dim_feedforward)



        self.mlp    = nn.Sequential(nn.Flatten(),
                                    nn.Linear(d_model*3,5))  ##################
        # 

    def forward(self, eeg: Tensor,eog: Tensor,eeg2: Tensor,finetune = True): 
        self_eeg = self.eeg_atten(eeg)
        self_eog = self.eog_atten(eog)
        self_eeg2 = self.eeg2_atten(eeg)
        # print(self_eeg.shape,self_eeg2.shape)
        self_eeg_new = torch.cat((self_eeg[:,0,:].unsqueeze(dim=1),self_eeg2[:,0,:].unsqueeze(dim=1)), dim=1)
        cross = self.cross_atten(self_eeg_new,self_eog[:,0,:])

        cross_cls = cross[:,0,:].unsqueeze(dim=1)
        cross_eeg = cross[:,1,:].unsqueeze(dim=1)
        cross_eog = cross[:,2,:].unsqueeze(dim=1)

        eeg_new =  torch.cat([cross_cls, self_eeg[:,1:,:]], dim=1)
        eog_new =  torch.cat([cross_cls, self_eog[:,1:,:]], dim=1)
        eeg2_new =  torch.cat([cross_cls, self_eeg2[:,1:,:]], dim=1)

        ff_eeg = self.eeg_ff(eeg_new)
        ff_eog = self.eog_ff(eog_new)
        ff_eeg2 = self.eeg2_ff(eeg2_new)

        

        # cls_out = torch.cat([cross_cls[:,0,:],ff_eeg[:,0,:], ff_eog[:,0,:]], dim=1).unsqueeze(dim=1) ######
        cls_out = torch.cat([ff_eeg[:,0,:], ff_eog[:,0,:],ff_eeg2[:,0,:]], dim=1).unsqueeze(dim=1) 

        feat_list = [cross_cls,ff_eeg,ff_eog,ff_eeg2]
        if finetune == True:
            out = self.mlp(cls_out)  #########
            return out,cls_out,feat_list
        else:
            return cls_out#,feat_list