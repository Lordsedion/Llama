import torch # type: ignore
from torch import nn # type: ignore
from torch.nn import functional as F # type: ignore
import numpy as np # type: ignore
from collections import OrderedDict


device = torch.device("cuda") if torch.cuda.is_available() == True else torch.device("cpu")

def get_rotary_matrix(context_window, embedding_dim, device_):
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False).to(device_)
    for position in range(context_window):
        for i in range(embedding_dim//2):
            theta = 10000. ** (-2.*(i - 1) / embedding_dim)
            m_theta = position * theta
            R[position, 2*i,2*i] = np.cos(m_theta)
            R[position, 2*i, 2*i+1] = - np.sin(m_theta)
            R[position, 2*i+1, 2*i] = np.sin(m_theta)
            R[position, 2*i+1, 2*i+1] = np.cos(m_theta)
    return R


class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))
        
    def forward(self, x):
        ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw
    

class RopeAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w_q = nn.Linear(config["d_model"], config["d_model"], bias=False)
        self.w_k = nn.Linear(config["d_model"], config["d_model"], bias=False)
        self.w_v = nn.Linear(config["d_model"], config["d_model"], bias=False)
        
        self.device = device
        self.R = get_rotary_matrix(config["context_window"], config["d_model"], self.device)
               
    
    def forward(self, x, return_attn_weights=False):
        b,m,d = x.shape
#         print(self.device)
        q = self.w_q(x).to(self.device)
        k = self.w_k(x).to(self.device)
        v = self.w_v(x).to(self.device)
        
        self.R = self.R.to(self.device)
        
#         print(self.R.get_device())
                        
        q_rotated = (torch.bmm(q.transpose(0,1), self.R[:m])).transpose(0,1)
        k_rotated = (torch.bmm(k.transpose(0,1), self.R[:m])).transpose(0,1)
        
        activations = F.scaled_dot_product_attention(q_rotated, k_rotated,
                                                     v, dropout_p=.1, is_causal=True)
        
        if return_attn_weights:
            attn_mask = torch.tril(torch.ones((m,m)), diagonal=0).to(self.device)
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1,2)) / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights
        return activations


class RopeMultiAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList(
            RopeAttentionHead(config) for _ in range(config["n_heads"])
        )
        self.linear = nn.Linear(config["n_heads"] * config["d_model"], config["d_model"])
        self.dropout = nn.Dropout(.1)
        
    def forward(self, x):
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        
        return x
    

class SwiGLU(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = torch.randn(1, requires_grad=True)
        
        self.beta = nn.Parameter(torch.ones(1))
        self.register_parameter("beta", self.beta)
    
    def forward(self, x):
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)
        return out
    

class LLamaBlock(nn.Module):
    def __init__(self,  config):
        super().__init__()
        self.config = config
        
        self.rms = RMSNorm((config["context_window"], config["d_model"]))
        self.attention = RopeMultiAttentionHead(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"]),
            SwiGLU(config["d_model"])
        )
    
    def forward(self, x):
        x = self.rms(x)
        x = x + self.attention(x)
        
        x = self.rms(x)
        x = x + self.feedforward(x)
        
        return x
    

class LLama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        self.llama_block = nn.Sequential(
            OrderedDict([(f"llama_{i}", LLamaBlock(config)) for i in range(config['n_layers'])])
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(config["d_model"], config['d_model']),
            SwiGLU(config["d_model"]),
            nn.Linear(config["d_model"], config["vocab_size"])
        )
        
        print(f"Model Parameters: {sum(m.numel() for m in self.parameters())}")
    
    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        x = self.llama_block(x)
        logits = self.ffn(x)
        
        if targets is None:
            return logits
        else:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss