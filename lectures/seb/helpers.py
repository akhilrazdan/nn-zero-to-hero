import torch
from torch import nn

class MultiHeadAttention(nn.Module): 
    def __init__(self, d_in, d_out, context_length, dropout, n_heads, qkv_biases=False):
        super().__init__()
        if d_out % n_heads != 0: 
            raise ValueError("Please make it divisible")
        
        self.n_heads = n_heads
        self.d_out = d_out
        self.head_dim = d_out // n_heads

        self.Wq = nn.Linear(d_in, d_out, bias=qkv_biases)
        self.Wk = nn.Linear(d_in, d_out, bias=qkv_biases)
        self.Wv = nn.Linear(d_in, d_out, bias=qkv_biases)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            'mask', 
            torch.triu(torch.ones((context_length, context_length)), diagonal=1)
        )

        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x): 
        b, t, d_in = x.shape
                
        q = self.Wq(x) # b, t, d_out
        k = self.Wk(x)
        v = self.Wv(x) # b, t, c

        q = q.view(b, t, self.n_heads, self.head_dim) # b, t, n_head, head_dim
        k = k.view(b, t, self.n_heads, self.head_dim)
        v = v.view(b, t, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2) # b, n_head, t, head_dim
        #print(q.shape, k.shape, v.shape)

        attn_scores = q @ k.transpose(2, 3) # b, n_head, t, head_dim @ b, n_head, head_dim, t
        attn_scores.masked_fill_(
            self.mask.bool()[:t, :t], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores, dim=-1)

        context_vec = (attn_weights @ v).transpose(1,2) # b, n_head, t, t @ b, n_head, t, head_dim = b, n_head, t, head_dim
        # print(context_vec.shape, self.d_out)
        context_vec = context_vec.contiguous().view(b, t, self.d_out)

        context_vec = self.out_proj(context_vec)
        return context_vec