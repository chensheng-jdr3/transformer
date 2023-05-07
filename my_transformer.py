import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def mask_attention(query,key,value,mask=None,dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)    #compute and scale the score
    if mask is not None:
        scores = scores.masked_fill(mask==0,-1e9)
    p_attn = F.softmax(scores,dim=-1)   #softmax
    if dropout is not None:
        p_attn = dropout(p_attn)    #dropout
    return torch.matmul(p_attn,value),p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self ,dim ,head = 8,mask = None ,dropout = True):
        super(MultiHeadAttention, self).__init__()
        self.mask = mask
        self.dropout = dropout
        self.head = head
        self.linear = []
        for i in range(head):
            self.linear.append(nn.Linear(dim ,dim/head , bias=None))

    def forward(self, query, key, value):
        learned_value = []
        query_list = []
        key_list = []
        value_list = []
        for i in range(self.head):
            query_list.append(self.linear[i](x))     #分为多个头
            key_list.append(self.linear[i](x))
            value_list.append(self.linear[i](x))
            learned_value[i],_ = mask_attention(query[i], key[i], value[i], self.mask ,self.dropout)
        out_put = torch.cat(learned_value,dim = -1) + value_list
        out_put = F.layer_norm(out_put)
        return out_put

class feedforward(nn.Module):
    def __init__(self, dim):
        super(feedforward,self).__init__()
        self.linear1 = nn.Linear(dim,dim*4)
        self.linear2 = nn.Linear(dim*4,dim)

    def forward(self,x):
        y = self.linear1(x)
        y = F.relu(y)
        y = self.linear2(y)
        y = y + x
        y = F.layer_norm(y)
        return y

def positional_encode(input):
    batch,maxwords,embedding = input.size
    positionalencode = torch.zero(batch,maxwords,embedding)
    for i in range(maxwords):
        for j in range(0, embedding, 2):
            positionalencode[:][i][j] = math.sin(i/(10000**(2*i/embedding)))
            positionalencode[:][i][j+1] = math.cos(i/(10000**(2*i/embedding)))
    output = input + positionalencode
    return output

class Encoder(nn.module):
    def __init__(self ,dim ,head ,):
        super(Encoder,self).__init__()
        self.multiheadattention = MultiHeadAttention(dim = dim, head = head, mask=True)
        self.feedforward = feedforward(dim=dim)

    def forward(self,x):
        y = self.multiheadattention(x,x,x)
        y = self.feedforward(y)
        return

class Decoder(nn.Module):
    def __init__(self ,dim ,head,  ):
        super(Decoder,self).__init__()
        self.encodermultihead = MultiHeadAttention(dim = dim, head = head, mask=ture)
        self.maskedmultihead = MultiHeadAttention(dim = dim, head = head, mask=ture)
        self.feedforward = feedforward(dim = dim)
        self.linear = nn.Linear(dim,111词源表长度)

    def forward(self,encoder):
        outputs = ["start"]
        while(outputs[-1] != "end"):
            x = self.maskedmultihead(outputs, outputs, outputs)
            x = x + outputs
            y = F.layer_norm(x)
            x = self.encodermultihead(encoder, encoder, y)
            x = x + y
            y = F.layer_norm(x)
            x = self.feedforward(y)
            x = x + y
            x = F.layer_norm(x)
            x = self.linear(x)
            x = F.softmax(x)
            outputs.append(x)
        


