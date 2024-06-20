########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

def RWKV_Init(module, item_num, args, rwkv_emb_scale = 4): # fancy initialization of all lin & emb layer in the module
    for m in module.modules():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        with torch.no_grad():
            name = '[unknown weight]'
            for name, parameter in module.named_parameters(): # find the name of the weight
                if id(m.weight) == id(parameter):
                    break

            shape = m.weight.data.shape
            gain = 1.0  # positive: gain for orthogonal, negative: std for normal
            scale = 1.0 # extra scale for gain

            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == item_num and shape[1] == args.hidden_units: # final projection?
                    scale = rwkv_emb_scale

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == item_num and shape[1] == args.hidden_units: # token emb?
                    scale = rwkv_emb_scale

            if hasattr(m, 'scale_init'):
                scale = m.scale_init

            print(str(shape[0]).ljust(5), str(shape[1]).ljust(5), f'{round(scale,2):g}'.ljust(4), name)

            gain *= scale
            if gain == 0:
                nn.init.zeros_(m.weight) # zero init is great for some RWKV matrices
            elif gain > 0:
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                nn.init.normal_(m.weight, mean=0, std=-gain)

class RWKV_TimeMix(nn.Module):
    def __init__(self, user_num, item_num, args, layer_id, maxlen=200, n_head=8, n_attn=8):
        super().__init__()
        self.layer_id = layer_id

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)

        self.n_head = n_head
        self.n_attn = n_attn
        assert self.n_attn % self.n_head == 0
        #  self.layer_id = layer_id
        self.maxlen = maxlen
        self.head_size = self.n_attn // self.n_head

        with torch.no_grad(): # initial time_w curves for better convergence
            ww = torch.ones(self.n_head, self.maxlen)
            curve = torch.tensor([-(self.maxlen - 1 - i) for i in range(self.maxlen)]) # the distance
            for h in range(self.n_head):
                if h < self.n_head - 1:
                    decay_speed = math.pow(self.maxlen, -(h+1)/(self.n_head-1))
                else:
                    decay_speed = 0.0
                ww[h] = torch.exp(curve * decay_speed)
                # print('layer', layer_id, 'head', h, 'decay_speed', round(decay_speed, 4), ww[h][:5].numpy(), '...', ww[h][-5:].numpy())
        self.time_w = nn.Parameter(ww)

        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, self.maxlen))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, self.maxlen, 1))
        self.time_gamma = nn.Parameter(torch.ones(self.maxlen, 1))
                
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))

        self.key = nn.Linear(args.hidden_units, self.n_attn)
        self.value = nn.Linear(args.hidden_units, self.n_attn)
        self.receptance = nn.Linear(args.hidden_units, self.n_attn)

        # if config.rwkv_tiny_attn > 0:
        #     self.tiny_att = RWKV_TinyAttn(config)

        self.output = nn.Linear(self.n_attn, args.hidden_units)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def rwkv(self, log_seqs):
        x = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # [128, 200, 100]
        B, T, C = x.size()
        TT = self.maxlen
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT-1:] # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim=-1)
        # if hasattr(self, 'tiny_att'):
        #     tiny_att = self.tiny_att(x, self.mask)

        k = self.key(x)  # [128, 200, 8]
        v = self.value(x)
        r = self.receptance(x)

        k = torch.clamp(k, max=30, min=-60) # clamp extreme values. e^30 = 10^13
        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)

        kv = (k * v).view(B, T, self.n_head, self.head_size)

        wkv = (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)

        rwkv = torch.sigmoid(r) * wkv / sum_k

        rwkv = self.output(rwkv)
        rwkv = rwkv * self.time_gamma[:T, :]
        # if hasattr(self, 'tiny_att'):
        #     rwkv += tiny_att

        return rwkv

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
        log_feats = self.rwkv(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs)
        print("pos logits shape:{}".format(pos_logits.shape))
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.rwkv(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)



class RWKV_ChannelMix(nn.Module):
    def __init__(self, user_num, item_num, args, layer_id, maxlen=200, n_head=8, n_attn=8, n_ffn=4):
        super().__init__()
        self.layer_id = layer_id

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)

        # self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        self.n_head = n_head
        self.n_ffn = n_ffn
        
        hidden_sz = 5 * self.n_ffn // 2 # can use smaller hidden_sz because of receptance gating
        self.key = nn.Linear(args.hidden_units, hidden_sz)
        self.value = nn.Linear(args.hidden_units, hidden_sz)
        self.weight = nn.Linear(hidden_sz, args.hidden_units)
        self.receptance = nn.Linear(args.hidden_units, args.hidden_units)

        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def rwkv(self, log_seqs):
        x = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # [128, 200, 100]
        B, T, C = x.size()
        
        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        
        wkv = self.weight(F.mish(k) * v) # i find mish is a bit better than gelu

        rwkv = torch.sigmoid(r) * wkv

        return rwkv

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
        log_feats = self.rwkv(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs)
        print("pos logits shape:{}".format(pos_logits.shape))
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.rwkv(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)



class RWKV_TinyAttn(nn.Module): # extra tiny attention

    def __init__(self, user_num, item_num, args, layer_id, maxlen=200, n_head=2, d_attn=2):
        super().__init__()
        self.layer_id = layer_id

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)

        self.d_attn = d_attn
        self.n_head = n_head
        self.head_size = self.d_attn // self.n_head

        self.qkv = nn.Linear(args.hidden_units, self.d_attn * 3)
        self.out = nn.Linear(self.d_attn, args.hidden_units)

    def rwkv(self, log_seqs):
        x = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # [128, 200, 100]
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim = -1)

        if self.n_head > 1:
            q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)      # (B, T, C) -> (B, nh, T, hs)
            k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)      # (B, T, C) -> (B, nh, T, hs)
            v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)      # (B, T, C) -> (B, nh, T, hs)

        qk = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))     # (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        # mask
        mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        mask = mask.unsqueeze(1).unsqueeze(1)  # 扩展维度以便与qk维度匹配 (B, 1, 1, T)
        qk = qk.masked_fill(mask.expand_as(qk), float('-inf'))

        qk = F.softmax(qk, dim = -1)
        qkv = qk @ v                                                           # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)

        if self.n_head > 1:
            qkv = qkv.transpose(1, 2).contiguous().view(B, T, -1)              # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        rwkv = self.out(qkv)

        return rwkv

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
        log_feats = self.rwkv(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs)
        print("pos logits shape:{}".format(pos_logits.shape))
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.rwkv(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)



########################################################################################################
# MHA_rotary: Multi-head Attention + Rotary Encoding + GeGLU FFN
########################################################################################################

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached, self.sin_cached

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos[...,:q.shape[-2],:], sin[...,:q.shape[-2],:]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class MHA_rotary(nn.Module):
    def __init__(self, user_num, item_num, args, layer_id, maxlen=200, n_head=8, n_attn=8, time_shift=False):
        super().__init__()
        self.layer_id = layer_id

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)

        self.n_head = n_head
        self.n_attn = n_attn
        assert self.n_attn % self.n_head == 0
        #  self.layer_id = layer_id
        self.maxlen = maxlen
        self.head_size = self.n_attn // self.n_head

        if time_shift:
            self.time_shift = nn.ZeroPad2d((0,0,1,-1))

        self.query = nn.Linear(args.hidden_units, self.n_attn)
        self.key = nn.Linear(args.hidden_units, self.n_attn)
        self.value = nn.Linear(args.hidden_units, self.n_attn)

        self.register_buffer("mask", torch.tril(torch.ones(self.maxlen, self.maxlen)))
        
        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.output = nn.Linear(self.n_attn, args.hidden_units)

    def MHA_rotary(self, log_seqs):
        x = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # [128, 200, 100]
        B, T, C = x.size()

        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)

        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)         # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)

        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)                                     # rotary encoding
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))                 # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.mask[:T,:T] == 0, float('-inf'))                     # causal mask
        att = F.softmax(att, dim = -1)                                                  # softmax

        x = att @ v                                                                     # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)                               # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x)
        return x

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
        log_feats = self.MHA_rotary(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs)
        print("pos logits shape:{}".format(pos_logits.shape))
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.MHA_rotary(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)

class GeGLU(torch.nn.Module):
    def __init__(self, user_num, item_num, args, layer_id, n_ffn=4, time_shift=False):
        super().__init__()
        self.layer_id = layer_id

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)

        if time_shift:
            self.time_shift = nn.ZeroPad2d((0,0,1,-1))

        self.n_ffn = n_ffn
        hidden_sz = 3 * self.n_ffn
        self.key = nn.Linear(args.hidden_units, hidden_sz)
        self.value = nn.Linear(args.hidden_units, hidden_sz)
        self.weight = nn.Linear(hidden_sz, args.hidden_units)

    def GeGLU(self, log_seqs):
        x = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # [128, 200, 100]
        B, T, C = x.size()
        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        
        k = self.key(x)
        v = self.value(x)        
        y = self.weight(F.gelu(k) * v)
        return y

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
        log_feats = self.GeGLU(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs)
        print("pos logits shape:{}".format(pos_logits.shape))
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.GeGLU(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)


########################################################################################################
# MHA_pro: with more tricks
########################################################################################################

class MHA_pro(nn.Module):
    def __init__(self, user_num, item_num, args, layer_id, maxlen=200, n_head=8, n_attn=8):
        super().__init__()
        self.layer_id = layer_id

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)

        self.n_head = n_head
        self.n_attn = n_attn
        assert self.n_attn % self.n_head == 0
        #  self.layer_id = layer_id
        self.maxlen = maxlen
        self.head_size = self.n_attn // self.n_head

        self.time_w = nn.Parameter(torch.ones(self.n_head, self.maxlen))
        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, self.maxlen))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, self.maxlen, 1))
        self.time_gamma = nn.Parameter(torch.ones(self.maxlen, 1))
        self.register_buffer("mask", torch.tril(torch.ones(self.maxlen, self.maxlen)))

        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        self.query = nn.Linear(args.hidden_units, self.n_attn)
        self.key = nn.Linear(args.hidden_units, self.n_attn)
        self.value = nn.Linear(args.hidden_units, self.n_attn)
        
        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.head_mix = nn.Conv2d(self.n_head, self.n_head, kernel_size=1, bias=False)  # talking heads

        self.output = nn.Linear(self.n_attn, args.hidden_units)

    def MHA_pro(self, log_seqs):
        x = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # [128, 200, 100]
        B, T, C = x.size()
        TT = self.maxlen
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT-1:] # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)      # time-shift mixing
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)         # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)

        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)                                     # rotary encoding
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)  
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))                 # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.mask[:T,:T] == 0, float('-inf'))                     # causal mask
        att = F.softmax(att, dim = -1)                                                  # softmax
        att = att * w                                                                   # time-weighting
        att = self.head_mix(att)                                                        # talking heads

        x = att @ v                                                                     # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)                               # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x) * self.time_gamma[:T, :]
        return x

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
        log_feats = self.MHA_pro(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs)
        print("pos logits shape:{}".format(pos_logits.shape))
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.MHA_pro(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)

########################################################################################################
# The GPT Model with our blocks
########################################################################################################

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1. / 2)
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm_x * self.dd + 1e-12)
        return self.weight * x_normed

class FixedNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1. / 2)

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm_x * self.dd + 1e-12)
        return x_normed

########################################################################################################

class GPTConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k,v in kwargs.items():
            setattr(self, k, v)

class Block(nn.Module):
    def __init__(self, user_num, item_num, args, layer_id, model_type=None):
        super().__init__()
        # self.config = config

        self.ln1 = nn.LayerNorm(args.hidden_units)
        self.ln2 = nn.LayerNorm(args.hidden_units)

        if model_type == 'RWKV':
            # self.ln1 = FixedNorm(config.n_embd)
            # self.ln2 = FixedNorm(config.n_embd)
            self.attn = RWKV_TimeMix(user_num, item_num, args, layer_id)
            self.mlp = RWKV_ChannelMix(user_num, item_num, args, layer_id)

        elif model_type == 'MHA_rotary':
            self.attn = MHA_rotary(user_num, item_num, args, layer_id)
            self.mlp = GeGLU(user_num, item_num, args, layer_id)
        
        elif model_type == 'MHA_shift':
            self.attn = MHA_rotary(user_num, item_num, args, layer_id, time_shift=True)
            self.mlp = GeGLU(user_num, item_num, args, layer_id, time_shift=True)
        
        elif model_type == 'MHA_pro':
            self.attn = MHA_pro(user_num, item_num, args, layer_id)
            self.mlp = RWKV_ChannelMix(user_num, item_num, args, layer_id)

    def forward(self, x):

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        
        return x

class GPT(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args,
                 maxlen=200,
                 n_layer=12,
                 model_type="RWKV",
                 n_head=8,
                 n_attn=8,
                 n_ffn=4,
                 rwkv_emb_scale=4
                 ):
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)

        self.maxlen = maxlen
        self.n_layer = n_layer
        self.model_type = model_type
        self.n_head = n_head
        self.n_attn = n_attn
        self.n_ffn = n_ffn
        self.rwkv_emb_scale = rwkv_emb_scale

        self.blocks = nn.Sequential(*[Block(self.user_num, self.item_num, args,  i, self.model_type) for i in range(self.n_layer)])

        self.ln_f = nn.LayerNorm(args.hidden_units)
        self.time_out = nn.Parameter(torch.ones(1, self.maxlen, 1))  # reduce confidence of early tokens
        self.head = nn.Linear(args.hidden_units, self.item_num + 1, bias=False)

        self.head_q = nn.Linear(args.hidden_units, 256)
        self.head_q.scale_init = 0.01
        self.head_k = nn.Linear(args.hidden_units, 256)
        self.head_k.scale_init = 0.01
        self.register_buffer("copy_mask", torch.tril(torch.ones(args.hidden_units, args.hidden_units)))


        if self.model_type == 'RWKV':
            RWKV_Init(self, self.item_num, args, self.rwkv_emb_scale)
        else:
            self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_ctx_len(self):
        return self.maxlen

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def configure_optimizers(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (RMSNorm, nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias') or ('time' in fpn) or ('head' in fpn):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps)
        return optimizer

    def GPT(self, log_seqs):
        breakpoint()
        x = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # [128, 200, 100]
        B, T, C = x.size()
       # assert T <= self.maxlen, "Cannot forward, because len(input) > model ctx_len."

        x = self.blocks(x)
        # TypeError: forward() missing 3 required positional arguments: 'log_seqs', 'pos_seqs', and 'neg_seqs'

        x = self.ln_f(x)

        q = self.head_q(x)[:,:T,:]
        k = self.head_k(x)[:,:T,:]
        c = (q @ k.transpose(-2, -1)) * (1.0 / 256)
        c = c.masked_fill(self.copy_mask[:T,:T] == 0, 0)
        print("C shape :{}.".format(c.shape))
        #c = c @ F.one_hot(idx, num_classes = self.config.vocab_size).float()

        x = x * self.time_out[:, :T, :] # reduce confidence of early tokens
        # x = self.head(x) + c #[Batch,seq,768]

        return x

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
        log_feats = self.GPT(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs)
        print("pos logits shape:{}".format(pos_logits.shape))
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.GPT(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
