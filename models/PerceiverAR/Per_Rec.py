
import torch
import torch.nn as nn
import math
import logging
from torch.nn import functional as F
logger = logging.getLogger(__name__)


class CrossAttention(nn.Module):
    def __init__(self, args, n_heads=16, bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.maxlen = args.maxlen
        self.scale = (args.hidden_units // n_heads) ** -0.5       # 1/sqrt(d)
        self.q = nn.Linear(args.hidden_units, args.hidden_units, bias = bias)
        self.k = nn.Linear(args.hidden_units, args.hidden_units, bias = bias)
        self.v = nn.Linear(args.hidden_units, args.hidden_units, bias = bias)

        self.o = nn.Linear(args.hidden_units, args.hidden_units, bias = bias)

        self.register_buffer("bias", torch.tril(torch.ones(self.maxlen, self.maxlen * 2)).view(1, 1, self.maxlen, self.maxlen * 2))

    def forward(self, kv, q):
        B, L_kv, D_kv = kv.shape
        B, L_q,  D_q = q.shape  # (128,1,50)

        head_dim = D_q // self.n_heads

        q = self.q(q)
        k = self.k(kv)
        v = self.v(kv)

        q = torch.reshape(q, [B, L_q, self.n_heads, head_dim])     # B, L_q,  nh,  i
        q = torch.permute(q, [0, 2, 1, 3])                   # B, nh,   L_q, i

        k = torch.reshape(k, [B, L_kv, self.n_heads, head_dim])    # B, L_kv, nh,  i
        k = torch.permute(k, [0, 2, 3, 1])                   # B, nh,   i,   L_kv

        v = torch.reshape(v, [B, L_kv, self.n_heads, head_dim])    # B, L_kv, nh,   i
        v = torch.permute(v, [0, 2, 1, 3])                   # B, nh,   L_kv, i

        qk = torch.matmul(q, k) * self.scale                 #(B, nh, L_q, i)(B, nh, i, L_kv)
                                                             # B, nh, L_q, L_kv

        qk = qk.masked_fill(self.bias[:,:,:L_q,:L_kv] == 0, float('-inf'))

        attn = torch.softmax(qk, dim=-1)

        v_attn = torch.matmul(attn, v)                       #(B, nh, L_q, L_kv)(B, nh, L_kv, i)
                                                             # B, nh, L_q, i
        v_attn = torch.permute(v_attn, [0, 2, 1, 3])         # B, L_q, nh, i
        v_attn = torch.reshape(v_attn, [B, L_q, D_q])        # B, L_q, D_q

        x = self.o(v_attn)
        return x


class SelfAttention(nn.Module):
    def __init__(self, args, n_heads=16, bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.maxlen = args.maxlen
        self.scale = (args.hidden_units // n_heads) ** -0.5
        self.qw = nn.Linear(args.hidden_units, args.hidden_units, bias = bias)
        self.kw = nn.Linear(args.hidden_units, args.hidden_units, bias = bias)
        self.vw = nn.Linear(args.hidden_units, args.hidden_units, bias = bias)

        self.ow = nn.Linear(args.hidden_units, args.hidden_units, bias = bias)
        self.register_buffer("bias", torch.tril(torch.ones(self.maxlen, self.maxlen)).view(1, 1, self.maxlen, self.maxlen))

    def forward(self, x):
        B, L, D = x.shape
        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)

        B, L, D = q.shape
        q = torch.reshape(q, [B, L, self.n_heads, -1])
        q = torch.permute(q, [0, 2, 1, 3])
        k = torch.reshape(k, [B, L, self.n_heads, -1])
        k = torch.permute(k, [0, 2, 3, 1])
        v = torch.reshape(v, [B, L, self.n_heads, -1])
        v = torch.permute(v, [0, 2, 1, 3])

        qk = torch.matmul(q, k) * self.scale
        qk = qk.masked_fill(self.bias[:,:,:L,:L] == 0, float('-inf'))

        attn = torch.softmax(qk, dim=-1)

        v_attn = torch.matmul(attn, v)
        v_attn = torch.permute(v_attn, [0, 2, 1, 3])
        v_attn = torch.reshape(v_attn, [B, L, D])

        x = self.ow(v_attn)
        return x


class CrossTransformer(nn.Module):
    def __init__(self, args, n_heads=16, mlp_dim=4096, rate=0.1):
        super().__init__()
        self.maxlen = args.maxlen

        self.ln_1 = nn.LayerNorm(args.hidden_units)

        self.attn = CrossAttention(args)

        self.ln_2 = nn.LayerNorm(args.hidden_units)

        self.mlp = nn.Sequential(
            nn.Linear(args.hidden_units, mlp_dim),
            nn.ReLU(),
            nn.Dropout(rate),
            nn.Linear(mlp_dim, args.hidden_units),
            nn.Dropout(rate),
        )

    def forward(self, kv, q):
        x = self.attn(self.ln_1(kv), self.ln_1(q)) + q
        return self.mlp(self.ln_2(x)) + x


class SelfTransformer(nn.Module):
    def __init__(self, args, n_heads=16, mlp_dim=4096, rate=0.0):
        super().__init__()
        self.maxlen = args.maxlen
        self.ln_1 = nn.LayerNorm(args.hidden_units)
        self.attn = SelfAttention(args)
        self.ln_2 = nn.LayerNorm(args.hidden_units)
        self.mlp = nn.Sequential(
            nn.Linear(args.hidden_units, mlp_dim),
            nn.ReLU(),
            nn.Dropout(rate),
            nn.Linear(mlp_dim, args.hidden_units),
            nn.Dropout(rate),
        )

    def forward(self, x):
        x = self.attn(self.ln_1(x)) + x
        return self.mlp(self.ln_2(x)) + x


class PerceiverAR(nn.Module):
    def __init__(self, user_num, item_num, args, depth=5,
                 mlp_dim=4096, rate=0.2):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # num latents = spanish len
        self.maxlen = args.maxlen

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        #  self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)  # TO IMPROVE

        #  self.embedding = nn.Embedding(vocab_size, input_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.maxlen * 2, args.hidden_units))

        self.cross_attn = CrossTransformer(args)

        self.transformer = nn.Sequential()

        self.transformer = nn.ModuleList([
            SelfTransformer(args) for _ in range(depth)
        ])

        self.head = nn.Linear(args.hidden_units, args.hidden_units, bias=False)

    def Perceiver(self, log_seqs):

        B, L = log_seqs.shape

        x = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # [128, 200, 50]

        # x = self.embedding(x)
        x += self.pos_embedding[:, :L]  # [128, 200, 50]

        # 修改y的切片起点为序列中间，向下取整
        mid_index = self.maxlen // 2
        y = x
        # y = x[:, mid_index:]  # 从序列中间位置到末尾的部分 (128,100,50)
        # y = x[:, -1:] # size=(128, 1, 50)
        # y = x[:, self.maxlen:]  # size=(128, 0, 50)

        x = self.cross_attn(kv = x, q = y)

        for layer in self.transformer:
            x = layer(x)  #  [128, 1, 64]


        x = self.head(x)
        return x

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.Perceiver(log_seqs)  # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs)
        print("pos logits shape:{}".format(pos_logits.shape))
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.Perceiver(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)
