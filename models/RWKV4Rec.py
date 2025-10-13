import numpy as np
import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

def RWKV_Init(module, item_num, args):  # fancy initialization of all lin & emb layer in the module
    for m in module.modules():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        with torch.no_grad():
            name = '[unknown weight]'
            for name, parameter in module.named_parameters():  # find the name of the weight
                if id(m.weight) == id(parameter):
                    break

            shape = m.weight.data.shape
            gain = 1.0  # positive: gain for orthogonal, negative: std for normal
            scale = 1.0  # extra scale for gain

            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == item_num and shape[1] == args.hidden_units:  # final projection?
                    scale = args.rwkv_emb_scale

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == item_num and shape[1] == args.hidden_units:  # token emb?
                    scale = args.rwkv_emb_scale

            if hasattr(m, 'scale_init'):
                scale = m.scale_init

            print(str(shape[0]).ljust(5), str(shape[1]).ljust(5), f'{round(scale, 2):g}'.ljust(4), name)

            gain *= scale
            if gain == 0:
                nn.init.zeros_(m.weight)  # zero init is great for some RWKV matrices
            elif gain > 0:
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                nn.init.normal_(m.weight, mean=0, std=-gain)


class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank, alpha=1.0):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer  # 原始线性层
        self.rank = rank
        self.alpha = alpha

        # 获取原始层的权重和偏置
        self.original_weight = original_layer.weight
        self.original_bias = original_layer.bias

        # LoRA的低秩矩阵
        self.A = nn.Parameter(torch.randn(original_layer.weight.size(0), rank))
        self.B = nn.Parameter(torch.zeros(rank, original_layer.weight.size(1)))

    def forward(self, x):
        # 原始层的输出
        original_output = F.linear(x, self.original_weight, self.original_bias)

        # LoRA的低秩更新
        lora_update = self.alpha * (x @ self.A.T @ self.B.T)  # 低秩更新
        return original_output + lora_update

class RWKV_TimeMix(nn.Module):
    def __init__(self, args, layer_id, n_head=8, n_attn=8, n_layer=12):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.use_lora = False  # 新增布尔参数控制是否使用LoRA
        self.lora_rank = args.lora_rank if self.use_lora else 0  # 只有使用LoRA时才设置rank

        self.n_head = n_head
        self.n_attn = n_attn
        assert self.n_attn % self.n_head == 0
        self.layer_id = layer_id
        self.maxlen = args.maxlen
        self.head_size = self.n_attn // self.n_head

        with torch.no_grad():  # initial time_w curves for better convergence
            ratio_0_to_1 = (layer_id / (self.n_layer - 1))  # 0 to 1
            ratio_1_to_almost0 = (1.0 - (layer_id / self.n_layer))  # 1 to ~0

            ww = torch.ones(self.n_head, self.maxlen)
            curve = torch.tensor([-(self.maxlen - 1 - i) for i in range(self.maxlen)])  # the distance
            for h in range(self.n_head):
                if h < self.n_head - 1:
                    decay_speed = math.pow(self.maxlen, -(h + 1) / (self.n_head - 1))
                else:
                    decay_speed = 0.0
                ww[h] = torch.exp(curve * decay_speed)
                # print('layer', layer_id, 'head', h, 'decay_speed', round(decay_speed, 4), ww[h][:5].numpy(), '...', ww[h][-5:].numpy())

            # fancy time_mix
            x = torch.ones(1, 1, args.hidden_units)
            for i in range(args.hidden_units):
                x[0, 0, i] = i / args.hidden_units
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        self.time_w = nn.Parameter(ww)

        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, self.maxlen))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, self.maxlen, 1))
        self.time_gamma = nn.Parameter(torch.ones(self.maxlen, 1))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        if self.use_lora:
            # LoRA模式：直接创建带LoRA的线性层
            self.key = LoRALayer(nn.Linear(args.hidden_units, self.n_attn), self.lora_rank)
            self.value = LoRALayer(nn.Linear(args.hidden_units, self.n_attn), self.lora_rank)
            self.receptance = LoRALayer(nn.Linear(args.hidden_units, self.n_attn), self.lora_rank)
        else:
            # 非LoRA模式：直接使用原生线性层
            self.key = nn.Linear(args.hidden_units, self.n_attn)
            self.value = nn.Linear(args.hidden_units, self.n_attn)
            self.receptance = nn.Linear(args.hidden_units, self.n_attn)


        self.output = nn.Linear(self.n_attn, args.hidden_units)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        TT = self.maxlen
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT - 1:]  # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]  # (self.n_head, T, T)

        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x)  # [128, 200, 50]
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)

        k = torch.clamp(k, max=30, min=-60)  # clamp extreme values. e^30 = 10^13
        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)

        kv = (k * v).view(B, T, self.n_head, self.head_size)

        wkv = (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)

        rwkv = torch.sigmoid(r) * wkv / sum_k

        rwkv = self.output(rwkv)

        return rwkv * self.time_gamma[:T, :]



class RWKV_ChannelMix(nn.Module):
    def __init__(self, args, layer_id, n_head=8, n_ffn=4):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.n_head = n_head
        # self.n_ffn = n_ffn

        with torch.no_grad():  # init to "shift half of the channels"
            x = torch.ones(1, 1, args.hidden_units)
            for i in range(args.hidden_units // 2):
                x[0, 0, i] = 0
        self.time_mix = nn.Parameter(x)

        # hidden_sz = 5 * self.n_ffn // 2  # can use smaller hidden_sz because of receptance gating
        hidden_sz = 4 * args.hidden_units
        self.key = nn.Linear(args.hidden_units, hidden_sz)
        self.value = nn.Linear(args.hidden_units, hidden_sz)
        self.weight = nn.Linear(hidden_sz, args.hidden_units)
        self.receptance = nn.Linear(args.hidden_units, args.hidden_units)

        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()

        #  x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        x = x * self.time_mix + self.time_shift(x) * (1 - self.time_mix)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        wkv = self.weight(F.mish(k) * v)  # i find mish is a bit better than gelu

        rwkv = torch.sigmoid(r) * wkv

        return rwkv

class RWKV_ExtraAttn(nn.Module):
    def __init__(self, config, n_attn=128):
        super().__init__()
        self.n_attn = n_attn
        self.n_head = config.num_heads
        self.head_size = self.n_attn // self.n_head

        self.qkv = nn.Linear(config.hidden_units, self.n_attn * 3)
        self.out = nn.Linear(self.n_attn, config.hidden_units)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        if self.n_head > 1:
            q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        qk = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        qk = F.softmax(qk, dim=-1)
        attn_output = qk @ v

        if self.n_head > 1:
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)

        return self.out(attn_output)


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

class Block(nn.Module):
    def __init__(self, args, layer_id, model_type):
        super().__init__()
        #  self.config = config

        self.ln1 = nn.LayerNorm(args.hidden_units)
        self.ln2 = nn.LayerNorm(args.hidden_units)
        # self.ln_extra_attn = nn.LayerNorm(args.hidden_units)  # 新增的LayerNorm层
        # self.ln1 = RMSNorm(args.hidden_units)
        # self.ln2 = RMSNorm(args.hidden_units)

        # self.ln1 = FixedNorm(args.hidden_units)
        # self.ln2 = FixedNorm(args.hidden_units)

        if model_type == 'RWKV':
            # self.attn = RWKV_TimeMix(args, layer_id)
            self.mlp = RWKV_ChannelMix(args, layer_id)
            self.attn = RWKV_ExtraAttn(args)  # 新增的注意力层


    def forward(self, x):
        x = x + self.ln1(self.attn(x))
        x = x + self.ln2(self.mlp(x))
        # x = x + self.ln2(self.extra_attn(x))

        return x


class RWKV4Rec(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args,
                 n_layer=12,
                 model_type="RWKV",
                 n_head=8,
                 n_attn=8,
                 ):
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.maxlen = args.maxlen
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        # self.user_emb = nn.Embedding(self.user_num + 1, args.hidden_units)  # TO IMPROVE

        self.n_layer = n_layer
        self.model_type = model_type
        self.n_head = n_head
        self.n_attn = n_attn
        self.rwkv_emb_scale = args.rwkv_emb_scale

        self.blocks = nn.Sequential(*[Block(args, i, self.model_type) for i in range(self.n_layer)])

        self.ln_f = nn.LayerNorm(args.hidden_units)
        # self.ln_f = FixedNorm(args.hidden_units)
        self.time_out = nn.Parameter(torch.ones(1, self.maxlen, 1))  # reduce confidence of early tokens
        self.head = nn.Linear(args.hidden_units, self.item_num + 1, bias=False)

        self.head_q = nn.Linear(args.hidden_units, 256)
        self.head_q.scale_init = 0.01
        self.head_k = nn.Linear(args.hidden_units, 256)
        self.head_k.scale_init = 0.01
        self.register_buffer("copy_mask", torch.tril(torch.ones(args.hidden_units, args.hidden_units)))

        if self.model_type == 'RWKV':
            RWKV_Init(self, self.item_num, args)
        else:
            self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def GPT(self, log_seqs):
        x = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # [128, 200, 100]

        # 添加用户信息
        # use_info = self.user_emb(torch.LongTensor(user_ids).unsqueeze(1).to(self.dev))
        # x += use_info

        B, T, C = x.size()

        x = self.blocks(x)

        x = self.ln_f(x)

        q = self.head_q(x)[:, :T, :]
        k = self.head_k(x)[:, :T, :]
        c = (q @ k.transpose(-2, -1)) * (1.0 / 256)

        # mask
        mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        mask = mask.unsqueeze(1).expand(B, T, T)  # 扩展维度以便与qk维度匹配 (B, T, T)
        c = c.masked_fill(mask, float('-inf'))

        # c = c.masked_fill(self.copy_mask[:T,:T] == 0, 0)
        # print("C shape :{}.".format(c.shape))
        # c = c @ F.one_hot(idx, num_classes = self.config.vocab_size).float()

        x = x * self.time_out[:, :T, :]  # reduce confidence of early tokens
        # x = self.head(x) + c #[Batch,seq,768]

        return x

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.GPT(log_seqs)  # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs)  # [128, 200, 128]
        print("pos logits shape:{}".format(pos_logits.shape))
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        # neg_logits = (log_feats * neg_embs)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.GPT(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)