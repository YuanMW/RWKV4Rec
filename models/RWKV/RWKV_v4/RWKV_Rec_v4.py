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

def RWKV_Init(model, item_num, args):

    for mm in model.modules():
        if not isinstance(mm, (nn.Linear, nn.Embedding)):
            continue

        with torch.no_grad():
            ww = mm.weight
            name = "[unknown weight]"
            for name, parameter in model.named_parameters():  # find the name of the weight
                if id(ww) == id(parameter):
                    break

            shape = ww.shape
            gain = 1.0
            scale = 1.0  # extra scale for gain

            if isinstance(mm, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == item_num and shape[1] == args.hidden_units:  # token emb?
                    scale = 1e-4
                else:
                    scale = 0

            if isinstance(mm, nn.Linear):
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == item_num and shape[1] == args.hidden_units:  # final projection?
                    scale = 0.5

            if hasattr(mm, "scale_init"):
                scale = mm.scale_init

            gain *= scale
            if scale == -999:
                nn.init.eye_(ww)
            elif gain == 0:
                # zero init is great for some RWKV matrices
                nn.init.zeros_(ww)
            elif gain > 0:
                nn.init.orthogonal_(ww, gain=gain)
            else:
                nn.init.normal_(ww, mean=0.0, std=-scale)


class RWKV_TimeMix(nn.Module):
    def __init__(self, args, layer_id, n_layer=6):
        super().__init__()
        self.layer_id = layer_id
        self.maxlen = args.maxlen
        self.hidden_units = args.hidden_units
        self.n_layer = n_layer

        self.attn_sz = args.hidden_units

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = (layer_id / (self.n_layer - 1))  # 0 to 1
            ratio_1_to_almost0 = (1.0 - (layer_id / self.n_layer))  # 1 to ~0

            # fancy time_decay
            decay_speed = torch.ones(self.maxlen)
            curve = torch.tensor([-(self.maxlen - 1 - i) for i in range(self.maxlen)])
            for h in range(self.maxlen):
                decay_speed[h] = -5 + 8 * (h / (self.maxlen - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            ww = torch.exp(curve * decay_speed)  # Add unsqueeze to match dimensions
            self.time_w = nn.Parameter(ww)

            # fancy time_first
            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.attn_sz)]) * 0.5)
            self.time_first = nn.Parameter(torch.ones(self.attn_sz) * math.log(0.3) + zigzag)

            # fancy time_mix
            x = torch.ones(1, 1, self.hidden_units)
            for i in range(self.hidden_units):
                x[0, 0, i] = i / self.hidden_units
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(self.hidden_units, self.attn_sz, bias=False)
        self.value = nn.Linear(self.hidden_units, self.attn_sz, bias=False)
        self.receptance = nn.Linear(self.hidden_units, self.attn_sz, bias=False)

        self.output = nn.Linear(self.attn_sz, self.hidden_units, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):

        B, T, C = x.size()  # x = (Batch,Time,Channel)(128,200,64)
        TT = self.maxlen # 200

        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x) # [128, 200, 50]n
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Compute keys, values, and receptances
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)

        # Clamp and exponentiate key values
        k = torch.clamp(k, max=30, min=-60)  # clamp extreme values. e^30 = 10^13
        k = torch.exp(k)

        # Compute cumulative sums of keys for normalization
        sum_k = torch.cumsum(k, dim=1)

        # Compute kv pairs
        kv = (k * v).view(B, T, 1, self.attn_sz)

        # Create a circulant matrix from time decay
        w = F.pad(self.time_w.view(1, self.maxlen), (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT - 1:]  # w is now a circulant matrix
        w = w[:, :T, :T]

        # Compute wkv using einsum
        wkv = (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)


        # Normalize and apply sigmoid to receptances
        rwkv = torch.sigmoid(r) * wkv / sum_k

        # Final output layer
        rwkv = self.output(rwkv)

        return rwkv


class RWKV_ChannelMix(nn.Module):
    def __init__(self, args, layer_id, n_layer=6):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = (1.0 - (layer_id / self.n_layer))  # 1 to ~0

            x = torch.ones(1, 1, args.hidden_units)
            for i in range(args.hidden_units):
                x[0, 0, i] = i / args.hidden_units

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = 4 * args.hidden_units
        self.key = nn.Linear(args.hidden_units, hidden_sz, bias=False)
        self.receptance = nn.Linear(args.hidden_units, args.hidden_units, bias=False)
        self.value = nn.Linear(hidden_sz, args.hidden_units, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv

########################################################################################################
# The GPT Model with our blocks
########################################################################################################

class Block(nn.Module):
    def __init__(self, args, layer_id, model_type):
        super().__init__()
        # self.config = config
        self.layer_id = layer_id
        self.model_type = model_type

        self.ln1 = nn.LayerNorm(args.hidden_units)
        self.ln2 = nn.LayerNorm(args.hidden_units)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.hidden_units)

        if self.layer_id == 0 and self.model_type == 'RWKV-ffnPre':
            self.ffnPre = RWKV_ChannelMix(args, layer_id=0)
        else:
            self.att = RWKV_TimeMix(args, layer_id)

        self.ffn = RWKV_ChannelMix(args, layer_id)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)
        if self.layer_id == 0 and self.model_type == 'RWKV-ffnPre':
            x = x + self.ffnPre(self.ln1(x))  # better in some cases
        else:
            x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args,
                 n_layer=6,
                 model_type="RWKV-ffnPre"
                 ):
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)

        self.maxlen = args.maxlen
        self.n_layer = n_layer
        self.model_type = model_type

        self.step = 0
        # self.config = config

        self.blocks = nn.Sequential(*[Block(args, i, self.model_type) for i in range(self.n_layer)])

        self.ln_out = nn.LayerNorm(args.hidden_units)
        self.time_out = nn.Parameter(torch.ones(1, self.maxlen, 1))
        self.head = nn.Linear(args.hidden_units, self.item_num + 1, bias=False)

        self.head_q = nn.Linear(args.hidden_units, 256, bias=False)
        self.head_q.scale_init = 0
        self.head_k = nn.Linear(args.hidden_units, 256, bias=False)
        self.head_k.scale_init = 0.1
        self.register_buffer("copy_mask", torch.tril(
            torch.ones(self.maxlen, self.maxlen)))


        if self.model_type == 'RWKV':
            RWKV_Init(self, self.item_num, args)
        else:
            self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def GPT(self, log_seqs, targets=None):
        x = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # [128, 200, 100]

        # self.step += 1
        B, T, C = x.size()
        assert T <= self.maxlen, "Cannot forward, because len(input) > model ctx_len."

        # x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)

        q = self.head_q(x)[:, :T, :]
        k = self.head_k(x)[:, :T, :]
        c = (q @ k.transpose(-2, -1)) * (1.0 / 256)
        # mask
        mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        mask = mask.unsqueeze(1).expand(B, T, T)  # 扩展维度以便与qk维度匹配 (B, T, T)
        c = c.masked_fill(mask, float('-inf'))

        x = x * self.time_out[:, :T, :]

        return x

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.GPT(log_seqs)  # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs)
        print("pos logits shape:{}".format(pos_logits.shape))
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

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