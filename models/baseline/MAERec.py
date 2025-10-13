import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix


class MAERec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(MAERec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.args = args
        self.dev = args.device

        # Graph Components
        self.encoder = Encoder(item_num, args)
        self.decoder = Decoder(args)
        self.masker = RandomMaskSubgraphs()
        self.sampler = LocalGraph()

        # Sequential Components (与SASRec保持一致)
        self.item_emb = nn.Embedding(item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        # Transformer layers
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(args.num_blocks):
            new_attn_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate, batch_first=True
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    def build_graph_from_sequences(self, user_train):
        """从用户序列构建物品-物品图"""
        rows, cols = [], []
        data = []

        max_neighbors = getattr(self.args, 'max_neighbors', 10)

        for u in user_train:
            items = user_train[u]
            # 为每个用户的序列创建物品对
            for i in range(len(items)):
                # 连接序列中的后续物品
                neighbors = range(i + 1, min(i + 1 + max_neighbors, len(items)))
                for j in neighbors:
                    if items[i] <= self.item_num and items[j] <= self.item_num:
                        rows.append(items[i])
                        cols.append(items[j])
                        data.append(1.0)
                        # 双向连接
                        rows.append(items[j])
                        cols.append(items[i])
                        data.append(1.0)

        if not rows:  # 如果没有边，创建自环
            rows = list(range(1, min(self.item_num + 1, 100)))
            cols = list(range(1, min(self.item_num + 1, 100)))
            data = [1.0] * len(rows)

        # 创建稀疏矩阵
        mat = coo_matrix((data, (rows, cols)),
                         shape=(self.item_num + 1, self.item_num + 1))

        # 构建图结构
        self.ii_adj = self._make_torch_adj(mat)
        self.ii_adj_all_one = self._make_all_one_adj(self.ii_adj)

    def _make_torch_adj(self, mat):
        # 添加自环
        mat = (mat + sp.eye(mat.shape[0]))
        mat = (mat != 0) * 1.0
        mat = self._normalize(mat)
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(self.dev)

    def _make_all_one_adj(self, adj):
        idxs = adj._indices()
        vals = t.ones_like(adj._values())
        shape = adj.shape
        return t.sparse.FloatTensor(idxs, vals, shape).to(self.dev)

    def _normalize(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def log2feats(self, log_seqs):
        # 确保输入是LongTensor
        if isinstance(log_seqs, np.ndarray):
            log_seqs = t.LongTensor(log_seqs).to(self.dev)

        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5

        # 位置编码
        positions = t.arange(log_seqs.size(1), dtype=t.long, device=self.dev)
        positions = positions.unsqueeze(0).expand_as(log_seqs)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        # 掩码处理
        timeline_mask = (log_seqs == 0)
        seqs = seqs * (~timeline_mask.unsqueeze(-1))

        # 因果注意力掩码
        tl = seqs.shape[1]
        attention_mask = ~t.tril(t.ones((tl, tl), dtype=t.bool, device=self.dev))

        # Transformer层
        for i in range(len(self.attention_layers)):
            # 自注意力层
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, Q, Q,
                attn_mask=attention_mask,
                need_weights=False
            )
            seqs = Q + mha_outputs

            # 前馈层
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs = seqs * (~timeline_mask.unsqueeze(-1))

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        # 确保图结构已构建
        if not hasattr(self, 'ii_adj'):
            raise RuntimeError("Please call build_graph_from_sequences() before training")

        # 转换输入为张量
        if isinstance(log_seqs, np.ndarray):
            log_seqs = t.LongTensor(log_seqs).to(self.dev)
        if isinstance(pos_seqs, np.ndarray):
            pos_seqs = t.LongTensor(pos_seqs).to(self.dev)
        if isinstance(neg_seqs, np.ndarray):
            neg_seqs = t.LongTensor(neg_seqs).to(self.dev)

        # Graph Encoding (仅在训练时使用掩码)
        if self.training and hasattr(self.args, 'mask_depth') and self.args.mask_depth > 0:
            sample_scr, candidates = self.sampler(
                self.ii_adj_all_one, self.encoder.get_ego_embeds(), self.args
            )
            masked_adj, masked_edg = self.masker(self.ii_adj, candidates, self.args)
            item_emb, item_emb_his = self.encoder(masked_adj)
        else:
            item_emb, item_emb_his = self.encoder(self.ii_adj)

        # Sequential Processing
        log_feats = self.log2feats(log_seqs)

        # 计算正负样本分数
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        if not hasattr(self, 'ii_adj'):
            raise RuntimeError("Please call build_graph_from_sequences() before prediction")

        # 转换输入
        if isinstance(log_seqs, np.ndarray):
            log_seqs = t.LongTensor(log_seqs).to(self.dev)
        if isinstance(item_indices, np.ndarray):
            item_indices = t.LongTensor(item_indices).to(self.dev)

        # Graph Encoding (推理时不使用掩码)
        item_emb, _ = self.encoder(self.ii_adj)

        # Sequential Processing
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]  # 使用最后一个位置进行预测

        # 计算所有候选物品的分数
        item_embs = self.item_emb(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class Encoder(nn.Module):
    def __init__(self, item_num, args):
        super(Encoder, self).__init__()
        self.item_num = item_num
        self.args = args
        self.num_gcn_layers = 2  # 固定GCN层数
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.gcn_layers = nn.ModuleList([GCNLayer() for _ in range(self.num_gcn_layers)])

        # 初始化嵌入权重
        nn.init.xavier_uniform_(self.item_emb.weight)

    def get_ego_embeds(self):
        return self.item_emb.weight

    def forward(self, encoder_adj):
        embeds = [self.item_emb.weight]
        for gcn in self.gcn_layers:
            embeds.append(gcn(encoder_adj, embeds[-1]))
        return sum(embeds), embeds


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.num_gcn_layers = 2
        self.MLP = nn.Sequential(
            nn.Linear(args.hidden_units * self.num_gcn_layers ** 2,
                      args.hidden_units * self.num_gcn_layers, bias=True),
            nn.ReLU(),
            nn.Linear(args.hidden_units * self.num_gcn_layers, args.hidden_units, bias=True),
            nn.ReLU(),
            nn.Linear(args.hidden_units, 1, bias=True),
            nn.Sigmoid()
        )
        self.apply(self.init_weights)

    def forward(self, embeds, pos, neg):
        pos_emb, neg_emb = [], []
        for i in range(self.num_gcn_layers):
            for j in range(self.num_gcn_layers):
                pos_emb.append(embeds[i][pos[:, 0]] * embeds[j][pos[:, 1]])
                neg_emb.append(embeds[i][neg[:, :, 0]] * embeds[j][neg[:, :, 1]])

        pos_emb = t.cat(pos_emb, -1)
        neg_emb = t.cat(neg_emb, -1)

        pos_scr = t.squeeze(self.MLP(pos_emb))
        neg_scr = t.squeeze(self.MLP(neg_emb))

        # 使用softmax计算损失
        logits = t.cat([pos_scr.unsqueeze(1), neg_scr], dim=1)
        labels = t.zeros(logits.shape[0], dtype=t.long).to(logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)


def sparse_dropout(x, keep_prob):
    """对稀疏张量进行dropout操作"""
    if keep_prob == 1.0:
        return x
    msk = (t.rand(x._values().size()) < keep_prob).to(x.device)
    idx = x._indices()[:, msk]
    val = x._values()[msk] / keep_prob
    return t.sparse.FloatTensor(idx, val, x.shape).to(x.device)


class LocalGraph(nn.Module):
    def __init__(self):
        super(LocalGraph, self).__init__()

    def make_noise(self, scores):
        noise = t.rand(scores.shape, device=scores.device)
        noise = -t.log(-t.log(noise))
        return scores + noise

    def forward(self, adj, embeds, args, foo=None):
        order = t.sparse.sum(adj, dim=-1).to_dense().view([-1, 1])
        fstEmbeds = t.spmm(adj, embeds) - embeds
        fstNum = order

        emb = [fstEmbeds]
        num = [fstNum]

        for i in range(getattr(args, 'mask_depth', 3)):
            adj = sparse_dropout(adj, getattr(args, 'path_prob', 0.5) ** (i + 1))
            emb.append((t.spmm(adj, emb[-1]) - emb[-1]) - order * emb[-1])
            num.append((t.spmm(adj, num[-1]) - num[-1]) - order)
            order = t.sparse.sum(adj, dim=-1).to_dense().view([-1, 1])

        subgraphEmbeds = sum(emb) / (sum(num) + 1e-8)
        subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)

        embeds = F.normalize(embeds, p=2)
        scores = t.sum(subgraphEmbeds * embeds, dim=-1)
        scores = self.make_noise(scores)

        _, candidates = t.topk(scores, min(getattr(args, 'num_mask_cand', 100), scores.shape[0]))

        return scores, candidates


class RandomMaskSubgraphs(nn.Module):
    def __init__(self):
        super(RandomMaskSubgraphs, self).__init__()

    def normalize(self, adj):
        degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

    def forward(self, adj, seeds, args):
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        masked_rows = []
        masked_cols = []

        for i in range(getattr(args, 'mask_depth', 3)):
            curSeeds = seeds if i == 0 else nxtSeeds
            nxtSeeds = []
            idct = None

            for seed in curSeeds:
                rowIdct = (rows == seed)
                colIdct = (cols == seed)
                if idct is None:
                    idct = t.logical_or(rowIdct, colIdct)
                else:
                    idct = t.logical_or(idct, t.logical_or(rowIdct, colIdct))

            if idct is None:
                break

            nxtRows = rows[idct]
            nxtCols = cols[idct]
            masked_rows.extend(nxtRows.cpu().tolist())
            masked_cols.extend(nxtCols.cpu().tolist())

            # 更新剩余的边
            remaining_mask = ~idct
            rows = rows[remaining_mask]
            cols = cols[remaining_mask]

            nxtSeeds = t.cat([nxtRows, nxtCols])
            if len(nxtSeeds) > 0 and i != getattr(args, 'mask_depth', 3) - 1:
                nxtSeeds = t.unique(nxtSeeds)
                cand = t.randperm(nxtSeeds.shape[0])
                keep_num = int(nxtSeeds.shape[0] * getattr(args, 'path_prob', 0.5) ** (i + 1))
                nxtSeeds = nxtSeeds[cand[:keep_num]]

        # 构建掩码后的邻接矩阵
        if len(rows) > 0:
            encoder_adj = self.normalize(
                t.sparse.FloatTensor(
                    t.stack([rows, cols], dim=0),
                    t.ones_like(rows, dtype=t.float32),
                    adj.shape
                ).to(adj.device)
            )
        else:
            # 如果所有边都被掩码，返回原始邻接矩阵
            encoder_adj = adj

        return encoder_adj, (masked_rows, masked_cols)