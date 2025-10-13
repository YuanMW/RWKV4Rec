import torch
from torch import nn

class GRU4RecModel(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(GRU4RecModel, self).__init__()

        # 从args获取参数
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.hidden_units = args.hidden_units
        self.num_blocks = args.num_blocks  # 作为GRU层数
        self.dropout_rate = args.dropout_rate

        # 定义模型层
        self.item_emb = nn.Embedding(item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        # GRU层
        self.gru_layers = nn.GRU(
            input_size=args.hidden_units,
            hidden_size=args.hidden_units,
            num_layers=args.num_blocks,
            batch_first=True,
        )


    def log2feats(self, log_seqs):
        # 先将输入转换为LongTensor
        log_seqs_tensor = torch.LongTensor(log_seqs).to(self.dev)

        # 获取物品嵌入
        seqs = self.item_emb(log_seqs_tensor)
        seqs *= self.item_emb.embedding_dim ** 0.5

        # 创建位置编码
        positions = torch.arange(log_seqs_tensor.size(1), device=self.dev).unsqueeze(0).expand_as(log_seqs_tensor)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        # 处理padding mask
        timeline_mask = (log_seqs_tensor == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        # 通过GRU层
        gru_output, _ = self.gru_layers(seqs)
        return gru_output

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        # 获取正负样本嵌入
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        # 计算logits (保持GRU4Rec原始的点积方式)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]  # 取最后一个时间步

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits