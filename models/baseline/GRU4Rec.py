import torch
from torch import nn

class GRU4RecModel(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(GRU4RecModel, self).__init__()

        # Get parameters from args
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.hidden_units = args.hidden_units
        self.num_blocks = args.num_blocks  # Used as number of GRU layers
        self.dropout_rate = args.dropout_rate

        # Define model layers
        self.item_emb = nn.Embedding(item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        # GRU layers
        self.gru_layers = nn.GRU(
            input_size=args.hidden_units,
            hidden_size=args.hidden_units,
            num_layers=args.num_blocks,
            batch_first=True,
        )

    def log2feats(self, log_seqs):
        # First convert input to LongTensor
        log_seqs_tensor = torch.LongTensor(log_seqs).to(self.dev)

        # Get item embeddings
        seqs = self.item_emb(log_seqs_tensor)
        seqs *= self.item_emb.embedding_dim ** 0.5

        # Create position encodings
        positions = torch.arange(log_seqs_tensor.size(1), device=self.dev).unsqueeze(0).expand_as(log_seqs_tensor)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        # Process padding mask
        timeline_mask = (log_seqs_tensor == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        # Pass through GRU layers
        gru_output, _ = self.gru_layers(seqs)
        return gru_output

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        # Get positive and negative sample embeddings
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        # Compute logits (maintain GRU4Rec original dot product approach)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]  # Take the last time step

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits