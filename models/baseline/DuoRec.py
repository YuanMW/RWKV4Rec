import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DuoRec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(DuoRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.device = args.device

        # Model parameters
        self.n_layers = args.num_blocks
        self.n_heads = args.num_heads
        self.hidden_size = args.hidden_units
        self.hidden_dropout_prob = args.dropout_rate
        self.attn_dropout_prob = args.attention_probs_dropout_prob

        # Contrastive learning parameters (get from args with default values)
        self.tau = getattr(args, 'tau', 1.0)
        self.ssl_mode = getattr(args, 'contrast', 'us_x')
        self.sim = getattr(args, 'sim', 'dot')
        self.lmd_sem = getattr(args, 'lmd_sem', 0.1)
        self.lmd = getattr(args, 'lmd', 0.1)

        # Item and position embeddings
        self.item_emb = nn.Embedding(item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(args) for _ in range(args.num_blocks)
        ])
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        # Contrastive learning loss function
        self.aug_nce_fct = nn.CrossEntropyLoss()

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def log2feats(self, log_seqs):
        """Sequence to feature conversion"""
        # Item embeddings
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.device))
        seqs *= self.item_emb.embedding_dim ** 0.5

        # Position embeddings
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.device))
        seqs = self.emb_dropout(seqs)

        # Process padding mask
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.device)
        seqs *= ~timeline_mask.unsqueeze(-1)

        # Attention mask (causal mask)
        seq_len = log_seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))

        # Pass through transformer encoder layers
        for layer in self.encoder_layers:
            seqs = layer(seqs, attention_mask, timeline_mask)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        """Training forward pass"""
        log_feats = self.log2feats(log_seqs)

        # Get positive and negative sample embeddings
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.device))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.device))

        # Compute logits
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        """Inference forward pass"""
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]  # Use last position
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.device))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits

    # Contrastive learning related methods
    def augmented_forward(self, seq):
        """Forward pass for augmented sequences"""
        return self.log2feats(seq)

    def info_nce(self, z_i, z_j, batch_size):
        """
        Contrastive learning InfoNCE loss calculation
        """
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        if self.sim == 'cos':
            sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.tau
        elif self.sim == 'dot':
            sim_matrix = torch.mm(z, z.T) / self.tau

        sim_i_j = torch.diag(sim_matrix, batch_size)
        sim_j_i = torch.diag(sim_matrix, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # Create negative sample mask
        mask = self.mask_correlated_samples(batch_size)
        negative_samples = sim_matrix[mask].reshape(N, -1)

        labels = torch.zeros(N, device=z_i.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def mask_correlated_samples(self, batch_size):
        """Create mask for correlated samples"""
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool, device=self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def calculate_contrastive_loss(self, original_seq, aug_seq=None, sem_aug_seq=None):
        """
        Calculate contrastive learning loss - adapted for us_x mode
        """
        batch_size = original_seq.shape[0]
        total_contrast_loss = 0.0

        # Get features of original sequence
        original_feats = self.log2feats(original_seq)
        original_output = original_feats[:, -1, :]  # Use last position

        # us_x mode: contrast between augmented view and semantic augmented view
        if self.ssl_mode == 'us_x' and aug_seq is not None and sem_aug_seq is not None:
            # Get features of augmented sequence
            aug_feats = self.augmented_forward(aug_seq)
            aug_output = aug_feats[:, -1, :]

            # Get features of semantic augmented sequence
            sem_aug_feats = self.augmented_forward(sem_aug_seq)
            sem_aug_output = sem_aug_feats[:, -1, :]

            # Calculate contrastive loss between augmented view and semantic augmented view
            nce_logits, nce_labels = self.info_nce(aug_output, sem_aug_output, batch_size)
            contrast_loss = F.cross_entropy(nce_logits, nce_labels)

            # Apply weight
            total_contrast_loss = self.lmd_sem * contrast_loss

        return total_contrast_loss

    def apply_augmentation(self, seq, aug_type='random_mask', mask_ratio=0.2):
        """
        Apply data augmentation
        """
        if aug_type == 'random_mask':
            return self.random_mask(seq, mask_ratio)
        elif aug_type == 'crop':
            return self.random_crop(seq)
        elif aug_type == 'reorder':
            return self.random_reorder(seq)
        else:
            return seq

    def random_mask(self, seq, mask_ratio=0.2):
        """Random mask augmentation"""
        batch_size, seq_len = seq.shape
        mask = torch.rand(seq.shape, device=seq.device) < mask_ratio
        masked_seq = seq.clone()
        masked_seq[mask] = 0
        return masked_seq

    def random_crop(self, seq, crop_ratio=0.8):
        """Random crop augmentation"""
        batch_size, seq_len = seq.shape
        crop_len = int(seq_len * crop_ratio)
        start_idx = torch.randint(0, seq_len - crop_len + 1, (batch_size,), device=seq.device)

        cropped_seq = torch.zeros_like(seq)
        for i in range(batch_size):
            end_idx = start_idx[i] + crop_len
            cropped_seq[i, :crop_len] = seq[i, start_idx[i]:end_idx]

        return cropped_seq

    def random_reorder(self, seq, reorder_ratio=0.2):
        """Random reorder augmentation"""
        batch_size, seq_len = seq.shape
        reordered_seq = seq.clone()

        for i in range(batch_size):
            reorder_num = max(1, int(seq_len * reorder_ratio))
            reorder_pos = torch.randperm(seq_len)[:reorder_num]

            values = seq[i, reorder_pos]
            shuffled_values = values[torch.randperm(reorder_num)]
            reordered_seq[i, reorder_pos] = shuffled_values

        return reordered_seq

    def create_semantic_augmentation(self, seq):
        """
        Create semantic augmented sequence
        """
        return self.random_mask(seq, mask_ratio=0.3)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            args.hidden_units,
            args.num_heads,
            dropout=args.dropout_rate,
            batch_first=True  # Use batch_first to simplify operations
        )
        self.linear1 = nn.Linear(args.hidden_units, args.hidden_units * 4)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.linear2 = nn.Linear(args.hidden_units * 4, args.hidden_units)
        self.norm1 = nn.LayerNorm(args.hidden_units, eps=1e-12)
        self.norm2 = nn.LayerNorm(args.hidden_units, eps=1e-12)
        self.dropout1 = nn.Dropout(args.dropout_rate)
        self.dropout2 = nn.Dropout(args.dropout_rate)

    def forward(self, src, src_mask=None, key_padding_mask=None):
        # Self-attention layer
        src2 = self.norm1(src)

        # Use batch_first=True to avoid transpose operations
        src2, _ = self.self_attn(
            src2, src2, src2,
            attn_mask=src_mask,
            key_padding_mask=key_padding_mask
        )
        src = src + self.dropout1(src2)

        # Feed-forward layer
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src