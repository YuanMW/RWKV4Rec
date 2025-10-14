import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class FEARec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(FEARec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.device = args.device

        # Model parameters
        self.n_layers = args.num_blocks
        self.n_heads = args.num_heads
        self.hidden_size = args.hidden_units
        self.inner_size = args.hidden_units * 4
        self.hidden_dropout_prob = args.dropout_rate
        self.attn_dropout_prob = args.attention_probs_dropout_prob
        self.hidden_act = args.hidden_act
        self.layer_norm_eps = 1e-12

        # Contrastive learning parameters
        self.lmd = getattr(args, 'lmd', 0.1)
        self.lmd_sem = getattr(args, 'lmd_sem', 0.1)
        self.tau = getattr(args, 'tau', 1.0)
        self.ssl_mode = getattr(args, 'contrast', 'us_x')
        self.sim = getattr(args, 'sim', 'dot')

        # Frequency domain enhancement parameters
        self.fredom = getattr(args, 'fredom', True)
        self.fredom_type = getattr(args, 'fredom_type', 'us_x')
        self.global_ratio = getattr(args, 'global_ratio', 0.6)
        self.dual_domain = getattr(args, 'dual_domain', True)
        self.spatial_ratio = getattr(args, 'spatial_ratio', 0.1)
        self.std = getattr(args, 'std', True)

        self.initializer_range = 0.02
        self.max_seq_length = args.maxlen

        # Define layers
        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_seq_length, self.hidden_size)

        # FEA encoder
        self.encoder = nn.ModuleList([
            FEARecBlock(args, i, self.global_ratio, self.dual_domain, self.spatial_ratio, self.std)
            for i in range(args.num_blocks)
        ])

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate causal attention mask"""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Causal attention mask (left to right)
        max_len = attention_mask.size(-1)
        subsequent_mask = torch.triu(torch.ones((1, max_len, max_len), device=item_seq.device), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).long()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def log2feats(self, log_seqs):
        """Sequence to feature conversion"""
        item_seq = torch.LongTensor(log_seqs).to(self.device)

        # Position embedding
        seq_len = item_seq.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.pos_emb(position_ids)

        # Item embedding
        item_embedding = self.item_emb(item_seq)
        input_emb = item_embedding + position_embedding

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # Attention mask
        attention_mask = self.get_attention_mask(item_seq)

        # Encoder processing
        for layer in self.encoder:
            input_emb = layer(input_emb, attention_mask)

        return input_emb

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        """Training forward pass"""
        log_feats = self.log2feats(log_seqs)

        # Get positive and negative sample embeddings
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.device))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.device))

        # Use hidden state of the last position
        final_output = log_feats[:, -1, :]  # [batch_size, hidden_size]

        # Compute logits
        pos_logits = (final_output.unsqueeze(1) * pos_embs).sum(dim=-1)
        neg_logits = (final_output.unsqueeze(1) * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        """Inference forward pass"""
        log_feats = self.log2feats(log_seqs)
        final_output = log_feats[:, -1, :]  # Last position

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.device))
        logits = torch.matmul(item_embs, final_output.unsqueeze(-1)).squeeze(-1)

        return logits

    # Contrastive learning and frequency domain enhancement methods
    def augmented_forward(self, seq):
        """Forward pass for augmented sequences"""
        return self.log2feats(seq)

    def info_nce(self, z_i, z_j, batch_size):
        """
        Contrastive learning InfoNCE loss calculation
        """
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        # Use dot product similarity with temperature parameter tau=1
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
        original_output = original_feats[:, -1, :]

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

            # Frequency domain enhancement loss (if enabled)
            if self.fredom and self.fredom_type == 'us_x':
                # Frequency domain transformation
                aug_output_f = torch.fft.rfft(aug_output, dim=-1, norm='ortho')
                sem_aug_output_f = torch.fft.rfft(sem_aug_output, dim=-1, norm='ortho')
                freq_loss = 0.1 * torch.abs(aug_output_f - sem_aug_output_f).flatten().mean()
                contrast_loss += freq_loss

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


class FEARecBlock(nn.Module):
    def __init__(self, args, layer_idx, global_ratio=0.6, dual_domain=True, spatial_ratio=0.1, std=True):
        super().__init__()
        self.attention = FEARecAttention(args, layer_idx, global_ratio, dual_domain, spatial_ratio, std)
        self.feed_forward = FeedForward(args)
        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(self, x, attention_mask):
        # Self-attention
        attn_output = self.attention(x, attention_mask)
        x = x + self.dropout(attn_output)

        # Feed forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)

        return x


class FEARecAttention(nn.Module):
    def __init__(self, args, layer_idx, global_ratio=0.6, dual_domain=True, spatial_ratio=0.1, std=True):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = args.hidden_units
        self.num_heads = args.num_heads
        self.head_size = args.hidden_units // args.num_heads

        # Frequency domain parameters
        self.global_ratio = global_ratio
        self.dual_domain = dual_domain
        self.spatial_ratio = spatial_ratio
        self.std = std
        self.total_layers = args.num_blocks
        self.max_freq = args.maxlen // 2 + 1

        # Calculate frequency window
        if self.global_ratio > (1 / self.total_layers):
            window_size = int(self.max_freq * self.global_ratio)
            slide_step = int((self.max_freq - window_size) / (self.total_layers - 1))
            self.freq_start = max(0, int(self.max_freq - window_size - layer_idx * slide_step))
        else:
            window_size = int(self.max_freq / self.total_layers)
            self.freq_start = layer_idx * window_size

        self.freq_end = min(self.freq_start + window_size, self.max_freq)
        self.freq_indices = list(range(self.freq_start, self.freq_end))

        # Projections
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)

        # Output
        self.output = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(self, x, attention_mask):
        batch_size, seq_len, _ = x.shape

        # Project inputs
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        # Frequency domain processing
        q_fft = torch.fft.rfft(q, dim=-1, norm='ortho')
        k_fft = torch.fft.rfft(k, dim=-1, norm='ortho')

        # Apply frequency filter
        q_fft_filtered = torch.zeros_like(q_fft)
        k_fft_filtered = torch.zeros_like(k_fft)

        valid_indices = [i for i in self.freq_indices if i < q_fft.size(-1)]
        q_fft_filtered[..., valid_indices] = q_fft[..., valid_indices]
        k_fft_filtered[..., valid_indices] = k_fft[..., valid_indices]

        # Frequency correlation
        corr = torch.fft.irfft(q_fft_filtered * torch.conj(k_fft_filtered), n=seq_len, dim=-1, norm='ortho')

        # Time delay aggregation
        v_reshaped = v.transpose(2, 3)
        agg_v = self._time_delay_aggregation(v_reshaped, corr).transpose(2, 3)
        freq_output = agg_v.reshape(batch_size, seq_len, self.hidden_size)

        if self.dual_domain:
            # Spatial attention
            attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)

            # Apply attention mask
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask

            attn_probs = F.softmax(attn_scores, dim=-1)
            spatial_output = (attn_probs @ v).transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)

            # Combine outputs with spatial ratio
            output = (1 - self.spatial_ratio) * freq_output + self.spatial_ratio * spatial_output
        else:
            output = freq_output

        # Standard deviation normalization if enabled
        if self.std:
            output = self._std_normalization(output)

        # Final projection
        output = self.output(output)
        return self.layer_norm(x + self.dropout(output))

    def _time_delay_aggregation(self, values, corr):
        batch, heads, channels, seq_len = values.shape
        top_k = min(int(10 * math.log(seq_len)), seq_len)

        mean_corr = torch.mean(corr, dim=(1, 2))
        top_k = min(top_k, seq_len)

        weights, top_indices = torch.topk(mean_corr, top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        agg_values = torch.zeros_like(values)
        base_indices = torch.arange(seq_len, device=values.device).view(1, 1, 1, seq_len)

        for i in range(top_k):
            shifts = top_indices[:, i].view(batch, 1, 1, 1)
            shifted_indices = (base_indices + shifts) % seq_len

            shifted_values = torch.gather(
                values,
                -1,
                shifted_indices.expand(batch, heads, channels, seq_len)
            )
            agg_values += shifted_values * weights[:, i].view(batch, 1, 1, 1)

        return agg_values

    def _std_normalization(self, x):
        """Standard deviation normalization"""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + 1e-8)


class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear1 = nn.Linear(args.hidden_units, args.hidden_units * 4)
        self.linear2 = nn.Linear(args.hidden_units * 4, args.hidden_units)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.layer_norm = nn.LayerNorm(args.hidden_units)

    def forward(self, x):
        out = self.linear2(F.gelu(self.linear1(x)))
        return self.layer_norm(x + self.dropout(out))