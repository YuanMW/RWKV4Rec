import torch
import torch.nn as nn
import numpy as np
import math


class BSARec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(BSARec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # Embedding layers
        self.item_emb = nn.Embedding(item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        # BSARec encoder layers
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(args.num_blocks):
            new_attn_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = BSARecLayer(args)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def log2feats(self, log_seqs):
        # Ensure input is LongTensor
        if isinstance(log_seqs, np.ndarray):
            log_seqs = torch.LongTensor(log_seqs).to(self.dev)

        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5  # Scale

        # Position embeddings
        positions = torch.arange(log_seqs.size(1), dtype=torch.long, device=self.dev)
        positions = positions.unsqueeze(0).expand_as(log_seqs)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        # Create masks
        timeline_mask = (log_seqs == 0)
        seqs = seqs * (~timeline_mask.unsqueeze(-1))

        # Causal attention mask
        seq_len = log_seqs.size(1)
        attention_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.dev))

        # Pass through BSARec layers
        for i in range(len(self.attention_layers)):
            # BSARec layer (frequency + attention)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](Q, attention_mask)
            seqs = Q + mha_outputs

            # Feed forward layer
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs = seqs * (~timeline_mask.unsqueeze(-1))

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        # Convert inputs to tensors if they are numpy arrays
        if isinstance(log_seqs, np.ndarray):
            log_seqs = torch.LongTensor(log_seqs).to(self.dev)
        if isinstance(pos_seqs, np.ndarray):
            pos_seqs = torch.LongTensor(pos_seqs).to(self.dev)
        if isinstance(neg_seqs, np.ndarray):
            neg_seqs = torch.LongTensor(neg_seqs).to(self.dev)

        log_feats = self.log2feats(log_seqs)

        # Get positive and negative item embeddings
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        # Compute logits using dot product
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        # Convert inputs to tensors if they are numpy arrays
        if isinstance(log_seqs, np.ndarray):
            log_seqs = torch.LongTensor(log_seqs).to(self.dev)
        if isinstance(item_indices, np.ndarray):
            item_indices = torch.LongTensor(item_indices).to(self.dev)

        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]  # Use last position for prediction

        item_embs = self.item_emb(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits


class BSARecLayer(nn.Module):
    def __init__(self, args):
        super(BSARecLayer, self).__init__()
        self.args = args
        self.filter_layer = FrequencyLayer(args)
        self.attention_layer = MultiHeadAttention(args)
        self.alpha = getattr(args, 'alpha', 0.5)  # Default alpha value

    def forward(self, input_tensor, attention_mask):
        # Frequency domain processing (DSP)
        dsp = self.filter_layer(input_tensor)

        # Attention processing (GSP)
        gsp = self.attention_layer(input_tensor, attention_mask)

        # Dual-stream fusion
        hidden_states = self.alpha * dsp + (1 - self.alpha) * gsp
        return hidden_states


class FrequencyLayer(nn.Module):
    def __init__(self, args):
        super(FrequencyLayer, self).__init__()
        self.hidden_units = args.hidden_units
        self.maxlen = args.maxlen
        self.out_dropout = nn.Dropout(p=args.dropout_rate)
        self.LayerNorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        # Frequency parameters
        self.c = args.maxlen // 2 + 1  # Frequency cutoff point
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, args.hidden_units))

    def forward(self, input_tensor):
        batch_size, seq_len, hidden_dim = input_tensor.shape

        # Apply FFT along the sequence dimension
        x_fft = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        # Create low-pass filter
        low_pass = x_fft.clone()
        low_pass[:, self.c:, :] = 0  # Zero out high frequencies

        # Inverse FFT to get low-pass component
        low_pass_time = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')

        # High-pass component is the residual
        high_pass_time = input_tensor - low_pass_time

        # Frequency enhancement: combine with learnable parameter
        enhanced_sequence = low_pass_time + (self.sqrt_beta ** 2) * high_pass_time

        # Residual connection and layer normalization
        hidden_states = self.out_dropout(enhanced_sequence)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.hidden_units = args.hidden_units
        self.num_heads = args.num_heads
        self.head_dim = args.hidden_units // args.num_heads

        if args.hidden_units % args.num_heads != 0:
            raise ValueError(
                f"Hidden size {args.hidden_units} must be divisible by num_heads {args.num_heads}"
            )

        self.query = nn.Linear(args.hidden_units, args.hidden_units)
        self.key = nn.Linear(args.hidden_units, args.hidden_units)
        self.value = nn.Linear(args.hidden_units, args.hidden_units)

        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.output = nn.Linear(args.hidden_units, args.hidden_units)
        self.LayerNorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

    def forward(self, hidden_states, attention_mask):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Linear projections
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Expand attention mask to match scores shape
        # attention_mask shape: [seq_len, seq_len] -> [batch_size, num_heads, seq_len, seq_len]
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            attention_mask = attention_mask.expand(batch_size, self.num_heads, seq_len, seq_len)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            attention_mask = attention_mask.expand(batch_size, self.num_heads, seq_len, seq_len)

        # Apply attention mask
        scores = scores.masked_fill(attention_mask, float('-inf'))

        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Reshape back
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )

        # Output projection
        output = self.output(context)
        output = self.dropout(output)
        output = self.LayerNorm(output + hidden_states)

        return output


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # Input shape: [batch_size, seq_len, hidden_units]
        outputs = self.dropout2(self.conv2(self.relu(
            self.dropout1(self.conv1(inputs.transpose(-1, -2)))
        )))
        outputs = outputs.transpose(-1, -2)  # Back to [batch_size, seq_len, hidden_units]
        outputs += inputs  # Residual connection
        return outputs