import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CL4SRec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(CL4SRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.device = args.device

        self.n_layers = args.num_blocks
        self.n_heads = args.num_heads
        self.hidden_size = args.hidden_units
        self.inner_size = args.hidden_units * 4  # Typically 4 times the hidden_size
        self.hidden_dropout_prob = args.dropout_rate
        self.attn_dropout_prob = args.attention_probs_dropout_prob
        self.hidden_act = args.hidden_act
        self.layer_norm_eps = 1e-12

        self.initializer_range = 0.02
        self.max_seq_length = args.maxlen
        self.temp = 0.2  # Contrastive learning temperature parameter

        # Define layers
        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

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
        """Generate attention mask"""
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

        # Transformer encoder
        trm_output = self.trm_encoder(input_emb, attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]  # Last layer output

        return output

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

    # Contrastive learning related methods
    def augmented_forward(self, seq, aug_seq):
        """Forward pass for contrastive learning"""
        seq_feats = self.log2feats(seq)
        aug_feats = self.log2feats(aug_seq)

        seq_output = seq_feats[:, -1, :]  # Last position
        aug_output = aug_feats[:, -1, :]

        return seq_output, aug_output

    def contrastive_loss(self, seq_output, aug_output):
        """Compute contrastive learning loss"""
        batch_size = seq_output.size(0)

        # Normalize
        seq_output = F.normalize(seq_output, dim=1)
        aug_output = F.normalize(aug_output, dim=1)

        # Compute similarity matrix
        logits = torch.matmul(seq_output, aug_output.T) / self.temp

        # Contrastive learning targets
        labels = torch.arange(batch_size, device=seq_output.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads))

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask
        attention_scores = attention_scores + attention_mask

        # Attention probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output projection
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        if hidden_act == 'gelu':
            self.activation = F.gelu
        elif hidden_act == 'relu':
            self.activation = F.relu
        else:
            self.activation = F.gelu

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerLayer(nn.Module):
    def __init__(self, n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
                 layer_norm_eps):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        layer_output = self.feed_forward(attention_output)
        return layer_output


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
                 layer_norm_eps):
        super(TransformerEncoder, self).__init__()
        self.layer = nn.ModuleList([
            TransformerLayer(n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
                             layer_norm_eps)
            for _ in range(n_layers)
        ])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers