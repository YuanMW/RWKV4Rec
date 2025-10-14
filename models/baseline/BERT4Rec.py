import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List


class BertConfig:
    """Configuration for BERT model"""

    def __init__(self,
                 vocab_size,
                 hidden_size=256,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


def gelu(x):
    """GELU activation function"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_activation(activation_string):
    """Get activation function"""
    if not isinstance(activation_string, str):
        return activation_string

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return torch.nn.ReLU()
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return torch.nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {activation_string}")


class BertEmbeddings(nn.Module):
    """BERT embedding layer"""

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    """BERT self-attention layer"""

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})")

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Attention probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class BertSelfOutput(nn.Module):
    """BERT self-attention output layer"""

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """BERT attention module"""

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask=None):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    """BERT intermediate layer"""

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        if self.intermediate_act_fn is not None:
            hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """BERT output layer"""

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """BERT layer"""

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    """BERT encoder"""

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BERT4Rec(nn.Module):
    """BERT4Rec model - adapted for recommendation systems"""

    def __init__(self, user_num, item_num, args):
        super(BERT4Rec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.device = args.device

        # BERT configuration
        config = BertConfig(
            vocab_size=item_num + 1,  # +1 for padding token (0)
            hidden_size=args.hidden_units,
            num_hidden_layers=args.num_blocks,
            num_attention_heads=args.num_heads,
            intermediate_size=args.hidden_units * 4,  # Typically 4x hidden_size
            hidden_act=args.hidden_act,
            hidden_dropout_prob=args.dropout_rate,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob,
            max_position_embeddings=args.maxlen,
            type_vocab_size=2,
            initializer_range=0.02
        )

        self.bert_config = config

        # Keep item_emb for compatibility with existing code
        self.item_emb = nn.Embedding(item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.embeddings = BertEmbeddings(config)
        # Use item_emb as word_embeddings
        self.embeddings.word_embeddings = self.item_emb

        self.encoder = BertEncoder(config)

        # Item prediction head
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.activation = get_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def create_attention_mask(self, seq):
        """Create attention mask"""
        # Create padding mask [batch_size, 1, 1, seq_length]
        attention_mask = (seq > 0).unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.float()
        attention_mask = (1.0 - attention_mask) * -10000.0
        return attention_mask

    def log2feats(self, log_seqs):
        """Sequence to feature conversion - similar to SASRec interface"""
        input_ids = torch.LongTensor(log_seqs).to(self.device)

        # Create attention mask
        attention_mask = self.create_attention_mask(input_ids)

        # BERT forward pass
        embedding_output = self.embeddings(input_ids)
        encoded_layers = self.encoder(embedding_output, attention_mask, output_all_encoded_layers=False)
        sequence_output = encoded_layers[-1]  # Last layer output

        return sequence_output

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        """Training forward pass"""
        log_feats = self.log2feats(log_seqs)

        # Get embeddings for positive and negative samples
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.device))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.device))

        # Use last hidden state for prediction
        final_output = log_feats[:, -1, :]  # [batch_size, hidden_size]

        # Compute logits
        pos_logits = (final_output.unsqueeze(1) * pos_embs).sum(dim=-1)  # [batch_size, seq_len]
        neg_logits = (final_output.unsqueeze(1) * neg_embs).sum(dim=-1)  # [batch_size, seq_len]

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        """Inference forward pass"""
        log_feats = self.log2feats(log_seqs)

        # Use last hidden state
        final_output = log_feats[:, -1, :]  # [batch_size, hidden_size]

        # Get candidate item embeddings
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.device))

        # Compute logits
        logits = torch.matmul(item_embs, final_output.unsqueeze(-1)).squeeze(-1)

        return logits