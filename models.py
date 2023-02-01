# STRec is implemented based on Recbole.
# https://recbole.io/
import time
import torch
from torch import nn
import torch.nn.functional as fn

from recbole.model.abstract_recommender import SequentialRecommender
import math
import copy
import numpy as np
from recbole.model.abstract_recommender import SequentialRecommender
import os

def print_time(start_time):
    end_time = time.time()
    print(end_time - start_time)

class MultiHeadSparseAttention_pre(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadSparseAttention_pre, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + (attention_mask.unsqueeze(1) - 1) * 100000000

        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer_pre(nn.Module):
    def __init__(
            self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
            layer_norm_eps
    ):
        super(TransformerLayer_pre, self).__init__()
        self.multi_head_attention = MultiHeadSparseAttention_pre(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder_pre(nn.Module):
    def __init__(
            self,
            n_layers=2,
            n_heads=2,
            hidden_size=64,
            inner_size=256,
            hidden_dropout_prob=0.5,
            attn_dropout_prob=0.5,
            hidden_act='gelu',
            layer_norm_eps=1e-12
    ):

        super(TransformerEncoder_pre, self).__init__()
        layer = TransformerLayer_pre(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        self.n_layers = n_layers

    def forward(self, hidden_states, index_source, output_all_encoded_layers=True):
        b = hidden_states.size(0)
        l = hidden_states.size(1)
        all_encoder_layers = []
        i = 0
        mask = []
        alpha = [2, -0.2, -0.2, -0.3, -0.3, -0.5, -0.5, -0.7, -0.7]
        rand = torch.randn(b, l).to(hidden_states.device)
        self.layer_norm = nn.LayerNorm(l, eps=1e-10, elementwise_affine=False)
        index_source = self.layer_norm(index_source) + rand
        index_source[:, -1] = 10
        for j in range(len(alpha)):
            maski = (index_source + alpha[j]) * 100
            maski = torch.sigmoid(maski)
            mask.append(maski.unsqueeze(2).expand(b, l, l))

        for layer_module in self.layer:
            attention_mask = mask[i].permute(0, 2, 1).contiguous() * mask[i + 1]
            hidden_states = layer_module(hidden_states, attention_mask)
            i = i + 1
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class STRec_pre(SequentialRecommender):

    def __init__(self, config, dataset):
        super(STRec_pre, self).__init__(config, dataset)

        
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder_pre(
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
        self.ffn = FFN(1, 1, inner_size=32, hidden_dropout_prob=0,
                       layer_norm_eps=None)


        if self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE']!")

        self.apply(self._init_weights)
        self.device = config['device']

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def creat_index_source_old(self, time_stamps_seq):
        l = time_stamps_seq.size(1)
        origin_position_tensor = time_stamps_seq.unsqueeze(2).expand(-1, -1, l)
        position_tensor = time_stamps_seq.unsqueeze(1).expand(-1, l, -1)
        relative_position_tensor = origin_position_tensor - position_tensor
        relative_position_tensor = self.ffn(relative_position_tensor.unsqueeze(-1))
        relative_position_tensor = torch.sum(relative_position_tensor, dim=2)
        return relative_position_tensor.squeeze(-1).squeeze(-1)

    def creat_index_source(self, time_stamps_seq):
        torch.set_printoptions(threshold=np.inf)
        l = time_stamps_seq.size(1)
        time_stamps_seq = time_stamps_seq-time_stamps_seq[:, -1].view(-1, 1).expand(-1, l)
        time_stamps_seq = self.ffn(time_stamps_seq.unsqueeze(2))
        return time_stamps_seq.squeeze(2)

    def delete(self, item_seq, time_stamps_seq, pos_items, length):
        index = (length == self.max_seq_length).nonzero()
        index1 = index.view(-1, 1).expand(-1, item_seq.size(1))
        index2 = index.view(-1, 1).expand(-1, time_stamps_seq.size(1))
        index3 = index.view(-1)
        item_seq_out = item_seq.gather(dim=0, index=index1)
        time_stamps_seq_out = time_stamps_seq.gather(dim=0, index=index2)
        pos_items_out = pos_items.gather(dim=0, index=index3)

        return item_seq_out, time_stamps_seq_out, pos_items_out

    def forward(self, item_seq, time_stamps_seq):
        b = item_seq.size(0)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        index_source = self.creat_index_source(time_stamps_seq)
        trm_output = self.trm_encoder(input_emb, index_source, output_all_encoded_layers=True)
        output = trm_output[-1][:, -1, :].squeeze(1)

        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_stamps_seq = interaction['timestamp_list']
        pos_items = interaction[self.POS_ITEM_ID]
        item_seq, time_stamps_seq, pos_items = self.delete(item_seq, time_stamps_seq, pos_items, item_seq_len)
        seq_output = self.forward(item_seq, time_stamps_seq)

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        b = item_seq.size(0)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_stamps_seq = interaction['timestamp_list']
        test_item = interaction[self.ITEM_ID]
        item_seq, time_stamps_seq, test_item = self.delete(item_seq, time_stamps_seq, test_item, item_seq_len)
        if item_seq.size(0)>0:
            seq_output = self.forward(item_seq, time_stamps_seq)
            test_item_emb = self.item_embedding(test_item)
            scores_tensor = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        scores = torch.zeros(b, device=item_seq.device)-10000000
        j = 0
        for i in range(b):
            if item_seq_len[i] == self.max_seq_length:
                scores[i] = scores_tensor[j]
                j += 1

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        b = item_seq.size(0)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_stamps_seq = interaction['timestamp_list']
        test_item = interaction[self.ITEM_ID]
        item_seq, time_stamps_seq, test_item = self.delete(item_seq, time_stamps_seq, test_item, item_seq_len)
        if item_seq.size(0)>0:
            seq_output = self.forward(item_seq, time_stamps_seq)
            test_items_emb = self.item_embedding.weight
            scores_tensor = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        scores = torch.zeros(b, self.n_items, device=item_seq.device)-10000000
        j = 0
        for i in range(b):
            if item_seq_len[i] == self.max_seq_length:
                scores[i, :] = scores_tensor[j, :]
                j += 1
        return scores



class MultiHeadSparseAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadSparseAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, X):
        X = X.view(X.shape[0], X.shape[1], self.num_attention_heads, self.attention_head_size)
        X = X.permute(0, 2, 1, 3).contiguous()
        return X

    def gather_indexes(self, output, gather_index):
        if gather_index is None:
            return output
        if isinstance(gather_index, int):
            return output[:, :gather_index]
        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, output.size(-1)).long()
        gather_index.requires_grad = False
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor

    def forward(self, input_tensor, downsample_index):
        h = input_tensor.size(-1)
        downsample_tensor = self.gather_indexes(input_tensor, downsample_index)
        mixed_query_layer = self.query(downsample_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 1, 3, 2)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + downsample_tensor)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "relu": fn.relu,
            "gelu": self.gelu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadSparseAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, downsample_index2):
        attention_output = self.multi_head_attention(hidden_states, downsample_index2)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            n_layers=2,
            n_heads=2,
            hidden_size=64,
            inner_size=256,
            hidden_dropout_prob=0.5,
            attn_dropout_prob=0.5,
            hidden_act='gelu',
            layer_norm_eps=1e-12
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        self.n_layers = n_layers



    def forward(self, hidden_states, index_source, output_all_encoded_layers=False):
        b = hidden_states.size(0)
        l = hidden_states.size(1)
        i = 0
        rand = torch.randn(b, l).to(hidden_states.device)
        self.layer_norm = nn.LayerNorm(l, eps=1e-10, elementwise_affine=False)
        index_source = self.layer_norm(index_source) + rand
        index_source[:, -1] = 100
        if l == 50:
            alpha = [50, 50, 20, 20, 5, 5, 5, 5, 1]
        elif l == 8:
            alpha = [8, 8, 3, 3, 3, 3, 3, 3, 1]
        else:
            raise NotImplementedError("Sparsity not defined")

        for layer_module in self.layer:
            if alpha[i+1]==alpha[i] and i!=0:
                indexj = None
            elif i!=0:
                indexj = alpha[i+1]
            else:
                _, indexj = index_source.topk(alpha[i + 1], dim=1, largest=True, sorted=True)
            hidden_states = layer_module(hidden_states, indexj)
            i = i + 1
        return hidden_states


class FFN(nn.Module):

    def __init__(
            self, inputsize, outputsize, inner_size=64, hidden_dropout_prob=0.5, layer_norm_eps=1e-10
    ):
        super(FFN, self).__init__()
        self.dense_1 = nn.Linear(inputsize, inner_size)
        self.dense_2 = nn.Linear(inner_size, outputsize)
        self.layer_norm_eps = layer_norm_eps
        if layer_norm_eps is not None:
            self.LayerNorm = nn.LayerNorm(outputsize, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        if input_tensor.size(1) == 50:
            self.intermediate_act_fn = nn.Tanh()
        elif input_tensor.size(1) == 8:
            self.intermediate_act_fn = nn.ReLU()
        else:
            raise NotImplementedError("Intermediate_act_fn not defined")


        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.layer_norm_eps is not None:
            hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class STRec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(STRec, self).__init__(config, dataset)

        self.config = config
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob =config['hidden_dropout_prob']
        self.attn_dropout_prob =config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
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
        self.ffn = FFN(1, 1, inner_size=32, hidden_dropout_prob=0,
                       layer_norm_eps=None)

        if self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE']!")

        self.apply(self._init_weights)
        self.device = config['device']

        if config['mode'] == 'train':
            checkpoint = torch.load(config['pre_file'])
            self.load_state_dict(checkpoint['state_dict'], strict=False)
            self.load_other_parameter(checkpoint.get('other_parameter'))
        self.v6 = torch.tensor(-0.7824).to(self.device)


    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def creat_index_source(self, time_stamps_seq):
        l = time_stamps_seq.size(1)
        time_stamps_seq = time_stamps_seq-time_stamps_seq[:, -1].view(-1, 1).expand(-1, l)
        if self.config['hash'] == 'True':
            time_stamps_seq = self.hash(time_stamps_seq)
        else:
            time_stamps_seq = self.ffn(time_stamps_seq.unsqueeze(-1))
        return time_stamps_seq.squeeze(-1)

    def hash(self, t):
        device = t.device
        if t.size(1) == 50:

            c0 = t > -70
            c1 = t > -150
            c2 = t > -250
            c3 = t > -400
            c4 = t > -600
            c5 = t > -1000

            result = self.v6 + c5*0.015 + c4*0.06 + c3*0.1544 + c2*0.17 + c1*0.22 + c0*0.2
            return result
        else:
            raise NotImplementedError("The hash function is not defined.")
            

    def delete(self, item_seq, time_stamps_seq, pos_items, length):
        index = (length == self.max_seq_length).nonzero()
        index1 = index.view(-1, 1).expand(-1, item_seq.size(1))
        index2 = index.view(-1, 1).expand(-1, time_stamps_seq.size(1))
        index3 = index.view(-1)
        item_seq_out = item_seq.gather(dim=0, index=index1)
        time_stamps_seq_out = time_stamps_seq.gather(dim=0, index=index2)
        pos_items_out = pos_items.gather(dim=0, index=index3)

        return item_seq_out, time_stamps_seq_out, pos_items_out

    def forward(self, item_seq, time_stamps_seq):
        b = item_seq.size(0)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        index_source = self.creat_index_source(time_stamps_seq)
        trm_output = self.trm_encoder(input_emb, index_source, output_all_encoded_layers=False)
        output = trm_output[:, -1, :].squeeze(1)
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_stamps_seq = interaction['timestamp_list']
        pos_items = interaction[self.POS_ITEM_ID]
        item_seq, time_stamps_seq, pos_items = self.delete(item_seq, time_stamps_seq, pos_items, item_seq_len)

        seq_output = self.forward(item_seq, time_stamps_seq)
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        b = item_seq.size(0)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_stamps_seq = interaction['timestamp_list']
        test_item = interaction[self.ITEM_ID]
        item_seq, time_stamps_seq, test_item = self.delete(item_seq, time_stamps_seq, test_item, item_seq_len)
        if item_seq.size(0)>0:
            seq_output = self.forward(item_seq, time_stamps_seq)
            test_item_emb = self.item_embedding(test_item)
            scores_tensor = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        scores = torch.zeros(b, device=item_seq.device)-10000000
        j = 0
        for i in range(b):
            if item_seq_len[i] == self.max_seq_length:
                scores[i] = scores_tensor[j]
                j += 1

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        b = item_seq.size(0)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_stamps_seq = interaction['timestamp_list']
        test_item = interaction[self.ITEM_ID]
        item_seq, time_stamps_seq, test_item = self.delete(item_seq, time_stamps_seq, test_item, item_seq_len)
        if item_seq.size(0)>0:
            seq_output = self.forward(item_seq, time_stamps_seq)
            test_items_emb = self.item_embedding.weight
            scores_tensor = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        scores = torch.zeros(b, self.n_items, device=item_seq.device)-10000000
        j = 0
        for i in range(b):
            if item_seq_len[i] == self.max_seq_length:
                scores[i, :] = scores_tensor[j, :]
                j += 1
        return scores

    def predictx(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        b = item_seq.size(0)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_stamps_seq = interaction['timestamp_list']
        test_item = interaction[self.ITEM_ID]
        item_seq, time_stamps_seq, test_item = self.delete(item_seq, time_stamps_seq, test_item, item_seq_len)
        seq_output = self.forward(item_seq, time_stamps_seq)
        return seq_output