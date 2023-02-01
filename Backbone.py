import sys
import os
import time
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from logging import getLogger
import logging
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
import numpy as np
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss

import os


class SASRec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        # load parameters info
        self.config = config
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config[
            "inner_size"
        ]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)
        self.device = config['device']

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        trm_output = self.trm_encoder(
            input_emb, torch.zeros(1).to(item_seq.device), output_all_encoded_layers=True
        )

        output = trm_output[-1][:, -1, :]
        return output

    def delete(self, item_seq, time_stamps_seq, pos_items, length):
        index = (length == self.max_seq_length).nonzero()
        index1 = index.view(-1, 1).expand(-1, item_seq.size(1))
        index2 = index.view(-1, 1).expand(-1, time_stamps_seq.size(1))
        index3 = index.view(-1)
        item_seq_out = item_seq.gather(dim=0, index=index1)
        time_stamps_seq_out = time_stamps_seq.gather(dim=0, index=index2)
        pos_items_out = pos_items.gather(dim=0, index=index3)

        return item_seq_out, time_stamps_seq_out, pos_items_out

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        time_stamps_seq = interaction['timestamp_list']
        item_seq, time_stamps_seq, pos_items = self.delete(item_seq, time_stamps_seq, pos_items, item_seq_len)
        seq_output = self.forward(item_seq, item_seq_len)
        if self.loss_type == "BPR":
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
        k = self.config['eval_neg_sample_args']['sample_num']
        if k == 100:
            if b%(k+1) == 0:


                torch.set_printoptions(threshold=np.inf)
                print(item_seq[:180, :])
                item_seq = item_seq.view(-1, k+1, self.max_seq_length)[:, 0, :]
                print(item_seq)



                item_seq_len = item_seq_len.view(-1, k+1)[:, 0]
                time_stamps_seq = time_stamps_seq.view(-1, k+1, self.max_seq_length)[:, 0, :]
                test_item = test_item.view(-1, k+1)
                item_seq, time_stamps_seq, _ = self.delete(item_seq, time_stamps_seq, item_seq_len, item_seq_len)
                test_item, time_stamps_seq, _ = self.delete(test_item, time_stamps_seq, item_seq_len, item_seq_len)
                if item_seq.size(0)>0:
                    seq_output = self.forward(item_seq, time_stamps_seq)
                    test_item_emb = self.item_embedding(test_item)
                    scores_tensor = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                scores = torch.zeros(b, device=item_seq.device) - 10000000
                scores = scores.view(-1, k+1)
                j = 0
                for i in range(scores.size(0)):
                    if item_seq_len[i] == self.max_seq_length:
                        scores[i] = scores_tensor[j]
                        j += 1
                scores.view(-1)
                return scores

        item_seq, time_stamps_seq, test_item = self.delete(item_seq, time_stamps_seq, test_item, item_seq_len)
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores_tensor = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        scores = torch.zeros(b, device=item_seq.device)-10000000
        j = 0
        for i in range(b):
          if item_seq_len[i]==self.max_seq_length:
            scores[i] = scores_tensor[j]
            j+=1

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        b = item_seq.size(0)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_stamps_seq = interaction['timestamp_list']
        test_item = interaction[self.ITEM_ID]
        item_seq, time_stamps_seq, test_item = self.delete(item_seq, time_stamps_seq, test_item, item_seq_len)
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores_tensor = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        scores = torch.zeros(b, scores_tensor.size(1), device=item_seq.device)-10000000
        j = 0
        for i in range(b):
          if item_seq_len[i]==self.max_seq_length:
            scores[i, :] = scores_tensor[j, :]
            j+=1
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

def print_time(start_time):
    end_time = time.time()
    print(end_time - start_time)


if __name__ == '__main__':
    start_time = time.time()
    config = Config(model=SASRec, config_file_list=['STRec.yaml'])

    init_seed(config['seed'], config['reproducibility'])
    logging.getLogger().setLevel(logging.INFO)
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    

    if config['mode'] == 'test':
        checkpoint = torch.load(config['load_dir'])
        model = locals()[config['model']](config, train_data.dataset).to(config['device'])
        model.load_state_dict(checkpoint['state_dict'])
        model.load_other_parameter(checkpoint.get('other_parameter'))
        trainer = Trainer(config, model)
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)
        print(test_result)
        sys.exit()

    if config['mode'] == 'speed':
        print_time(start_time)
        checkpoint = torch.load(config['load_dir'])
        model = locals()[config['model']](config, train_data.dataset).to(config['device'])
        model.load_state_dict(checkpoint['state_dict'])
        model.load_other_parameter(checkpoint.get('other_parameter'))
        model.eval()
        for batch_idx, batched_data in enumerate(train_data):
            if batch_idx==2:
                print(torch.cuda.max_memory_allocated())
            batched_data = batched_data.to(config['device'])
            out = model.predictx(batched_data)
        print_time(start_time)
        sys.exit()


    # model loading and initialization
    model = SASRec(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))