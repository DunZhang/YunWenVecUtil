import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import torch
from .ISentenceEncoder import ISentenceEncoder
import logging
from typing import Iterable


class BERTSentenceEncoder(ISentenceEncoder):
    def __init__(self, pretrained_model_name_or_path, device, pooling_modes=["cls"], batch_size=128, max_length=128):
        """
        :param pretrained_model_name_or_path:
            see parameter in transformer.BertModel.from_pretrained
        :param device:
            torch.device,
        :param pooling_modes:
            List-like, the way of getting vector, support 'mean' 'cls' 'max'
        :param batch_size:
            batch_size
        :param max_length:
            max length of sentenc

        """
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                              output_attentions=False, output_hidden_states=False).to(device)
        self.bert.eval()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.device = device
        self.pooling_modes = [i.strip().lower() for i in pooling_modes]
        self.batch_size = batch_size
        self.max_length = max_length
        logging.info("vec dim:{}".format(self.bert.config.hidden_size * len(pooling_modes)))

    def get_sens_vec(self, sens: Iterable[str]):
        res = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=sens, pad_to_max_length=True,
                                               return_tensors="pt", max_length=self.max_length)
        input_ids = res["input_ids"]
        attention_mask = res["attention_mask"]
        token_type_ids = res["token_type_ids"]
        logging.info("input ids shape: {},{}".format(input_ids.shape[0], input_ids.shape[1]))
        tensor_dataset = TensorDataset(input_ids, attention_mask, token_type_ids)
        sampler = SequentialSampler(tensor_dataset)
        data_loader = DataLoader(tensor_dataset, sampler=sampler, batch_size=self.batch_size)
        ### get sen vec
        all_sen_vec = []
        with torch.no_grad():
            for idx, batch_data in enumerate(data_loader):
                logging.info("get sentences vector: {}/{}".format(idx + 1, len(data_loader)))
                batch_data = [i.to(self.device) for i in batch_data]
                token_embeddings, pooler_output = self.bert(input_ids=batch_data[0], attention_mask=batch_data[1],
                                                            token_type_ids=batch_data[2])
                sen_vecs = []
                for pooling_mode in self.pooling_modes:
                    if pooling_mode == "cls":
                        sen_vec = pooler_output
                    elif pooling_mode == "mean":
                        # get mean token sen vec
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                        sum_mask = input_mask_expanded.sum(1)
                        sum_mask = torch.clamp(sum_mask, min=1e-9)
                        sen_vec = sum_embeddings / sum_mask
                    elif pooling_mode == "max":
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                        sen_vec = torch.max(token_embeddings, 1)[0]
                    sen_vecs.append(sen_vec)
                sen_vec = torch.cat(sen_vecs, 1)

                all_sen_vec.append(sen_vec.to("cpu").numpy())
        return np.vstack(all_sen_vec)
