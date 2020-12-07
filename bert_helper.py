import math

import torch
from typing import Union, List, Dict
import os
import re
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn  # ネットワーク構築用


# 最新版だと型ヒントは List でいいらしい. すごい！


class BertWords(nn.Module):
    def __init__(self, bert_model: BertModel):
        super().__init__()
        self.bert_model: BertModel = bert_model

    def forward(self, x, token_type_ids=None, **args):
        return self.bert_model(x, token_type_ids=token_type_ids, **args)[0]


class BertSentence(nn.Module):
    def __init__(self, bert_model: BertModel):
        super().__init__()
        self.bert_model = bert_model
        self.pooler = True
        if not self.pooler:
            self.bert_model.pooler = None

    def forward(self, x, token_type_ids=None, **args):
        if self.pooler:
            x = self.bert_model(x, token_type_ids=token_type_ids, **args)[1]
        else:
            x = self.bert_model(x, token_type_ids=token_type_ids, )[0]
            x = x[:, 0, :]
        return x


class BERTHelper:
    def __init__(self, mode: str = 'BASE'):
        self.mode = mode
        if mode == "BASE":
            self.root = "dataset/bert/model/Japanese/Japanese_L-12_H-768_A-12_E-30_BPE_transformers"
            self.modelPath = self.root + "/pytorch_model.bin"
        elif mode == "WWM":
            self.root = "./dataset/bert/model/Japanese/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers"
            self.modelPath = self.root + "/pytorch_model.bin"
        elif mode == 'TouhokuBERT':
            self.root = 'dataset/bert/model/Japanese/Touhoku_japanese-whole-word-masking'
            self.modelPath = self.root + "/pytorch_model.bin"
        elif mode == '' or mode is None:
            return
        self.config = BertConfig.from_json_file(self.root + "/config.json")
        self.bert_tokenizer = BertTokenizer(
            f"{self.root}/vocab.txt",
            do_lower_case=False, do_basic_tokenize=False)
        self.word_dict = {}

    # 使った試しはない
    def change_mode(self, mode):
        return self.__init__(mode=mode)

    def make_bert_model(self) -> BertModel:
        model = BertModel.from_pretrained(self.modelPath, config=self.config)
        model.eval()
        return model

    def make_bert_sentence_model(self) -> BertSentence:
        model = self.make_bert_model()
        return BertSentence(model)

    def make_bert_words_model(self) -> BertWords:
        model = self.make_bert_model()
        return BertWords(model)

    def tokenize(self, text: Union[str, list,tuple]):
        if isinstance(text, str):
            text = text
        elif isinstance(text, list) or isinstance(text, tuple):
            text = ' '.join(text)
        return self.bert_tokenizer.tokenize(text)

    # 最近はよくこれを使う
    def batch_encode_plus(self, sentence_list: str,
                          padding=True,
                          add_special_tokens=True,
                          is_split_into_words=True,
                          max_length=None) \
            -> Dict[str, List[int]]:
        # id_dic = {
        #    'input_ids': [[...],[...],[...]],
        #    'token_type_ids': [[...],[...],[...]],
        #    'attention_mask': [[...],[...],[...]]
        # }
        return self.bert_tokenizer.batch_encode_plus(
            sentence_list,  # 文章リスト
            add_special_tokens=add_special_tokens,  # [SEP]　などの特殊トークンをつける際は True
            is_split_into_words=is_split_into_words,  # すでに分かち書きされている文章の場合 True
            max_length=max_length,
            padding=padding
        )


if __name__ == '__main__':
    reibuns = ["しかし 、 彼 は 来 なかった 。", "そこで 、 しぶしぶ 彼女 が 来た 。"]  # 分かち書き済み
    bh = BERTHelper(mode='BASE')
    ids_dict = bh.batch_encode_plus(reibuns,
                                    padding=True,  # 最大長に合わせてpadを入れてくれる
                                    add_special_tokens=True,  # cls 文 sep にしてくれる
                                    is_split_into_words=True,  # 分かち書き済みなのでTrue
                                    max_length=502  # 500ちょいが限界？
                                    )
    # いい感じに subword 化 されるよ
    print(ids_dict)
    bert_sentence_model = bh.make_bert_sentence_model()
    bert_words_model = bh.make_bert_words_model()
    # 一応確かめておくよ
    print(bh.bert_tokenizer.convert_ids_to_tokens(ids_dict['input_ids'][0]))
    print(bh.bert_tokenizer.convert_ids_to_tokens(ids_dict['input_ids'][1]))
    # BERT にいれるよ
    input_ids = torch.LongTensor(ids_dict['input_ids'])
    token_type_ids = torch.LongTensor(ids_dict['token_type_ids'])
    attention_mask = torch.LongTensor(ids_dict['attention_mask'])
    print("入力サイズ",input_ids.size(), token_type_ids.size(), attention_mask.size())
    result1 = bert_sentence_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    result2 = bert_words_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    print("出力サイズ",result1.size(), result2.size())
    del bert_sentence_model, bert_words_model
    # ローカルで回すと重いから注意しよう！

