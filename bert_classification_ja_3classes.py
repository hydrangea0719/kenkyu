#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 00:15:26 2020

@author: okada

2020.10.x
Fix: OKADA
毎日新聞社説接続詞前後一貫性推定用に修正

2020.10.16
交差検証用に変更するために作成
変更予定点
学習のところを train, test じゃなくて
(train, valid) x K times, and test にして
loss と accuracy のデータもそれで

2020.10.24
予測用関数追加

"""


# 使用する前に，訓練データ，テストデータのファイル名，train() か traincv() かを確認すること


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchtext

import transformers
import time
import string
import re


#-----------------------------------------------------------------------------
# tokenizer について
#
# torchtext.data.Field のメンバ変数として tokenize がある．
# 入力されるテキストをトークン化するために必要な関数を渡す．

# 前処理
def preprocessing_text(text):
    # 必要な処理を適宜書く
    # 改行用タグの除去
    # 念のために．
    text = re.sub('<br />', '', text)

    # カンマ，ピリオド以外の記号をスペースに置換
    #for p in string.punctuation:
    #    if (p == ".") or (p == ","):
    #        continue
    #    else:
    #        text = text.replace(p, " ")

    #　ピリオドなどの前後にスペースを挿入する
    #text = text.replace(".", " . ")
    #text = text.replace(",", " , ")

    return text

# 単語分割用 tokenizer の準備
# 今回，BERT の tokenizer を使う
# BERT の日本語版として東北大学乾研究室の BERT モデルが登録されているので
# それを使う．MeCab と mecab-python3 (mecab-python-windows) などの
# Python とつなげるもののインストールが必要
# 2020.10.10
# mecab-python-windows の後継 mecab が今のところ一番使い勝手がいい
# 東北大 BERT は fugashi を組み込んだみたい．
from transformers import BertJapaneseTokenizer
tokenizer_bert = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

#動作チェック
#test="[CLS]初めてですよ，[SEP]ここまで私をコケにしたおバカさん達は．[SEP]"
#tokens = tokenizer_bert.tokenize(test, add_special_tokens=True)
#print(tokens)

def tokenize(text, tokenizer=tokenizer_bert):
    text = preprocessing_text(text) # 前処理
#    text = tokenizer.tokenize(text, add_special_tokens=True) # トークン化
    text = tokenizer.tokenize(text) # トークン化
    text = ['[CLS]']+text+['[SEP]']
    return text

# torchtext の Field の設定
max_length = 350
LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

TEXT = torchtext.data.Field(sequential=True,
                            tokenize=tokenize,
                            use_vocab=True,
                            include_lengths=True,
                            pad_token="[PAD]",
                            unk_token="[UNK]",
                            batch_first=True,
                            fix_length=max_length)

# データを読み込む
# dataset の読み込み

# データ 1 個読み込み
def loaddata(data_file):
    ds = torchtext.data.TabularDataset(
        path=data_file,
        format="TSV",
        fields=[("label",LABEL),("text",TEXT)]
    )
    return ds

# 訓練データ，テストデータ
def loaddatatraintest(train_data_file, test_data_file):
    train_ds = loaddata(train_data_file)
    test_ds = loaddata(test_data_file)
    return train_ds, test_ds

# 交差検証
# CV 用訓練データと検証データ
def loaddatatraintestcv(train_cv_data_file, valid_cv_data_file, s_ext, no_k_fold):
    trainvalid_ds = []
    for i in range(no_k_fold):
        train_cv_ds = loaddata(train_cv_data_file+str(i)+s_ext)
        valid_cv_ds = loaddata(valid_cv_data_file+str(i)+s_ext)
        # ここもしかしてタプルを append してるな
        trainvalid_ds.append((train_cv_ds, valid_cv_ds))
    return trainvalid_ds

# dataset の設定
# dataloader 1 個設定
# ds: データセット
# batchsize: バッチサイズ
# train: 訓練 (True) / 訓練でない (False) デフォルト True
# shuffle: シャッフルするかしないか (True/False) デフォルト None
# sort: 整列するかしないか (True/False) デフォルト None
# デフォルト値は元の torchtext.data.BucketIterator に準拠．
def createdataloader(ds, batchsize, train=True, sort=None, shuffle=None):
    dl = torchtext.data.BucketIterator(
        dataset=ds,
        batch_size=batchsize,
        train=train,
        shuffle=shuffle,
        sort=sort
    )
    return dl

# 訓練データとテストデータ
def createdataloadertraintest(train_ds, test_ds, batchsize):
    train_dl = createdataloader(train_ds, batchsize, train=True, shuffle=True)
    test_dl = createdataloader(test_ds, batchsize, train=False, sort=False)
    return train_dl, test_dl

# 交差検証用
def createdataloadercv(trainvalid_ds, batchsize):
    trainvalid_dl=[]
    for i in range(len(trainvalid_ds)):
        train_cv_dl, valid_cv_dl = createdataloadertraintest(
            trainvalid_ds[i][0], trainvalid_ds[i][1], batchsize
        )
        trainvalid_dl.append((train_cv_dl, valid_cv_dl))
    return trainvalid_dl

# 保存
def savemodelandresults(train_file_name, train_ds, test_file_name, test_ds, num_epoch,
                        t_start, t_end, bn, optimizer, criterion,
                        train_loss_list, val_loss_list, train_acc_list, val_acc_list):
    # 情報
    with open("info.dat", "w") as f:
        f.write("Experimental information\n\n")
        # 訓練データファイル名，テストデータファイル名
        f.write("Training data: {}\n".format(train_file_name))
        f.write("No. of training data: {}\n".format(len(train_ds)))
        f.write("Test data: {}\n".format(test_file_name))
        f.write("No. of test data: {}\n".format(len(test_ds)))
        f.write("\n")

        # 総エポック数
        f.write("Total number of epoch: {}\n".format(num_epoch))
        f.write("\n")

        # 時間
        f.write("Total time: {}\n".format(t_end-t_start))
        f.write("\n")

        # モデルの情報
        f.write("Model\n")
        f.write(str(bn))
        f.write("\n")

        # 最適化関数の情報
        f.write("Optimizer\n")
        f.write(str(optimizer))
        f.write("\n")

        # 損失関数の情報
        f.write("Criterion (Loss function)\n")
        f.write(str(criterion))
        f.write("\n")


    # モデルの保存
    save_path = "bert_classfication_model_weight.pth"
    torch.save(bn.to('cpu').state_dict(), save_path) # モデルを GPU から CPU　へ．

    #　loss と accuracy
    import pickle
    with open("train_loss.pkl", 'wb') as f:
        pickle.dump(train_loss_list, f)
    with open("val_loss.pkl", 'wb') as f:
        pickle.dump(val_loss_list, f)
    with open("train_acc.pkl", 'wb') as f:
        pickle.dump(train_acc_list, f)
    with open("val_acc.pkl", 'wb') as f:
        pickle.dump(val_acc_list, f)


# 描画
def drawlossaccdata(num_epoch, train_loss_list, val_loss_list, train_acc_list, val_acc_list):
    import matplotlib.pyplot as plt
    #%matplotlib inline

    plt.figure()

    plt.plot(range(num_epoch), train_loss_list, color="blue",
             linestyle='-', label="train_loss")
    plt.plot(range(num_epoch), val_loss_list, color="green",
             linestyle='--', label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training and validation loss")
    plt.grid()
    plt.show()
    plt.savefig("bert_classification_loss.png")

    plt.figure()
    plt.plot(range(num_epoch), train_acc_list, color="blue",
             linestyle='-', label="train_acc")
    plt.plot(range(num_epoch), val_acc_list, color="green",
             linestyle='--', label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Training and validation accuracy")
    plt.grid()
    plt.ylim([0.0,1.0]) # y 軸表示範囲 0 から 1 までに設定
    plt.show()
    plt.savefig("bert_classification_acc.png")



def train_body(train_ds, test_ds, train_dl, test_dl, train_file_name, test_file_name):
    #-------------------------------------------------------------------------
    # モデルの読み込み（日本語 BERT (東北大) ）
    #
    print("--- Loading BERT pretrained model ---")

    from transformers import BertTokenizer, BertModel, BertConfig

    # パラメータを読み込む
    # transformers で用意されている場合は from_pretrained(),
    # それ以外はそのモデルが指定する方法で読み込む
    # 京都大学黒橋研の日本語 BERT なら json 形式．
    config = BertConfig.from_json_file(f"BERT-base_mecab-ipadic-bpe-32k/config.json")
    # 読み込んだパラメータを使ってモデルを読み込む
    model = BertModel.from_pretrained(f"BERT-base_mecab-ipadic-bpe-32k/pytorch_model.bin", config=config)
    # パラメータを使って単語と ID の変換器を読み込む
#    tokenizer = BertTokenizer(f"BERT-base_mecab-ipadic-bpe-32k/vocab.txt",
#                              do_lower_case=False, do_basic_tokenize=False)
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')


    #-------------------------------------------------------------------------
    # 学習と推測
    # ファインチューニング

    # モデル：BERT + 線形層1層
    class BERT_Net(nn.Module):
        def __init__(self, bert_model):
            super(BERT_Net, self).__init__()
            self.bert_model = bert_model
            # ここで二値分類と三値分類の区別ができるはずなんだが
            # self.L = nn.Linear(768, 2)
            self.L = nn.Linear(768, 3)


        def forward(self, x):
            outputs = model(x)
            last_hidden_states = outputs[0]
            sentence_vec_list = last_hidden_states[:,0,:]
            return self.L(sentence_vec_list)


    bn = BERT_Net(bert_model=model)

    # GPU に投げられそうなら投げる
    device = 'cuda' if torch.cuda.is_available() == True else 'cpu'
    bn = bn.to(device)

    # ファインチューニングの準備
    # いったん全部のパラメータの requires_grad を False で更新
    for name, param in bn.named_parameters():
        param.requires_grad = False
    # BERT　の Encoder の最終レイヤの requires_grad を True で更新
    for name, param in bn.bert_model.encoder.layer[-1].named_parameters():
        param.requires_grad = True
    # 付け加えた線形層も True で更新
    for name, param in bn.L.named_parameters():
        param.requires_grad = True

    # 損失関数と最適化関数
    # 損失関数はクロスエントロピー
    criterion = nn.CrossEntropyLoss()
    # 最適化関数 Adam
    # パラメータ設定は BERT の元論文で推奨されている値
    # BERT モデルの最終レイヤと付加した線形層のパラメータをそれぞれ設定している
    optimizer = optim.Adam([
        {"params": bn.bert_model.encoder.layer[-1].parameters(), "lr": 5e-5},
        {"params": bn.L.parameters(), "lr": 5e-5}
        ], betas=(0.9, 0.999))


    # エポック数
    num_epoch = 10 if torch.cuda.is_available() == True else 1

    print("No. of Epoch == {}".format(num_epoch))

    # バッチサイズ
    batch_size = train_dl.batch_size

    # loss と accuracy のグラフをプロットするためのリスト
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    print("--- START ---")

    t_start=time.time() # start time

    for epoch in range(num_epoch):
        # エポックごとに初期化
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        iteration=1

        # 開始時間
        t_epoch_start=time.time()
        t_iter_start=time.time()

        # train --------------------------------------------------------------
        bn.train()

        for batch in train_dl:
            inputs = batch.text[0].to(device)
            labels = batch.label.to(device)

            # 勾配リセット
            optimizer.zero_grad()

            # feed forward
            outputs = bn(inputs)

            # calculate loss
            loss = criterion(outputs, labels)
            # ラベル予測
            axis=1 # row (行)
            _, preds = torch.max(outputs, axis)

            # loss と accuracy のミニバッチ分をため込む
            train_loss += loss.item()
            train_acc += torch.sum(preds == labels.data)

            # back propagation
            loss.backward()
            # 重みの計算
            optimizer.step()

            if (iteration % 10 == 0):
                t_iter_finish = time.time()
                duration = t_iter_finish - t_iter_start
                acc = (torch.sum(preds == labels.data)).double()/train_dl.batch_size
                print("イテレーション　{} || Loss: {:.4f} || 10iter: {:.4f} sec. \
|| 本イテレーションの正解率: {}".format(iteration, loss.item(), duration, acc))
                t_iter_start=time.time()

            iteration += 1

            #train_loss = loss.item() * train_dl.batch_size
            #train_acc += torch.sum(preds == labels.data)

        t_epoch_finish = time.time()
        # loss と accuracy の計算
        train_loss = train_loss / len(train_dl.dataset)
        train_acc = train_acc.double() / len(train_dl.dataset)

        print("Epoch {}/{} | {:^5} | Loss: {:.4f} Acc: {:.4f}".format(
            epoch+1, num_epoch, "train", train_loss, train_acc))

        # loss と accuracy をプロットする
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc.item())


        # eval ---------------------------------------------------------------
        # 評価モードへ切り替え
        bn.eval()

        # 開始時間記録
        t_iter_start_val=time.time()

        # 評価時に勾配計算などをしないように torch.no_grad() を使用
        with torch.no_grad():
            # test data で推定してチェックする
            # 簡単のために，train data で回してみる
            # 流れは train と一緒
            # 勾配計算部分が除かれている
            for batch_val in test_dl:
                inputs_val = batch_val.text[0].to(device)
                labels_val = batch_val.label.to(device)

                # feed forward
                outputs_val = bn(inputs_val)

                # calclate loss
                loss = criterion(outputs_val, labels_val)

                # loss のミニバッチ分をため込む
                val_loss += loss.item()

                # accuracy のミニバッチ分をため込む
                axis=1 # row (行)
                _, preds = torch.max(outputs_val, axis) # ラベル予測
                val_acc += torch.sum(preds == labels_val.data)
                print(preds, labels_val.data)

            val_loss = val_loss / len(test_dl.dataset)
            val_acc = val_acc.double() / len(test_dl.dataset)

            print("Epoch {}/{} | {:^5} | Loss: {:.4f} Acc: {:.4f}".format(
                epoch+1, num_epoch, "val", val_loss, val_acc))

        # loss と accuracy をプロットする
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc.item())

    t_end = time.time() # finish time

    # 保存 --------------------------------------------------------------------
    # モデルや現在時刻のディレクトリを作ってそこに保存する
    import datetime
    now = datetime.datetime.now()
    save_path_dir = now.strftime("%Y%m%d-%H%M%S")


    import os
    os.mkdir(save_path_dir)

    os.chdir(save_path_dir)

    savemodelandresults(train_file_name, train_ds, test_file_name, test_ds, num_epoch,
                        t_start, t_end, bn, optimizer, criterion,
                        train_loss_list, val_loss_list, train_acc_list, val_acc_list)
    # 訓練データファイル名，テストデータファイル名，総エポック数，時間，モデルの情報，最適化関数の情報，損失関数の情報
    # を保存する

    # 元のディレクトリに返る
    os.chdir("..")

    print("--- END ---")

    #-------------------------------------------------------------------------
    # 描画
    # ディレクトリ移動
    os.chdir(save_path_dir)

    drawlossaccdata(num_epoch, train_loss_list, val_loss_list, train_acc_list, val_acc_list)

    # 元のディレクトリに返る
    os.chdir("..")


# def train():
#
#     print("--- Sentiment Analysis Using BERT Pretrained Model ---")
#
#     print('--- Loading training dataset and test dataset ---')
#
#     # データロード
#     # 訓練データ，テストデータ
#     # 訓練データファイル名とテストデータファイル名
#     # train_data_file = "mai.shasetsu.gyakusetsu.train.txt"
#     # test_data_file = "mai.shasetsu.gyakusetsu.test.txt"
#
#     # 多分ここを触ればどうにかなると思うがｾﾝｾのコード触るの嫌すぎるなあとかたもなく破壊しそう
#     # train_data_file = "./dataset_mask/kakujoshi/mai2008_01_mask_kakujoshi.txt"
#     # test_data_file = "./dataset_mask/kakujoshi/mai2009_01_mask_kakujoshi.txt"
#     # train_data_file = './dataset_mask/kakujoshi/train.txt'
#     # test_data_file = './dataset_mask/kakujoshi/test.txt'
#
#
#
#     train_ds, test_ds = loaddatatraintest(train_data_file, test_data_file)
#
#     # vocab を生成する
#     # BERTTokenizer の vocabulary を取ってくる
#     # 辞書型．単語と ID を変換する．
#     print("--- Get vocabulary data ---")
#     vocab = tokenizer_bert.get_vocab()
#
#     # Field "TEXT" に　vocab を渡したらいいのだけれど，そのままでは渡せない．
#     # 一度 train_ds のデータを渡して，作ってから上書きする．
#     # train_ds が必要だから外に出せない．めんどい．
#     # ref. 「PyTorch による発展ディープラーニング」．この本すげぇ．全部書いてある．
#     TEXT.build_vocab(train_ds, min_freq=1)
#     TEXT.vocab.stoi=vocab # stoi: "string to id"．文字列から ID へ．辞書型．
#
#     # バッチ処理用にデータを取ってくる
#     # Dataloader を準備する
#     # torchtext だと iterator になる
#     # まとめて作る方法もあるけれど，練習と分かりやすさのために別々にしている．
#     print("--- Create dataloader (iterator) ---")
#
#     batchsize = 10 if torch.cuda.is_available() == True else 2 # GPU 周り処理
#
#     print("--- Batch size == {} ---".format(batchsize))
#
#     train_dl, test_dl = createdataloadertraintest(train_ds, test_ds, batchsize) # 訓練データとテストデータ
#
#     print("--- Finish loading training dataset and test dataset ---")
#
#     # 学習：訓練データとテストデータ
#     train_body(train_ds, test_ds, train_dl, test_dl, train_data_file, test_data_file)

def traincv():

    print("--- Sentiment Analysis Using BERT Pretrained Model ---")

    print('--- Loading training dataset and test dataset ---')

    # データロード
    # 交差検証用データ
    # train_cv_data_file = "mai.shasetsu.gyakusetsu.traincv"
    # valid_cv_data_file = "mai.shasetsu.gyakusetsu.validcv"
    train_cv_data_file = 'train_mai2008_01_0_cv'
    valid_cv_data_file = 'valid_mai2008_01_0_cv'
    s_ext = ".txt"
    no_k_fold = 5 # 交差検証の回数．ここでは 5 回．

    trainvalid_ds = loaddatatraintestcv(train_cv_data_file, valid_cv_data_file, s_ext, no_k_fold)

    # vocab を生成する
    # BERTTokenizer の vocabulary を取ってくる
    # 辞書型．単語と ID を変換する．
    print("--- Get vocabulary data ---")
    vocab = tokenizer_bert.get_vocab()

    # Field "TEXT" に　vocab を渡したらいいのだけれど，そのままでは渡せない．
    # 一度 train_ds のデータを渡して，作ってから上書きする．
    # train_ds が必要だから外に出せない．めんどい．
    # ref. 「PyTorch による発展ディープラーニング」．この本すげぇ．全部書いてある．
    # TEXT.build_vocab(train_ds, min_freq=1)
    # k=5 で分割してるので5つ！！！やったーーーー
    TEXT.build_vocab(trainvalid_ds[0][0], trainvalid_ds[0][1], trainvalid_ds[1][0], trainvalid_ds[1][1], trainvalid_ds[2][0], trainvalid_ds[2][1], trainvalid_ds[3][0], trainvalid_ds[3][1], trainvalid_ds[4][0], trainvalid_ds[4][1], min_freq=1)
    # なにやってもうまくいかん！！！
    # うまくいった！！！動いたら勝ち！！！
    TEXT.vocab.stoi=vocab # stoi: "string to id"．文字列から ID へ．辞書型．

    # バッチ処理用にデータを取ってくる
    # Dataloader を準備する
    # torchtext だと iterator になる
    # まとめて作る方法もあるけれど，練習と分かりやすさのために別々にしている．
    print("--- Create dataloader (iterator) ---")

    batchsize = 10 if torch.cuda.is_available() == True else 2 # GPU 周り処理

    print("--- Batch size == {} ---".format(batchsize))

    trainvalid_dl = createdataloadercv(trainvalid_ds, batchsize) # 交差検証用データ

    print("--- Finish loading training dataset and test dataset ---")

    # 学習：交差検証
    for i in range(no_k_fold):
        train_cv_ds = trainvalid_ds[i][0]
        valid_cv_ds = trainvalid_ds[i][1]
        train_cv_dl = trainvalid_dl[i][0]
        valid_cv_dl = trainvalid_dl[i][1]
        train_cv_data_name = train_cv_data_file+str(i)+s_ext
        valid_cv_data_name = valid_cv_data_file+str(i)+s_ext
        train_body(train_cv_ds, valid_cv_ds, train_cv_dl, valid_cv_dl, train_cv_data_name, valid_cv_data_name)

def predict():
    # model を読み込んで，データを分類する
    #-------------------------------------------------------------------------
    # モデルの読み込み（日本語 BERT (東北大) ）
    #
    print("--- Loading BERT pretrained model ---")

    from transformers import BertTokenizer, BertModel, BertConfig

    # パラメータを読み込む
    # transformers で用意されている場合は from_pretrained(),
    # それ以外はそのモデルが指定する方法で読み込む
    # 京都大学黒橋研の日本語 BERT なら json 形式．
    config = BertConfig.from_json_file(f"BERT-base_mecab-ipadic-bpe-32k/config.json")
    # 読み込んだパラメータを使ってモデルを読み込む
    model = BertModel.from_pretrained(f"BERT-base_mecab-ipadic-bpe-32k/pytorch_model.bin", config=config)
    # パラメータを使って単語と ID の変換器を読み込む
#    tokenizer = BertTokenizer(f"BERT-base_mecab-ipadic-bpe-32k/vocab.txt",
#                              do_lower_case=False, do_basic_tokenize=False)
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

    # モデル：BERT + 線形層1層
    class BERT_Net(nn.Module):
        def __init__(self, bert_model):
            super(BERT_Net, self).__init__()
            self.bert_model = bert_model
            self.L = nn.Linear(768, 2)
        def forward(self, x):
            outputs = model(x)
            last_hidden_states = outputs[0]
            sentence_vec_list = last_hidden_states[:,0,:]
            return self.L(sentence_vec_list)

    bn = BERT_Net(bert_model=model)

    # モデルの読み込み
    bn.load_state_dict(torch.load("bert_classfication_model_weight.pth"))
    bn.eval()

    # CPU / GPU
    # GPU に投げられそうなら投げる
    device = 'cuda' if torch.cuda.is_available() == True else 'cpu'
    bn = bn.to(device)

    # 損失関数と最適化関数
    # 損失関数はクロスエントロピー
    criterion = nn.CrossEntropyLoss()

    # データの読み込み
    print('--- Loading training dataset and test dataset ---')

    # データロード
    # 訓練データ，テストデータ
    # 訓練データファイル名とテストデータファイル名
    test_data_file = "mai.shasetsu.gyakusetsu.test.txt"

    test_ds = loaddata(test_data_file)

    # vocab を生成する
    # BERTTokenizer の vocabulary を取ってくる
    # 辞書型．単語と ID を変換する．
    print("--- Get vocabulary data ---")
    vocab = tokenizer_bert.get_vocab()

    # Field "TEXT" に　vocab を渡したらいいのだけれど，そのままでは渡せない．
    # 一度 train_ds のデータを渡して，作ってから上書きする．
    # train_ds が必要だから外に出せない．めんどい．
    # ref. 「PyTorch による発展ディープラーニング」．この本すげぇ．全部書いてある．
    TEXT.build_vocab(test_ds, min_freq=1)
    TEXT.vocab.stoi=vocab # stoi: "string to id"．文字列から ID へ．辞書型．

    # バッチ処理用にデータを取ってくる
    # Dataloader を準備する
    # torchtext だと iterator になる
    # まとめて作る方法もあるけれど，練習と分かりやすさのために別々にしている．
    print("--- Create dataloader (iterator) ---")

    batchsize = 10 if torch.cuda.is_available() == True else 2 # GPU 周り処理

    print("--- Batch size == {} ---".format(batchsize))

    test_dl = createdataloader(test_ds, batchsize, train=False, sort=False) # 予測データ

    print("--- Finish loading training dataset and test dataset ---")

    # エポック数
    num_epoch = 10 if torch.cuda.is_available() == True else 1

    print("No. of Epoch == {}".format(num_epoch))

    # バッチサイズ
    batch_size = test_dl.batch_size

    # loss と accuracy のグラフをプロットするためのリスト
    val_loss_list = []
    val_acc_list = []

    print("--- START ---")

    t_start=time.time() # start time

    # 分類と正誤チェック
    # eval ---------------------------------------------------------------
    # 評価モードへ切り替え
    #bn.eval()

    val_loss = 0.0
    val_acc = 0.0
    # 開始時間記録
    t_iter_start_val=time.time()

    # 評価時に勾配計算などをしないように torch.no_grad() を使用
    with torch.no_grad():
        # test data で推定してチェックする
        # 簡単のために，train data で回してみる
        # 流れは train と一緒
        # 勾配計算部分が除かれている
        for batch_val in test_dl:
            inputs_val = batch_val.text[0].to(device)
            labels_val = batch_val.label.to(device)

            # feed forward
            outputs_val = bn(inputs_val)

            # calclate loss
            loss = criterion(outputs_val, labels_val)

            # loss のミニバッチ分をため込む
            val_loss += loss.item()

            # accuracy のミニバッチ分をため込む
            axis=1 # row (行)
            _, preds = torch.max(outputs_val, axis) # ラベル予測
            val_acc += torch.sum(preds == labels_val.data)

            print(preds, labels_val.data)

        val_loss = val_loss / len(test_dl.dataset)
        val_acc = val_acc.double() / len(test_dl.dataset)

        print("{:^5} | Loss: {:.4f} Acc: {:.4f}".format(
            "val", val_loss, val_acc))

    # loss と accuracy をプロットする
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc.item())

    t_end = time.time() # finish time

    # 出力
    return

def main():
    # train()
    traincv()
    #predict()

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
