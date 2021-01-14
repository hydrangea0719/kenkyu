#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pkl ファイルの中身を取り出す
# mode='a' でファイルに追記できる，存在しない場合は新規作成
# .write() で文字列を記入，writelines() でリストを記入
import pickle


path = ['train_acc.pkl', 'train_loss.pkl', 'val_acc.pkl', 'val_loss.pkl']
path_w = 'pkl.txt'

for i in path:
    with open(i, mode='rb') as f:
        a = pickle.load(f)
    with open(path_w, mode='a')as f:
        print('{}\t{}'.format(i, a), file=f)
        # f.write('\n' + str(i) + '\t')
        # f.writelines(a)
