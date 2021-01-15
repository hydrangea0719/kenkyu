# word_segmentation_by_mecab.py
# MeCabで形態素解析を行い，分かち書きをする

# 参考
# PythonでMeCabの出力をリスト化するモジュール(mecab-python)
# https://qiita.com/menon/items/2b5ad487a98882289567

import MeCab


def mecab_list(text):

    # MeCabを呼び出す(Taggerの引数は辞書の指定)
    tagger = MeCab.Tagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')
    tagger.parse('')
    node = tagger.parseToNode(text)
    # MeCab.Tagger というクラスのインスタンスを作成し，
    # parse もしくは parseToString メソッドを呼ぶことで，解析結果が文字列として取得できる

    word_class = []
    while node:
        word = node.surface
        wclass = node.feature.split(',')
        if wclass[0] != u'BOS/EOS':
            if wclass[6] is None:
                word_class.append((word, wclass[0], wclass[1], wclass[2], ""))
            else:
                word_class.append((word, wclass[0], wclass[1], wclass[2], wclass[6]))
        node = node.next
    return word_class


# 形態素解析したい文章
input_file = open("mai2008_01_short.txt", 'r')

# # 1行ずつリストに突っ込む（もしかして .split のほうが良い？）
# text = []
# for a in input_file:
#     text.append(a)

# str に入れる
text = ''
for a in input_file:
    text += a


file_output1 = open("mai2008_01_short_watati.txt", 'w')
file_output2 = open("mai2008_01_short_kaiseki.txt", 'w')

# ここでいろいろやる
# for a in text:
for a in text.split('\n'):
    # 形態素解析
    result = mecab_list(a)
    for line in result:
        # こっちはそのまま書き込み
        file_output2.writelines(line)

        # こっちは形態素だけを半角スペース区切りで抽出
        file_output1.writelines(line[0])
        file_output1.writelines(' ')

    # 改行
    file_output1.writelines('\n')
    file_output2.writelines('\n')


# 各行ごとに文章の構成単位に分解
# items = (re.split('[\t]', line) for line in lines)


# # 形態素解析した結果を表示
# for line in result:
#     file_output.writelines(line[0])
#     file_output.writelines(' ')
#     # file_output.writelines('\n')


file_output.close()
