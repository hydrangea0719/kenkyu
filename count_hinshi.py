
#

import MeCab
# from my_function import divide_into_sentences.py

# import random
# import collections
from pprint import pprint


# 適切な辞書を入れてください未来の俺
tagger = MeCab.Tagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')


# path は絶対パスか相対パスで指定する（これ str のファイル名じゃなかったんやな...）
tmp1 = ['2008', '2009', '2010', '2011', '2012']
tmp2 = ['01', '02', '03']


print('--- kaiseki suruyo ---')

for year in tmp1:
    for page in tmp2:

        # ファイルの準備をしてね
        filename = 'mai' + year + '_' + page
        print('--- ' + filename + '.txt wo hiraite iruyo ---')

        path_r = './dataset_text/' + filename + '.txt'
        path_w = './dataset_text/info_hinshi/info_' + filename + '.txt'

        f_w = open(path_w, mode='w')

        # 品詞とその出現回数を辞書で管理する
        # 今のところ，助詞の中の格助詞と接続助詞を分けることはできない
        h_count = {}

        with open(path_r) as f:
            for line in f:

                node = tagger.parseToNode(line)

                while node:
                    # 品詞（IPA品詞体系）に従ってカウント
                    hinshi_1 = node.feature.split(',')[0]
                    hinshi_2 = node.feature.split(',')[1]
                    hinshi = hinshi_1+'_'+hinshi_2

                    if hinshi in h_count.keys():
                        freq = h_count[hinshi]
                        h_count[hinshi] = freq + 1
                    else:
                        h_count[hinshi] = 1

                    node = node.next

        # for key, value in h_count.items():
            # print('{:<10} : {}'.format(key, str(value)), file=f_w)

        # pprint(h_count, stream=f_w)
        dic1 = sorted(h_count.items())
        dic2 = sorted(h_count.items(), key=lambda x:x[1], reverse=True)

        pprint(dic1, stream=f_w)
        f_w.write('----------\n')
        pprint(dic2, stream=f_w)


        f_w.close()


print('--- oshi-mai! ---')
