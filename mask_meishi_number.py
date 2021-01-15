
# 名詞に [MASK] をかけ，数詞かどうかを判定する

import MeCab
# from my_function import divide_into_sentences.py

import random
# import collections


# 適切な辞書を入れてください未来の俺
tagger = MeCab.Tagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')

# input = '今週末は金剛山に登りに行って、そのあと新世界かどっかで打ち上げしましょ。'
# result = tagger.parse(input)
# print(result)
# node = tagger.parseToNode(input)
# input をファイルの形式にする必要がある


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
        path_w = './dataset_mask/meishi_number/' + filename + '_mask_meishi_number' + '.txt'
        # path_r = './short/mai2008_01_short.txt'
        # path_w = './short/mai2008_01_short_mask_meishi.txt'

        # 書き込み用ファイルを開く
        f_w = open(path_w, mode='w')

        with open(path_r) as f:
            for line in f:
                # print(line)

                result = tagger.parse(line)
                # print(result)

                node = tagger.parseToNode(line)
                # print(type(node))
                # print(dir(node))
                # print()


                # ランダムに [MASK] でおきかえた後の文を入れる
                words = []
                # 置き換えたやつを入れる
                replaces = []
                # 名詞であれば１を入れる
                is_meishi = []

                while node:

                    if node.feature.split(',')[0] == '名詞':
                        words.append('[MASK]')
                        replaces.append(node.surface)

                        if node.feature.split(',')[1] == '数':
                            is_meishi.append('1')
                        else:
                            is_meishi.append('0')

                    else:
                        words.append(node.surface)

                    node = node.next


                text = ''.join(words)
                text = text.replace('[MASK]', '{}')


                for i, iranai in enumerate(replaces):

                    replaces_tmp = replaces[:]
                    replaces_tmp[i] = '[MASK]'
                    tmp_str = is_meishi[i] + '\t' + text.format(*replaces_tmp)
                    f_w.writelines(tmp_str + '\n')

        f_w.close()


print('--- oshi-mai! ---')
