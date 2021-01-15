
#

import MeCab
# from my_function import divide_into_sentences.py

import random
# import collections


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
        path_w = './dataset_mask/keiyoushi/' + filename + '_mask_keiyoushi' + '.txt'
        # path_r = './short/mai2008_01_short.txt'
        # path_w = './short/mai2008_01_short_mask_keiyoushi.txt'

        # 書き込み用ファイルを開く
        f_w = open(path_w, mode='w')

        with open(path_r) as f:
            for line in f:
                # print(line)

                # result = tagger.parse(line)
                # print(result)

                node = tagger.parseToNode(line)
                # print(type(node))
                # print(dir(node))
                # print()

                # ランダムに [MASK] でおきかえた後の文を入れる
                words = []
                # 置き換えたやつを入れる
                replaces = []
                # 形容詞であれば１を入れる
                is_keiyoushi = []

                while node:
                    # 全てのノードの 30% に [MASK] をかける
                    # 形容詞であれば [MASK] を，そうでないなら 20% で [MASK] をかける
                    if node.feature.split(',')[0] == '形容詞':
                        words.append('[MASK]')
                        replaces.append(node.surface)
                        is_keiyoushi.append('1')
                    else:
                        if node.feature.split(',')[0] != '記号':
                            if random.random() <= 0.2:
                                words.append('[MASK]')
                                replaces.append(node.surface)
                                is_keiyoushi.append('0')
                            else:
                                words.append(node.surface)
                        else:
                            words.append(node.surface)

                    node = node.next


                    # if node.feature.split(',')[0] != '記号':
                    #     if random.random() <= 0.3:
                    #
                    #         words.append('[MASK]')
                    #         replaces.append(node.surface)
                    #
                    #         if node.feature.split(',')[0] == '形容詞':
                    #             is_keiyoushi.append('1')
                    #         else:
                    #             is_keiyoushi.append('0')
                    #
                    #         # print(f'--- replace:{line} ---')
                    #     else:
                    #         words.append(node.surface)
                    #
                    # else:
                    #     words.append(node.surface)
                    #
                    # node = node.next


                text = ''.join(words)

                # [MASK] に置き換えるのは一文につき一箇所にしたいので，一つずつ置き換え直す
                text = text.replace('[MASK]', '{}')

                # [MASK] 部分が何であったかはこれで確認できる
                # print(' '.join(zyosis))

                for i, iranai in enumerate(replaces):

                    replaces_tmp = replaces[:]

                    # i 番目を [MASK] に置き換える
                    # これをループするので，[MASK] が一箇所ずつずれた文を取得できる
                    replaces_tmp[i] = '[MASK]'
                    # print(zyosis_tmp)

                    tmp_str = is_keiyoushi[i] + '\t' + text.format(*replaces_tmp)
                    f_w.writelines(tmp_str + '\n')

        # 書き込み用ファイルを閉じる
        f_w.close()


print('--- oshi-mai! ---')
