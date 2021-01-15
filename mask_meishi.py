
#

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
        path_w = './dataset_mask/meishi/' + filename + '_mask_meishi' + '.txt'
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
                    # 全てのノードの 20% に [MASK] をかける
                    if node.feature.split(',')[0] != '記号':
                        if random.random() <= 0.2:

                            words.append('[MASK]')
                            replaces.append(node.surface)

                            if node.feature.split(',')[0] == '名詞':
                                is_meishi.append('1')
                            else:
                                is_meishi.append('0')

                            # print(f'--- replace:{line} ---')
                        else:
                            words.append(node.surface)
                            
                    else:
                        words.append(node.surface)


                    node = node.next


                # 一文のなかの全ての助詞が [MASK] でおきかわった文
                # text = ' '.join(words)
                # print(text)
                # ここってスペースいらないのか？
                text = ''.join(words)

                # [MASK] に置き換えるのは一文につき一箇所にしたいので，一つずつ置き換え直す
                text = text.replace('[MASK]', '{}')



                # [MASK] 部分が何であったかはこれで確認できる
                # print(' '.join(zyosis))


                # ここの文法について
                # for 変数1　変数2　in enumerate(オブジェクト):
                # for 文でリストやタプルなどのオブジェクトから，要素とインデックスを同時に取得している
                # ここでは i にインデックス（1, 2, 3...），zyo に要素（'は', 'が', 'で'...）が格納される
                for i, meishi in enumerate(replaces):

                    # リストのスライスは，[start:stop]で記述し，start <= x < stop の範囲が選択される
                    # 両方とも省略した場合は全ての値が選択される
                    # zyosis_tmp に zyosis をコピーしている
                    replaces_tmp = replaces[:]

                    # i 番目を [MASK] に置き換える
                    # これをループするので，[MASK] が一箇所ずつずれた文を取得できる
                    replaces_tmp[i] = '[MASK]'
                    # print(zyosis_tmp)


                    # ここの文法（アンパック）について
                    # 関数を呼び出すときのアスタリスクは，引数としてリストやタプルなどを分解して渡すことを意味している
                    # つまり，print(x[0], x[1], x[2]...) と同義
                    # print(zyosis_type[i], text.format(*zyosis_tmp))
                    # text の中の {} に名詞のリスト（一部 [MASK]）をいれていく
                    tmp_str = is_meishi[i] + '\t' + text.format(*replaces_tmp)
                    f_w.writelines(tmp_str + '\n')

        # 書き込み用ファイルを閉じる
        f_w.close()


print('--- oshi-mai! ---')


# データ多すぎひん？？？
#
# --- kaiseki suruyo ---
# --- mai2008_01.txt wo hiraite iruyo ---
# --- 1 ha 227878 attayo! ---
# --- mai2008_02.txt wo hiraite iruyo ---
# --- 1 ha 118718 attayo! ---
# --- mai2008_03.txt wo hiraite iruyo ---
# --- 1 ha 135890 attayo! ---
# --- mai2009_01.txt wo hiraite iruyo ---
# --- 1 ha 233147 attayo! ---
# --- mai2009_02.txt wo hiraite iruyo ---
# --- 1 ha 112283 attayo! ---
# --- mai2009_03.txt wo hiraite iruyo ---
# --- 1 ha 137181 attayo! ---
# --- mai2010_01.txt wo hiraite iruyo ---
# --- 1 ha 238469 attayo! ---
# --- mai2010_02.txt wo hiraite iruyo ---
# --- 1 ha 113966 attayo! ---
# --- mai2010_03.txt wo hiraite iruyo ---
# --- 1 ha 134450 attayo! ---
# --- mai2011_01.txt wo hiraite iruyo ---
# --- 1 ha 243442 attayo! ---
# --- mai2011_02.txt wo hiraite iruyo ---
# --- 1 ha 133091 attayo! ---
# --- mai2011_03.txt wo hiraite iruyo ---
# --- 1 ha 135849 attayo! ---
# --- mai2012_01.txt wo hiraite iruyo ---
# --- 1 ha 252011 attayo! ---
# --- mai2012_02.txt wo hiraite iruyo ---
# --- 1 ha 161721 attayo! ---
# --- mai2012_03.txt wo hiraite iruyo ---
# --- 1 ha 129966 attayo! ---
# --- oshi-mai! ---
#
