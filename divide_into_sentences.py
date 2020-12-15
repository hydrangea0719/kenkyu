# divide_into_sentences.py
# 毎日新聞データセット（csvファイル）から本文だけを取得
# 本文を一文ずつに分ける


import csv
import re
import unicodedata

# dataset の中にあるcsvファイルを読み込みたい（願望）
# 自作関数のパッケージ化と同じように，ディレクトリに mai2008_01.csv と __init__.py を入れてみた
# from dataset import mai2008_01 as maidata
# できんが？？？？？？？？？？？？わからん
# dataset の中に入れるのはスクリプト！！！

csv_file = open("mai2008_01.csv", 'r')

a_list = []
for row in csv.reader(csv_file):
    # 取得したい本文のデータは，csvファイルのC列にある
    a_list.append(row[2])

# 先頭行 "honbun" を削除
del a_list[0]

# csv_file.close()


# 最初に文字の置換をする
# 次に句点で分ける

a_text = ""
for a in a_list:
    # 句点で区切る
    l_sentences = a.split('。')
    a_text += '。\n'.join(l_sentences)

# 文字表記の統一
a_text = unicodedata.normalize('NFKC', a_text)


# .replace().replace() は操作を複数回呼んでいるだけ
# 1：[SEP]タグで改行，2：意味のない文を削除
a_text = a_text.replace('[SEP]', '\n').replace('【現在著作権交渉中の為、本文は表示できません】', '')



# .join()は list を str に変換する

# table を使って置換することもできる
table = str.maketrans({
    '\u3000': '',
    ' ': '',
    '\t': '',
    '◇': '',
    '.': '',
    '■': '',
    '*': '',
    '▲': '。',  # これは句点の代わりに使われているみたい
    '▽': '、'  # これは箇条書きの代わりに使われているみたい
})
a_text = a_text.translate(table)

# # これは全角半角の変換．半角に統一する
# a_text = a_text.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}))
# # FF01 = 全角の ！，FF5E = 全角の 〜
# # 21 = 半角の !，7E = 半角の ~
# # https://ja.wikipedia.org/wiki/Unicode%E4%B8%80%E8%A6%A7_0000-0FFF


# a_text = re.sub(r'\s+', '', a_text)    # これは複数の連続する空白を1つに置換する


# 正規表現で括弧の中身を消す
a_text = re.sub(r'\(.+?\)', '', a_text)
a_text = re.sub(r'【.+?】', '', a_text)


# 一行に着目して strip した結果が空文でないなら，改行記号で split して list 型にする
# それを改行記号を挟みながら str 型にする
a_text = '\n'.join([line for line in a_text.split('\n') if not line.strip() == ''])





#
file_output = open("mai2008_01.txt", 'w')
file_output.writelines(a_text)
file_output.close()

# s = "aaaab"
# s = s.replace("aa", "a")
# print(s)
# >> aab


# csvファイルから本文のみを抽出し，句点区切りで１文ずつに分けることができましたやったね！！！


# （２２面に関連記事）[SEP]　判決を受け、下山判事は法曹資格を失った。
# [SEP] タグも一文とカウントして区切るべき？
# -> とりあえず区切ることにした

# 什〓(じゅうほう)市の化学工場の倒壊現場
# 文字化け？どう対応するか

# 【高瀬浩平、写真・森園道子】<1日1句医者いらず>
# ここらへんなんとかしたい，「」は消せないけれど，それ以外の括弧は消せそう
# 【 で検索をかけると4500件ヒットする
# -> とりあえず消した

# ◆電気料金の試算結果◆
# 電力会社1〜3月5月最大差額
# 北海道67966200〜6400▼596
# 東北68856300〜6500▼585
# 東京72066000〜6400▼1206
# 中部71936500〜6800▼693
# 北陸68266400〜6600▼426
# 関西68716400〜6600▼471
# 中国74366900〜7100▼536
# 四国70016700〜6800▼301
# 九州66286300〜6500▼328
# 沖縄82427400〜7800▼842
# ※標準家庭の1カ月当たりの料金、単位・円。
# 1〜3月は確定値。
# なんやこいつ？？？？？？？？？？

# でもワースト1
# ？？？？？
# [SEP]　◇でもワースト１[SEP]　
# これはブチギレ案件




# https://note.nkmk.me/python-str-remove-strip/
