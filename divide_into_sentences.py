# divide_into_sentences.py
# 毎日新聞データセット（csvファイル）から本文だけを取得
# 本文を一文ずつに分ける


import csv

# dataset の中にあるcsvファイルを読み込みたい（願望）
# 自作関数のパッケージ化と同じように，ディレクトリに mai2008_01.csv と __init__.py を入れてみた
# from dataset import mai2008_01 as maidata
# できんが？？？？？？？？？？？？わからん
# dataset の中に入れるのはスクリプト

csv_file = open("mai2008_01.csv", 'r')

a_list = []
for row in csv.reader(csv_file):
    # 取得したい本文のデータは，csvファイルのC列にある
    a_list.append(row[2])

# 先頭行 "honbun" を削除
del a_list[0]

# csv_file.close()


a_text = ""
for a in a_list:
    # 句点で区切る
    l_sentences = a.split('。')
    # l_sentences = [line.strip("[SEP]") for line in l_sentences if line.startswith("[SEP]")]
    a_text += '。\n'.join(l_sentences)
    # a_text += "\n"

# .join() ha list -> str ni henkan

a_text = a_text.replace('[SEP]', '\n').replace('【現在著作権交渉中の為、本文は表示できません】', '')


# .replace().replace() 複数回呼んでいるだけ

table = str.maketrans({
    '\u3000': '',
    ' ': '',
    '\t': '',
    '◇': '',
    '…': '',
    '■': '',
    '▲': '\n',  # kore ha kuten no kawari ni tukawarete-iru mitai
    '▽': '\n'  # kore ha kajou-gaki ?
})

a_text = a_text.translate(table)
a_text = a_text.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}))

a_text = '\n'.join([line for line in a_text.split('\n') if not line.strip() == ''])

#for line in a_text:
#    if line == "\n":
#        line.replace("\n", "")

file_output = open("mai2008_01.txt", 'w')
file_output.writelines(a_text)
file_output.close()


# s = "aaaab"
# s = s.replace("aa", "a")
# print(s)
# >> aab


# csvファイルから本文のみを抽出し，句点区切りで１文ずつに分けることができましたやったね！！！

# （２２面に関連記事）[SEP]　判決を受け、下山判事は法曹資格を失った。
# [SEP] tagu demo kugiru beki?
# 什〓(じゅうほう)市の化学工場の倒壊現場
# moji-bake ?

# strip
# https://note.nkmk.me/python-str-remove-strip/

