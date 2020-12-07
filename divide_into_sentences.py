# divide_into_sentences.py
# 毎日新聞データセット（csvファイル）から本文だけを取得
# 本文を一文ずつに分ける


import csv

# dataset の中にあるcsvファイルを読み込みたい（願望）
# 自作関数のパッケージ化と同じように，ディレクトリに mai2008_01.csv と __init__.py を入れてみた
# from dataset import mai2008_01 as maidata
# できんが？？？？？？？？？？？？わからん

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
    sentences = a.split('。')
    a_text += '。\n'.join(sentences)
    # a_text += "\n"

file_output = open("mai2008_01.txt", 'w')
file_output.writelines(a_text)
file_output.close()


# csvファイルから本文のみを抽出し，句点区切りで１文ずつに分けることができましたやったね！！！

