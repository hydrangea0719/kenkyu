
# format_text.py
# 手作業でいじったテキストファイルの空白行とかを取り除くだけ


import csv
import re
import unicodedata

# バックアップを取るのに必要
import shutil


# dataset の中にあるcsvファイルを読み込みたい（願望）
# 自作関数のパッケージ化と同じように，ディレクトリに mai2008_01.csv と __init__.py を入れてみた
# from dataset import mai2008_01 as maidata
# できんが？？？？？？？？？？？？わからん
# dataset の中に入れるのはスクリプト！！！

tmp1 = ['08', '09', '10', '11', '12']
tmp2 = ['01', '02', '03']


for i in tmp1:
    for j in tmp2:

        input_file = 'mai20' + i + '_' + j + '.txt'

        # 念の為バックアップを取る
        backname = input_file + '.bak'
        shutil.copy(input_file, backname)


        a_text = ""

        with open(input_file) as f:
            for a in f:
                if not a.strip() == '':
                    a_text += a


        # table を使ってとりこぼしを置換する
        table = str.maketrans({
            '\u3000': '',
            ' ': '',
            '\t': '',
            '◇': '',
            '.': '',
            '■': '',
            '*': '',
            '◆': '',
            '▲': '。',  # これは句点の代わりに使われているみたい
            '▽': '、'  # これは箇条書きの代わりに使われているみたい
        })

        a_text = a_text.translate(table)


        with open(input_file, mode='w') as f:
            f.writelines(a_text)
