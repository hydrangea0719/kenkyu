###フォルダ kenkyu の中身
- dataset
	- mai2008_01.csv
	- mai2008_02.csv
	- ...
- bert_helper.py
- divide_into_sentences.py
- jumanpp_use.py
- mai2008_01_segmented.txt
- mai2008_01_short.txt
- make_csv.py
- readme.md
- word_segmentation_by_mecab.py



###dataset 概要  
ファイル名 mai{年(複数の場合は(0_0)的に)}_{ジャンル(複数の場合は(0_0)的に)}
encode utf-8
pandasから使うとき
```
import pandas as pd
pd.read_csv(path, encoding='utf-8', index_col=0)
```
uid, 識別id オリジナルに年度を加えた  
あとはオリジナルのものをcsvにしただけ  
なお, 本文のパラグラフ間は'[SEP]'で区切っている. 

ファイル
mai(2008_2013)_(０１,０２,０３).csv
は2008年から2013年の1，2，3面記事がまとめて記載されていたが，ファイルサイズが大きくて扱いづらかったので
mai2008_01.csv
のように
mai{年}_{ジャンル（恐らく何面記事か）}
に分割した．


###bert_helper.py
###divide_into_sentences.py

###jumanpp_use.py
###mai2008_01_segmented.txt
###mai2008_01_short.txt
###make_csv.py
金田さんにもらったやつ，未解読
恐らく何かの形式のファイルを .csv に変えるのだと思う
###readme.md
このファイル
###word_segmentation_by_mecab.py


###2020年12月21日 多田
