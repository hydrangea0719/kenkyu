**このファイル概要**  
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
   
