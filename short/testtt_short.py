
# 金田さんにもらったやつ
# 文章を入力して形態素解析(node)
# それを1つずつ判定

# Command を押しながらリンクをクリックするとブラウザで開いてくれるよ嬉しいね
# https://medium.com/@jiraffestaff/mecabrc-%E3%81%8C%E8%A6%8B%E3%81%A4%E3%81%8B%E3%82%89%E3%81%AA%E3%81%84%E3%81%A8%E3%81%84%E3%81%86%E3%82%A8%E3%83%A9%E3%83%BC-b3e278e9ed07
# https://qiita.com/ekzemplaro/items/c98c7f6698f130b55d53
# 追加辞書は任意


import MeCab
# from my_function import divide_into_sentences.py


# 適切な辞書を入れてください未来の俺
tagger = MeCab.Tagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')

# input = '今週末は金剛山に登りに行って、そのあと新世界かどっかで打ち上げしましょ。'
# result = tagger.parse(input)
# print(result)
# node = tagger.parseToNode(input)
# input をファイルの形式にする必要がある


# path は絶対パスか相対パスで指定する（これ str のファイル名じゃなかったんやな...）
path_r = './mai2008_01_short.txt'
path_w = './mai2008_01_short_mask_kakujoshi.txt'

    # 書き込み用ファイルを開く
f_w = open(path_w, mode='w')


    # with ブロックを使ってファイルを開くと，ブロックの終了時に自動的にファイルがクローズされる
with open(path_r) as f:
    for line in f:

            # 形態素解析の結果を表示する（parse：解析する）
            # MeCab::Tagger というクラスのインスタンスを生成し，.parse のメソッドを呼ぶことで，解析結果を文字列として取得する（.parseToString も同様）
        result = tagger.parse(line)

            # .parseToNode() は，引数に str 型をとって，MeCab::Node 型を返す
            # MeCab::Node 型は，形態素の情報を持つ構造体
        node = tagger.parseToNode(line)

            # リストを準備する
            # 全ての助詞を [MASK] でおきかえた後の文を入れる
        words = []
            # 助詞（は，が，を...）だけを入れる
        zyosis = []
            # 助詞の種類（格助詞，接続助詞，副助詞，終助詞の四種類？要確認）を入れる
        zyosis_type = []

        while node:

                # 形態素解析の結果から助詞を取り出す
                # feature は csv で表記された素性情報（MeCab::Node のインスタンスで str 型）
            if node.feature.split(',')[0] == '助詞':

                    # その形態素の品詞が助詞だった場合，[MASK] に置き換える
                words.append('[MASK]')

                    # 抜き出した助詞を格納する
                    # surface は 形態素の文字列情報（MeCab::Node のインスタンスで str 型）
                zyosis.append(node.surface)

                    # 抜き出した助詞が格助詞だった場合に 1(True) とする
                zyosis_type.append('1' if node.feature.split(',')[1] == '格助詞' else '0')

            else:
                    # 助詞じゃなかったらそのまま
                words.append(node.surface)

                # next は 次の形態素へのポインタ （MeCab::Node のインスタンス）
            node = node.next
            # end while


            # 一文のなかの全ての助詞が [MASK] でおきかわった文
        text = ''.join(words)
            # print(text)


            # [MASK] 部分が何であったかはこれで確認できる
            # print(' '.join(zyosis))

            # [MASK] に置き換えるのは一文につき一箇所にしたいので，一つずつ置き換え直す

        text = text.replace('[MASK]', '{}')
            # print(text)

            # ここの文法について
            # for 変数1　変数2　in enumerate(オブジェクト):
            # for 文でリストやタプルなどのオブジェクトから，要素とインデックスを同時に取得している
            # ここでは i にインデックス（1, 2, 3...），zyo に要素（'は', 'が', 'で'...）が格納される
        for i, zyo in enumerate(zyosis):

                # リストのスライスは，[start:stop]で記述し，start <= x < stop の範囲が選択される
                # 両方とも省略した場合は全ての値が選択される
                # zyosis_tmp に zyosis をコピーしている
            zyosis_tmp = zyosis[:]

                # i 番目を [MASK] に置き換える
                # これをループするので，[MASK] が一箇所ずつずれた文を取得できる
            zyosis_tmp[i] = '[MASK]'
                # print(zyosis_tmp)


                # ここの文法（アンパック）について
                # 関数を呼び出すときのアスタリスクは，引数としてリストやタプルなどを分解して渡すことを意味している
                # つまり，print(x[0], x[1], x[2]...) と同義
                # print(zyosis_type[i], text.format(*zyosis_tmp))
            tmp_str = zyosis_type[i] + '\t' + text.format(*zyosis_tmp)
            # print(tmp_str)
            f_w.writelines(tmp_str + '\n')

    # 書き込み用ファイルを閉じる
f_w.close()
