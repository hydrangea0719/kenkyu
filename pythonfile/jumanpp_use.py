# jumanpp_use.py
# Juman++で形態素解析を行う


from pyknp import Juman


jumanpp = Juman()

def how_to_use_jumanpp():

    result = jumanpp.analysis("しかし、彼は来なかった。なので、彼女が来た。")

    # mrph は morpheme = 形態素のこと
    for mrph in result.mrph_list():
        # pythonのf文字列（フォーマット文字列）を用いて，文字列中の置換フィールドに変数をそのまま指定している
        print(f"見出し:{mrph.midasi}, 読み:{mrph.yomi}, 原形:{mrph.genkei}, 品詞:{mrph.hinsi}, "
              f"品詞細分類:{mrph.bunrui}, 活用型:{mrph.katuyou1}, 活用形:{mrph.katuyou2}, "
              f"意味情報:{mrph.imis}, 代表表記:{mrph.repname}")
    # '間に挿入する文字列'.join([連結したい文字列のリスト])
    # 入力を形態素ごとに空白区切りで表示
    print(' '.join(mrph.midasi for mrph in result.mrph_list()))

    # mrph.hinsi の中から，自立語（[MASK]をかけない語）を指定する
    ziritugo_type = ['動詞', '名詞', '形容詞']# ect
    ziritugo = []
    huzokugo = []
    words = []

    for mrph in result.mrph_list():
        # 各形態素について，自立語であれば ziritugo ，付属語であれば huzokugo のリストに append する
        # mrph.midasi は各形態素そのもののこと
        if mrph.hinsi in ziritugo_type:
            ziritugo.append(mrph.midasi)
            words.append(mrph.midasi)
        else:
            huzokugo.append(mrph.midasi)
            # 付属語だけに [MASK] をかける
            words.append('[MASK]')

    print(' '.join(ziritugo))
    print(' '.join(huzokugo))
    print(' '.join(words))


if __name__ == '__main__':
    how_to_use_jumanpp()
