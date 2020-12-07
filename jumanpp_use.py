from pyknp import Juman

jumanpp = Juman()


def how_to_use_jumanpp():
    result = jumanpp.analysis("しかし、彼は来なかった。なので、彼女が来た。")
    for mrph in result.mrph_list():
        print(f"見出し:{mrph.midasi}, 読み:{mrph.yomi}, 原形:{mrph.genkei}, 品詞:{mrph.hinsi}, "
              f"品詞細分類:{mrph.bunrui}, 活用型:{mrph.katuyou1}, 活用形:{mrph.katuyou2}, "
              f"意味情報:{mrph.imis}, 代表表記:{mrph.repname}")
    print(' '.join(mrph.midasi for mrph in result.mrph_list()))
    ziritugo_type = ['動詞', '名詞', '形容詞']# ect
    ziritugo = []
    huzokugo = []
    words = []
    for mrph in result.mrph_list():
        if mrph.hinsi in ziritugo_type:
            ziritugo.append(mrph.midasi)
            words.append(mrph.midasi)
        else:
            huzokugo.append(mrph.midasi)
            words.append('[MASK]')

    print(' '.join(ziritugo))
    print(' '.join(huzokugo))
    print(' '.join(words))


if __name__ == '__main__':
    how_to_use_jumanpp()

