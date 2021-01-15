
# ラベルごとのデータの数を等しくします
# 回す前にファイル名 4 箇所確認

import numpy as np
import random
import collections


print('--- kaiseki suruyo ---')

# 各ラベルのデータの数
NUM = 5000


# ファイルの準備をしてね
path_r = './dataset_mask/meishi_number/mai2008_01_mask_meishi_number' + '.txt'
# path_w = './dataset_align/keiyoushi/mai2008_01_mask.txt'



data = []
label = []


with open(path_r, mode='r') as f:
    # これにラベル０の個数，ラベル１の個数，ラベル２の個数... を入れていく
    # sample_nums = np.array([])
    sample_nums = []

    print('--- data wo kakunin suru ne! ---')

    for line in f:

        label_tmp = int(line.split()[0])  # label
        data_tmp = line.split()[1]  # data

        label.append(label_tmp)
        data.append(data_tmp)

    # ラベルの種類を数える
    c = collections.Counter(label)

    # 各ラベルと，そのデータがいくつあるかを表示する
    # print(c)

    for i in range(len(c)):
        # ndarray 全体に対して条件を満たす要素数を数える
        # ここでは label の要素のうち，label が i であるものの数の合計を求めている
        sample_num = sum(x == i for x in label)
        sample_nums.append(sample_num)

    print('--- Number of data for each label:', sample_nums)

    # 全クラス内の最小サンプル数を取得し，NUMのデータがいくつできるか確認
    min_num = min(sample_nums)
    count = min_num//NUM


    for i in range(count):
        filename1 = './dataset_mask/meishi_number/mai2008_01/mai2008_01_' + str(i) + '.txt'

        f_w = open(filename1, mode='w')

        add_list = []

        for j in range(len(sample_nums)):

            indexes = [id for id, x in enumerate(label) if x == j]
            add_indexes = random.sample(indexes, NUM)
            add_indexes.sort(reverse=True)

            for k in add_indexes:
                add_list.append([label.pop(k), data.pop(k)])

        print('--- Number of data for each label:', collections.Counter(row[0] for row in add_list))

        random.shuffle(add_list)

        for j in range(len(add_list)):
            str_tmp = str(add_list[j][0]) + '\t' + add_list[j][1]
            f_w.writelines(str_tmp + '\n')


    # f_w.close()


    filename2 = './dataset_mask/meishi_number/mai2008_01/remain.txt'
    with open(filename2, mode='w') as f_w2:
        for l in range(len(label)):
            str_tmp = str(label[l]) + '\t' + data[l] + '\n'
            f_w2.writelines(str_tmp)


print('= = = = = = = = = =')



print('--- oshi-mai! ---')
