# https://medium.com/@jiraffestaff/mecabrc-%E3%81%8C%E8%A6%8B%E3%81%A4%E3%81%8B%E3%82%89%E3%81%AA%E3%81%84%E3%81%A8%E3%81%84%E3%81%86%E3%82%A8%E3%83%A9%E3%83%BC-b3e278e9ed07
# https://qiita.com/ekzemplaro/items/c98c7f6698f130b55d53
# 追加辞書は任意
import MeCab
#
tagger = MeCab.Tagger("")
input = '今週末は金剛山に登りに行って、そのあと新世界かどっかで打ち上げしましょ。'

result = tagger.parse(input)
# print(result)

node = tagger.parseToNode(input)

words = []
zyosis = []
zyosis_type = []
while node:
    if node.feature.split(',')[0] == '助詞':
        words.append('[MASK]')
        zyosis.append(node.surface)
        zyosis_type.append('1' if node.feature.split(',')[1] == '格助詞' else '0')
    else:
        words.append(node.surface)
    node = node.next

text = ' '.join(words)
print(text)
print(' '.join(zyosis))

text = text.replace('[MASK]','{}')
print(text)
for i, zyo in enumerate(zyosis):
    zyosis_tmp = zyosis[:]
    zyosis_tmp[i] = '[MASK]'
    # print(zyosis_tmp)
    print(zyosis_type[i],text.format(*zyosis_tmp))
