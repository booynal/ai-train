import jieba


def get_content(path):
    with open(path, 'r', encoding='GBK') as file:
        return ' '.join([x.strip() for x in file])

def get_TF(words, topK = 10):
    tf_dic = {}
    for w in words:
        tf_dic[w] = tf_dic.get(w, 0) + 1
    return sorted(tf_dic.items(), key=lambda l:l[1], reverse=True)[:topK]

if __name__ == '__main__':
    file = './data/news/C000008/10.txt'
    sent = get_content(file)
    split_words = list(jieba.cut(sent))
    print('当前样本:', sent)
    print('样本分词效果:', '/'.join(split_words))
    print('样本topK:', get_TF(split_words))

