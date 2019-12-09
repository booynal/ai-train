import jieba


def get_content(path):
    with open(path, 'r', encoding='GBK') as file:
        return ' '.join([x.strip() for x in file])

def get_TF(words, topK = 10):
    tf_dic = {}
    for w in words:
        tf_dic[w] = tf_dic.get(w, 0) + 1
    return sorted(tf_dic.items(), key=lambda l:l[1], reverse=True)[:topK]

def stop_words(path):
    with open(path) as file:
        return [l.strip() for l in file]

if __name__ == '__main__':
    file = './data/news/C000008/10.txt'
    stop_word_path = './data/stop_words.utf8'
    user_dic_path = './data/user_dic.utf8.txt'

    jieba.load_userdict(user_dic_path)
    print(list(jieba.cut('快钱来了')))

    sent = get_content(file)
    stop_words = stop_words(stop_word_path)
    split_words = [x for x in jieba.cut(sent) if x not in stop_words]
    print('当前样本:', sent)
    print('样本分词效果:', '/'.join(split_words))
    print('样本topK:', get_TF(split_words))

