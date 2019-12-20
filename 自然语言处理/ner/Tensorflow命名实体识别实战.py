'''
该代码来自于 言有三AI-NLP 的微信公众号
《【NLP实战系列】Tensorflow命名实体识别实战》
ref: https://mp.weixin.qq.com/s/AfyT0Dd0WXjlJpLc5yfz2g
创建时间：2019/12
'''

import codecs
import numpy as np
import tensorflow as tf


# 读取训练数据
def load_sentences(path):
    """
    加载训练，测试，验证数据的函数
    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num += 1
        # line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
            else:
                word = line.split()
            assert len(word) == 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


# 标记格式转化
def update_tag_scheme(sentences, tag_scheme):
    """
    将IOB格式转化为BIOES格式。两种模式: iob / iobes
    """
    for i, sentence in enumerate(sentences):
        tags = [w[-1] for w in sentence]
        # 保证语料上BIO格式
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in sentence)
        ##产生异常
        if tag_scheme == 'iob':
            # 转化为BIOES
            for word, new_tag in zip(sentence, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(sentence, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def iob2(tags):
    '''
    :desc: 判断tags是否在iob范围，将不在这个范围的标签用O代替
    :param tags: 待判断的标签集合
    :return: None
    '''
    if not tags:
        return
    iobTags = ['B', 'I', 'O']
    for i in range(len(tags)):
        tag = tags[i].split('-')[0]
        tags[i] = tags[i] if tag in iobTags else 'O'

def iob_iobes(tags):
    '''
    :desc: 将bio标签格式转换为bioes标签格式
    :param tags: 待转换的标签集合
    :return: 新标签集合
    '''
    if not tags:
        return tags
    newTags = tags.copy()
    lastIsI = False
    length = len(tags)
    for i in range(length):
        subTags = newTags[i].split('-')
        currIsI = 'I' == subTags[0]

        if currIsI:
            if lastIsI:
                subfix = newTags[i - 1].split('-')[-1]
                newTags[i - 1] = 'M' + '-' + subfix
            if i == length - 1: # 如果当前是最后一个I，则将I改为E
                subfix = newTags[i].split('-')[-1]
                newTags[i] = 'E' + '-' + subfix
        else:
            if lastIsI:
                subfix = newTags[i - 1].split('-')[-1]
                newTags[i - 1] = 'E' + '-' + subfix
        lastIsI = currIsI
    return newTags


# 构造字典
def char_mapping(sentences, lower):
    # 生成字典和mapping
    chars = [[word[0].lower() if lower else word[0] for word in sentence] for sentence in sentences]
    dico = create_dico(chars)
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (len(dico), sum(len(x) for x in chars)))
    return dico, char_to_id, id_to_char


def create_dico(item_list):
    # 根据传入的列表，生成一个字典
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            dico[item] = dico.get(item, 0) + 1
    return dico


def create_mapping(dico):
    # 生成两个字典，id->word word->id
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


# 模型实现
# 1）word embedding
def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
    """
    :param char_inputs: one-hot encoding of sentence
    :param seg_inputs: segmentation feature
    :param config: wither use segmentation feature
    :return: [1, num_steps, embedding size]
    """
    embedding = []
    with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
        # 生成一个表，用于后面查表
        self.char_lookup = tf.get_variable(
            name="char_embedding",
            shape=[self.num_chars, self.char_dim],
            initializer=self.initializer)

        # 查表，将词向量化

        embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))

    # 分词的维度，可先不关注
    if config["seg_dim"]:
        with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
            self.seg_lookup = tf.get_variable(
                name="seg_embedding",
                shape=[self.num_segs, self.seg_dim],
                initializer=self.initializer)

    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
    embed = tf.concat(embedding, axis=-1)
    return embed


if __name__ == '__main__':
    path = './data.txt'
    sentences = load_sentences(path)
    print('原始sentences:', np.array(sentences))
    tags = ['O', 'B-LOC', 'I-LOC', 'I-LOC', 'O', 'B-ORG', 'I-ORG']
    iob2(tags)
    print('转换为标准BIO格式:', tags)
    print('iob_iobes:', iob_iobes(tags))
    update_tag_scheme(sentences, 'iobes')
    print('sentences2:', np.array(sentences))
    char_mapping(sentences, True)

