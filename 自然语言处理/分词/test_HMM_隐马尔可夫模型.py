'''
该代码参考：《Python自然语言处理实战：核心技术与算法》第3章 统计分词-HMM模型
隐马尔科夫模型 (Hidden Markov Model, HMM)
代码：
https://github.com/nlpinaction/learning-nlp

'''


class HMM(object):
    def __init__(self):
        # 用于存放算法的中间结果，不用每次都训练模型
        self.mode_file = './data/hmm_model.pkl'
        # 状态值集合
        self.state_list = ['B', 'M', 'E', 'S']
        # 用于判断是否需要重新加载模型文件
        self.load_para = False
        pass

    # 用于加载已计算的中间结果，当需要重新训练时，需要初始化清空结果
    def try_load_model(self, loaded: bool):
        if loaded:
            import pickle
            with open(self.mode_file, 'rb') as file:
                self.A_dic = pickle.load(file)
                self.B_dic = pickle.load(file)
                self.Pi_dic = pickle.load(file)
                self.load_para = True
        else:
            # 状态转移概率(状态 -> 状态的条件概率)
            self.A_dic = {}
            # 发射概率(状态 -> 词语的条件概率)
            self.B_dic = {}
            # 状态的初始概率
            self.Pi_dic = {}
            self.load_para = False
        pass

    # 计算转移概率、发射概率以及初始概率
    def train(self, path):
        '''path为语料文件所在的路径'''
        # 重置几个概率矩阵
        self.try_load_model(False)
        # 统计状态出现次数，求P(o)
        count_dic = {}

        # 初始化参数
        def init_parameters():
            for state in self.state_list:
                self.A_dic[state] = {s: 0. for s in self.state_list}
                self.B_dic[state] = {}
                self.Pi_dic[state] = 0.
                count_dic[state] = 0

        def makeLabel(text):
            out_text = []
            if len(text) == 1:
                out_text.append('S')
            else:
                out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']
            return out_text

        init_parameters()
        line_num = -1

        # 观察者集合，主要是字以及标点等
        words = set()
        with open(path, encoding='utf8') as file:
            for line in file:
                line_num += 1
                line = line.strip()
                if not line:
                    continue

                word_list = [w for w in line if w != ' ']
                # 更新字符集合
                words |= set(word_list)

                line_state = [makeLabel(w) for w in line.split()]

                # 为什么要有这句话？
                # assert len(word_list) == len(line_state)

                for k, v in enumerate(line_state):
                    for vv in v:
                        count_dic[vv] += 1
                    if k == 0:
                        # 每个句子的第一个字符的状态，用于计算初始化概率
                        for vv in v:
                            self.Pi_dic[vv] += 1
                    else:
                        # 计算转移概率
                        for line_state_k_ in line_state[k - 1]:
                            for vv in v:
                                self.A_dic[line_state_k_][vv] += 1
                        # 计算发射概率
                        for state_k_ in line_state[k]:
                            self.B_dic[state_k_][word_list[k]] = self.B_dic[state_k_].get(word_list[k], 0) + 1.0

        self.Pi_dic = {k: v * 1.0 / line_num for k, v in self.Pi_dic.items()}
        self.A_dic = {k: {k1: v1 / count_dic[k] for k1, v1 in v.items()} for k, v in self.A_dic.items()}

        # 加1平滑
        self.B_dic = {k:{k1: (v1+1) / count_dic[k] for k1, v1 in v.items()} for k, v in self.B_dic.items()}

        # 序列化
        import pickle
        with open(self.mode_file, 'wb') as file:
            pickle.dump(self.A_dic, file)
            pickle.dump(self.B_dic, file)
            pickle.dump(self.Pi_dic, file)

        return self

    # Veterbi算法的实现，是基于动态规划的一种实现，主要是求最大概率的路径
    # 其输入参数为初始概率、转移概率以及发射概率，加上需要切分的句子
    def veterbi(self, text, states, start_p, trains_p, emit_p):
        V = [{}]
        path = {}
        for y in states:
            V[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]
        for t in range(1, len(text)):
            V.append({})
            newpath = {}

            # 检验训练的发射概率矩阵中是否有该字
            neverSeen = text[t] not in emit_p['S'].keys() \
                        and text[t] not in emit_p['M'].keys() \
                        and text[t] not in emit_p['E'].keys() \
                        and text[t] not in emit_p['B'].keys()
            for y in states:
                # 设置未知字单独成词
                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0
                (prob, state) = max([(V[t - 1][y0] * trains_p[y0].get(y, 0) * emitP, y0) for y0 in states if V[t - 1][y0] > 0])
                V[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath

        if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
            (prob, state) = max([(V[len(text) - 1][y], y) for y in ('E', 'M')])
        else:
            (prob, state) = max([(V[len(text) - 1][y], y) for y in states])

        return (prob, path[state])

    def cut(self, text):
        import os
        if not self.load_para:
            self.try_load_model(os.path.exists(self.mode_file))
            prob, pos_list = self.veterbi(text, self.state_list, self.Pi_dic, self.A_dic, self.B_dic)
            begin, next = 0, 0
            for i, char in enumerate(text):
                pos = pos_list[i]
                if pos == 'B':
                    begin = i
                elif pos == 'E':
                    yield text[begin: i + 1]
                    next = i + 1
                elif pos == 'S':
                    yield char
                    next = i + 1
            if next < len(text):
                yield text[next:]
        pass

if __name__ == '__main__':
    hmm = HMM()
    # 该语料文件太大
    hmm.train('./data/trainCorpus.txt_utf8')

    text = '这是一个非常帮的方案'
    res = hmm.cut(text)
    # TODO 这个分词不对，待排查
    # ['这', '是', '一', '个', '这是一个非', '这是一个非常', '这是一个非常帮', '这是一个非常帮的', '这是一个非常帮的方', '案']
    print(text)
    print(str(list(res)))
