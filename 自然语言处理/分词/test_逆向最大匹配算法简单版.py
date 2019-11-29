'''
该代码参考：《Python自然语言处理实战：核心技术与算法》第3章 规则分词-逆向最大匹配
逆向最大匹配(Reverse Maximum Match, RMM)
    其基本原理与正向最大匹配(MM)算法类似，不同点在于分词切分的方向与MM相反。
    RMM从目标字符串的末端开始匹配扫描，每次取最末端的i个字符（i为词典中最长词的长度）作为匹配字符串，与词典进行匹配，
    若匹配失败，则去掉目标字符串的最前面一个字符，将剩下的字符串继续与词典进行匹配
    若匹配成功，则将目标字符串当作一个词语放入分词结果集
'''


# 逆向最大匹配(Reverse Maximum Match, RMM)
class RMM(object):
    def __init__(self, dict_path, debug = False):
        with open(dict_path, 'r') as file:
            self.dict = set([line.strip() for line in file if line.strip()])
            self.maxDictLength = max([len(x) for x in self.dict])
            self.debug = debug

    def cat(self, text):
        result = []
        current_index = len(text)
        # 遍历整个待分词文本
        while current_index > 0:
            # 从后往前(逆序)处理字符序列
            matched = False
            for size in range(self.maxDictLength, 0, -1):
                if current_index - size < 0:
                    continue
                subText = text[(current_index - size): current_index]
                if self.debug:
                    print('subText: ', subText)
                # 判断该子文本是否在词典中存在
                if subText in self.dict:
                    result.insert(0, subText)
                    matched = True
                    break
            if matched:
                current_index -= size
            else:
                current_index -= 1
        return result


if __name__ == '__main__':
    dict_path = 'dict.txt'
    text = "南京市长江大桥 结婚的和尚未结婚的"
    print('text:', text)

    rmm = RMM(dict_path, True)
    result = rmm.cat(text)
    print('result:', result)
