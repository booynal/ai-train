'''
该代码参考：《Python自然语言处理实战：核心技术与算法》第3章 规则分词-正向最大匹配
正向最大匹配(Maximum Match, MM)
    假定分词词典中的最长词有i个汉字字符，则用目标文本的前i个字符作为一次迭代的临时字符序列，将该序列在字典中匹配
    若匹配失败，则缩短临时字符序列，从什么方向缩短呢？从右往左的方向缩短，即去掉临时字符序列的最后一个字符，对剩下的字符序列继续匹配词典
    若匹配成功，则将该临时字符序列当作一个词输出
'''


# 正向最大匹配(Maximum Match, MM)
class MM(object):
    def __init__(self, dict_path, debug = False):
        with open(dict_path, 'r') as file:
            self.dict = set([line.strip() for line in file if line.strip()])
            self.maxDictLength = max([len(x) for x in self.dict])
            self.debug = debug

    def cat(self, text):
        result = []
        # 遍历整个待分词文本
        current_index = 0
        while current_index < len(text):
            # 从前往后(正序)处理字符序列
            matched = False
            for size in range(self.maxDictLength, 0, -1):
                if current_index + size > len(text):
                    continue
                subText = text[current_index : (current_index + size)]
                if self.debug:
                    print('subText: ', subText)
                # 判断该子文本是否在词典中存在
                if subText in self.dict:
                    result.append(subText)
                    matched = True
                    break
            if matched:
                current_index += size
            else:
                current_index += 1
        return result


if __name__ == '__main__':
    dict_path = 'dict.txt'
    text = "南京市长江大桥 结婚的和尚未结婚的"
    print('text:', text)

    rmm = MM(dict_path, True)
    result = rmm.cat(text)
    print('result:', result)
