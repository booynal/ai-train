'''
该代码参考：《Python自然语言处理实战：核心技术与算法》第3章 规则分词-双向最大匹配
双向最大匹配(Bi-direction Maximum Match, BDMM)
    是将正向与逆向最大匹配算法匹配的结果进行比较，然后按照最大匹配原则，选取词数切分最少的作为最终结果
'''

from 自然语言处理.分词.test_正向最大匹配算法简单版 import MM
from 自然语言处理.分词.test_逆向最大匹配算法简单版 import RMM


# 双向最大匹配(Bi-direction Maximum Match, BDMM)
class BDMM(object):
    def __init__(self, dict_path):
        with open(dict_path, 'r') as file:
            self.mm = MM(dict_path)
            self.rmm = RMM(dict_path)

    def cat(self, text):
        mm_result = self.mm.cat(text)
        rmm_result = self.rmm.cat(text)
        return mm_result if mm_result.__len__() < rmm_result.__len__() else rmm_result


if __name__ == '__main__':
    dict_path = 'dict.txt'
    text = "南京市长江大桥 结婚的和尚未结婚的"
    print('text:', text)

    rmm = BDMM(dict_path)
    result = rmm.cat(text)
    print('result:', result)
