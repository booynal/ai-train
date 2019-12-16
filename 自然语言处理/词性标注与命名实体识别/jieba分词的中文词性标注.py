'''
该代码参考：《Python自然语言处理实战：核心技术与算法》第4章 词性标注与命名实体识别

创建时间：2019/12/15
'''

import jieba.posseg as psg


sent = '中文分词是文本处理不可或缺的一步！'

seg_list = psg.cut(sent)
print(type(seg_list))   # <class 'generator'>
seg_list2 = ['{}/{}'.format(w, t) for w, t in seg_list]
print(seg_list2) # ['中文/nz', '分词/n', '是/v', '文本处理/n', '不可或缺/l', '的/uj', '一步/m', '！/x']
print(', '.join(seg_list2)) # 中文/nz, 分词/n, 是/v, 文本处理/n, 不可或缺/l, 的/uj, 一步/m, ！/x
