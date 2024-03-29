import jieba


sent = '中文分词是文本处理不可或缺的一步！'

seg_list = jieba.cut(sent, cut_all=True, HMM=True)
print('全模式:', '/'.join(seg_list))

seg_list = jieba.cut(sent, cut_all=False, HMM=True)
print('精确模式:', '/'.join(seg_list))

seg_list = jieba.cut_for_search(sent, HMM=True)
print('搜索引擎模式:', '/'.join(seg_list))
