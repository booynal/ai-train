# 定义求均值函数
def mean(values):
    return sum(values) / float(len(values))

# 计算协方差的函数
def covariance(x, mean_x, y, mean_y):
    # x为列表，mean_x为x列表的均值；y列表与x列表等长，mean_y为y列表的均值
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

# 开始计算均值和协方差
dataset= [[1.2, 1.1], [2.4, 3.5], [4.1, 3.2], [3.4, 2.8], [5, 5.4]]
x = [row[0] for row in dataset] # 取出x轴列表
y = [row[1] for row in dataset] # 取出y轴列表
mean_x, mean_y = mean(x), mean(y)
covar = covariance(x, mean_x, y, mean_y)
print('x均值 = %.3f，y均值 = %.3f，协方差 = %.3f' %(mean_x, mean_y, covar))

