from random import seed, randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt


# 导入csv文件
def load_csv(filename):
    dataset = list()
    with open(filename, mode='r') as file:
        csv_reader = reader(file)
        # 读取表头：X, Y
        headings = next(csv_reader)
        print('headings: %s' %headings)
        for row in csv_reader:
            # 判定是否为空行，如果是空行，则忽略，继续处理后面的行
            if not row:
                continue
            dataset.append(row)
    return dataset

# 将字符串转换为浮点数
def str_to_float(dataset):
    try:
        for row_index in range(len(dataset)):
            for col_index in range(len(dataset[row_index])):
                dataset[row_index][col_index] = float(dataset[row_index][col_index].strip())
        return dataset
        # 以上嵌套for等价于以下一行代码(嵌套列表推导式)
        # return [[float(one.strip()) for one in row] for row in dataset]
    except Exception as err:
        print(err)

# 将数据集分割为训练集与测试集两部分
def train_test_split(dataset, train_percent):
    # 训练集与测试集应该不存在交集，所以需要从原始数据集中分离出两个子集合
    train = list()
    train_size = len(dataset) * train_percent
    # 拷贝一份原始数据集，并在拷贝的数据集上做减法操作，减出来的元素当作训练集，剩下的数据就当作是测试集
    test = list(dataset)
    while len(train) < train_size:
        index = randrange(len(test))
        train.append(test.pop(index))
    return train, test

# -----之前定义过的函数--------
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

# 计算求方差的函数
def variance(values, mean):
    return sum([(x - mean) ** 2 for x in values])

# 计算回归系数函数
def coefficients(dataset):
    x = [row[0] for row in dataset] # 取出x轴列表
    y = [row[1] for row in dataset] # 取出y轴列表
    mean_x, mean_y = mean(x), mean(y)
    w1 = covariance(x, mean_x, y, mean_y) / variance(x, mean_x)
    w0 = mean_y - w1 * mean_x
    return w0, w1 # 打包过的匿名元组

# 计算均方根误差
def rmse_metric(actual, predicted):
    sum_error = 0.
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error) ** 2
    # 以上4行等价与以下一行（利用了"列表推导式"）
    # sum_error = sum([(predicted[i] - actual[i]) ** 2 for i in range(len(actual))])
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

# 构建简单线性回归
def simple_linear_regression(train, test):
    w0, w1 = coefficients(train)

    predictions = list()
    for row in test:
        y_model = w1 * row[0] + w0
        predictions.append([row[0], y_model])
    # 以上4行等价与以下一行（利用了"列表推导式"）
    # predictions = [w1 * row[0] + w0 for row in test]
    return predictions


# "后勤函数" 评估算法数据准备及协调
def evaluate_algorithm(dataset, train_percent, algorithm):
    train, test = train_test_split(dataset, train_percent)
    test_without_label = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_without_label.append(row_copy)
    predict = algorithm(train, test_without_label)

    print('dataset: ', dataset)
    print('train: ', train)
    print('test: ', test)
    print('predict: ', predict)

    actual = [row[-1] for row in test]
    predicted = [row[-1] for row in predict]
    rmse = rmse_metric(actual, predicted)
    return rmse, predict


# 开始计算
# 设置随机数种子，为随机数挑选训练和测试数据集做准备
seed(2)
# 导入数据并清洗
filename = 'insurance.csv'
dataset = load_csv(filename)
dataset = str_to_float(dataset)

train_percent = 0.6
rmse, predict_set = evaluate_algorithm(dataset, train_percent, simple_linear_regression)
print('RMSE = %.3f' % rmse)

x = [row[0] for row in dataset]
y = [row[1] for row in dataset]
x_ = [row[0] for row in predict_set]
y_ = [row[1] for row in predict_set]
# plt.axis([0, 400, 0, 400])
plt.plot(x, y, 'bs')    # 第一次，绘制原始数据集的点，用blue的圆点
plt.plot(x_, y_, 'ro-') # 第二次，绘制预测到的数据集，用red的圆点，并用线条链接
plt.grid()
plt.show()

