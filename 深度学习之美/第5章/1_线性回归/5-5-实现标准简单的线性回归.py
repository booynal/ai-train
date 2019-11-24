from math import sqrt
import matplotlib.pyplot as plt

# 定义数据集
dataset = [[1.2, 1.1], [2.4, 3.5], [4.1, 3.2], [3.4, 2.8], [5, 5.4]]

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
def evaluate_algorithm(dataset, algorithm):
    test_set = list()
    for row in dataset:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predict_set = algorithm(dataset, test_set)

    print('dataset: ', dataset)
    print('test_set: ', test_set)
    print('predict_set: ', predict_set)
    # for i in range(len(predicted)):
    #     print('%.3f, %.3f\t' %(dataset[i][0], predicted[i]))

    actual = [row[-1] for row in dataset]
    predicted = [row[-1] for row in predict_set]
    rmse = rmse_metric(actual, predicted)
    return rmse, predict_set


# 开始计算
rmse, predict_set = evaluate_algorithm(dataset, simple_linear_regression)
print('RMSE = %.3f' % rmse)

x = [row[0] for row in dataset]
y = [row[1] for row in dataset]
x_ = [row[0] for row in predict_set]
y_ = [row[1] for row in predict_set]
plt.axis([0, 6, 0, 6])
plt.plot(x, y, 'b^')
plt.plot(x_, y_, 'ro-')
plt.grid()
plt.show()

