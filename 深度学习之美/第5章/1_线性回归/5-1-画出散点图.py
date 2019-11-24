import matplotlib.pyplot as plt
dataset= [[1.2, 1.1], [2.4, 3.5], [4.1, 3.2], [3.4, 2.8], [5, 5.4]]
x = [row[0] for row in dataset] # 取出x轴列表
y = [row[1] for row in dataset] # 取出y轴列表
plt.axis([0, 6, 0, 6])
plt.plot(x, y, 'bs')
plt.grid()
plt.show()
# plt.savefig('scatter.png')
