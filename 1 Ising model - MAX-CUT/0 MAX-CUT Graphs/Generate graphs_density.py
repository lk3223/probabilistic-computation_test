'''
UTF-8
用于生成给定[nodes, density]的随机MAX-CUT图
'''
# 引入jit用于即时编译just-in-time，支持循环和大部分numpy函数，不支持scipy等高级库
from numba import jit
# 矩阵计算等尽量使用numpy，优化好，速度快
import numpy as np
# 创建文件夹用
import os
# 计算程序运行时间
import time

# 随机生成新的MAX-CUT coupling matrix J，n为节点数nodes，d为图密度density
@jit(nopython=True, parallel=True)
def random_J(n, d):
	
	# 计算MAX-CUT图的边数edges
	edges = int(0.5*n*(n-1)*d)
	# 计算MAX-CUT图density=1时的总边数(全连接时)
	edges0 = int(0.5*n*(n-1))
	# 从np.array(elements)中生成size为edges的随机数组，choice0中的元素不重复。即从[0, 1, 2, ..., elements-1]中随机选取edges个元素，排列顺序也是随机的。
	choice0 = np.random.choice(edges0, size=edges, replace=False)
	# 从小到大给choice0中元素排序
	choice1 = np.sort(choice0)
	# 创建矩阵J
	J = np.zeros((n, n))
	
	# 给矩阵J随机赋值edges个矩阵元=1(i>j)
	for i in range(n):
		#计算矩阵第i行开头(a)和结尾(b-1)的index(转化为size=edges0的一维数组)
		a = int(0.5*i*(i-1))
		b = int(0.5*i*(i+1))

		# 读取choice1中的元素，当 a <= c < b时，赋值对应的矩阵元
		for c in choice1:
			if (c >= a) & (c < b):
				# 计算列index j
				j = int(c - a)
				J[i][j] = 1
	
	# 将矩阵J对称化
	for i in range(n):
		J[i][i] = 0
		for j in range(n):
			if j < i:
				J[j][i] = J[i][j]

	return J

# 计算矩阵J中的edge数，用于检查density是否正确
@jit(nopython=True)
def edge_J(J):
	count0 = np.count_nonzero(J)
	count = int(count0/2)

	return count

# 创建新文件夹
def mkdir(path):
	folder = os.getcwd() + '/' + path
	print(folder)
	if not os.path.exists(path):
		os.makedirs(folder)
	else:
		pass

# 输入待生成MAX-CUT图的nodes(多个)，注意n需要整除2/d.
Ns = [20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
# 输入待生成MAX-CUT图的density(多个)
# 生成[0.1, 0.9]，总数为num=9个的平均取样的数组，即[0.1, 0.2, ..., 0.9]，‘decimals=1’表示组元精确到小数点后一位
Ds = np.round(np.linspace(0.1, 0.9, num=9), decimals=1)
# 每种[n, d]对应生成graphs个不同的随机图
graphs = 10

t0 = time.time()

for n in Ns:
	for d in Ds:
		# 创建新文件夹
		path = '1n=' + str(n) + '/d=' + str(d)
		mkdir(path)
		print(path)
		# 计算并输出edges，用于与后面生成的J对比
		edges = int(0.5*n*(n-1)*d)

		for g in range(graphs):
			# 生成矩阵J
			J = random_J(n, d)

			# 将J保存为.csv文件，分隔符统一用','，命名示例：'n=100/d=0.1/J100-d=0.1-0.csv'
			fn_graph = path + '/J' + str(n) + '-' + 'd=' + str(d) + '-' + str(g) + '.csv'
			np.savetxt(fn_graph, J, fmt='%d', delimiter=',')

			# edge数正确时输出为True。部分情况下会出现错误，比如n不能整除2/d
			count = edge_J(J)
			if count != edges:
				print(count == edges)
		
print('This programe is Done.')

# 计算程序运行时间
t1 = time.time()
T0 = t1 - t0
# 将程序运行时间写入txt文件
word0 = 'The generated time (njit) is: ' + str(T0) + ' s.\n'
fn_time = 'The generated time.txt'
with open(fn_time, 'a') as a:
	a.write(word0)
