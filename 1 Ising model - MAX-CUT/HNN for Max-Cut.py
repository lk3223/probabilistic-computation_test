'''
UTF-8
这个程序是与p-computing比较传统算法, 不需要在FPGA上执行

Hopfeild Neural Network是一种用于求解组合优化问题的传统非随机算法
作为与Probabilistic computing的对比
'''
# 这是HNN全自动求解版，没有保存迭代轨迹，只保存所求得的最优解
import numpy as np
from numba import jit
import os
import time

# 根据Ii输出m，类似于激活函数
@jit(nopython=True)
def out_m(Ii):

	if Ii <= 0:
		m = 1
	else:
		m = -1

	return m

# 计算MAX-CUT问题的Hamiltonian function，类似于代价函数
@jit(nopython=True)
def H_CUT(M, J, n, edges):
	# 计算Hamiltonian function = -0.5*(edges-0.5*M*J*M)，前面负号用于将问题转化为求解最小值
	H0 = 0
	for i in range(n):
		for j in range(n):
			if i > j:
				H0 += J[i][j] * M[i] * M[j]
	
	H = int(-0.5 * (edges - H0))
	return H

# 随机初始化M
def random_M(n):
	M = np.random.choice([-1, 1], size=n)    # 生成size为n，元素为-1或1的随机数组
	return M

# 创建新文件夹
def mkdir(path):
	folder = os.getcwd() + '/' + path
	if not os.path.exists(path):
		os.makedirs(folder)
	else:
		pass

# 输入[nodes, density, graphs]20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000
Ns = [20, 40, 60, 80, 100]
# 生成[0.1, 1.0)，step为0.1的数组，即[0.1, 0.2, ..., 0.9]，‘decimals=1’表示组元精确到小数点后一位
Ds = [0.5] # np.round(np.arange(0.1, 1.0, 0.1), decimals=1)
graphs = np.arange(0, 10, dtype=int)
# 输入重复运行次数，通常为100次
runtimes = 100

t0 = time.time()

for n in Ns:
	# 输入最大迭代次数，可以根据问题规模n调节
	iterations = n * 20

	for d in Ds:
		# 计算edges
		edges = 0.5*n*(n-1)*d
		for g in graphs:
			# 创建矩阵，用于存储求解得到的Hamiltonian最小值Hmin=Sol[:, 0]，及其对应的{mi}组态Msol
			Sol = np.zeros((runtimes, n+1))
			# J读取路径
			path_load = '0 MAX-CUT Graphs/n=' + str(n) + '/d=' + str(d)
			# 求解结果保存路径
			path_save = '2 HNN solutions/n=' + str(n) + '/d=' + str(d)
			mkdir(path_save)

			# load J
			fn_load = path_load + '/J' + str(n) + '-d=' + str(d) + '-' + str(g) + '.csv'
			J = np.loadtxt(open(fn_load,"rb"), delimiter=",", skiprows=0)

			for rt in range(runtimes):
				# 生成局部变量Hrt，用于存储Hamiltonian更新值
				Hrt = 0
				# 生成局部变量Mrt并随机初始化，用于存储{mi}组态，mi取值为-1或1
				Mrt = random_M(n)
				
				# 开始运行HNN
				c = 0
				for ite in range(iterations):
					# 存储旧数据
					Mmid = Mrt
					
					# 异步更新{mi}
					for i in range(n):
						# 计算第i个自旋的耦合强度Ii
						Ii = np.dot(J[i], Mrt)
						Mrt[i] = out_m(Ii)
					# mi全部更新一次后计算H_CUT
					Hrt = H_CUT(Mrt, J, n, edges)
					
					if Hrt == Sol[rt, 0]:
						if (Mmid == Mrt).all():
							c += 1
						else:
							c = 0
					else:
						c = 0
						# 仅保存目前求解到的最优解，节约内存资源
						if Hrt < Sol[rt, 0]:
							Sol[rt, 0] = Hrt
							for i in range(n):
								Sol[rt, i+1] = Mrt[i]
					
					# 当Mrt连续3次没有变化后，终止ite循环，节省计算时间
					if c == 2:
						break

				# 为了方便随时查看结果，每完成一次rt保存一次。
				# 存储路径和文件名示例：‘2 HNN solutions/n=100/d=0.1/HNN Sol-n=100-d=0.1-0.csv’
				fn_save = path_save + '/HNN Sol-n=' + str(n) + '-d=' + str(d) + '-' + str(g) + '.csv'
				np.savetxt(fn_save, Sol, fmt='%d', delimiter=',')

t1 = time.time()

T0 = t1 - t0
word0 = 'The HNN computing time (njit) is: ' + str(T0) + ' s.\n'
fn_time = 'The HNN computing time.txt'
with open(fn_time, 'a') as a:
	a.write(word0)
