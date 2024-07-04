'''
UTF-8
这个程序用于测试p-computing性能, 不需要在FPGA上执行

极为简单的Probabilistic computing算法
mi按顺序执行异步更新, 耦合系数beta在计算过程中不变
'''
# 这是p-bit全自动求解版，没有保存迭代轨迹，只保存所求得的最优解
from numba import jit
import os
import numpy as np
import time

# 根据Ii输出m，类似于激活函数
@jit(nopython=True)
def out_m(Ii):
	# p-bit建模，其中xc为sigmod函数中心，k为缩放系数
	# P-V关系为：P=1/(1+np.exp(-k*(Vx-xc)))
	# xc = 0.7
	# k = 48.68

	# 先将逻辑输入Ii转化为器件写入电压Vx
	# Vx = 5*Ii/k + xc
	
	# 需要大规模计算，因此计算过程简化，具体可以看async-trace.py
	
	# mi=1的最终写入概率
	x = 5*Ii
	P = 1/(1 + np.exp(-x))

	# 生成0~1的随机数p0
	p0 = np.random.random()
	if p0 < P:
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
	M[n-1] = -1
	return M

# 创建新文件夹
def mkdir(path):
	folder = os.getcwd() + '/' + path
	if not os.path.exists(path):
		os.makedirs(folder)
	else:
		pass

# 输入[nodes, density, graphs]。, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000
Ns = [20, 40, 60, 80, 100]
# 生成[0.1, 1.0)，step为0.1的数组，即[0.1, 0.2, ..., 0.9]，‘decimals=1’表示组元精确到小数点后一位
Ds = np.round(np.arange(0.0, 1.0, 0.1), decimals=1)
graphs = np.arange(0, 10, dtype=int)
# 输入的耦合系数0<=beta<=1即可，beta=0时为纯随机(P=0.5)求解，beta>=1时mi将转变为确定性输出，
Betas = np.round(np.arange(0.0, 1.05, 0.05), decimals=2)
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
			for beta in Betas:
				# 创建矩阵，用于存储求解得到的Hamiltonian最小值Hmin=Sol[:, 0]，及其对应的{mi}组态Msol
				Sol = np.zeros((runtimes, n+1))
				# J读取路径
				path_load = '0 MAX-CUT Graphs/n=' + str(n) + '/d=' + str(d)
				# 求解结果保存路径
				path_save = '3 p-bit solutions/n=' + str(n) + '/d=' + str(d)
				mkdir(path_save)

				# load J
				fn_load = path_load + '/J' + str(n) + '-d=' + str(d) + '-' + str(g) + '.csv'
				J = np.loadtxt(open(fn_load,"rb"), delimiter=",", skiprows=0)

				for rt in range(runtimes):
					# 生成局部变量Hrt，用于存储Hamiltonian更新值
					Hrt = 0
					# 生成局部变量Mrt并随机初始化，用于存储{mi}组态，mi取值为-1或1
					Mrt = random_M(n)

					# 开始运行p-computing
					for ite in range(iterations):
						# 异步更新{mi}，由于解的对称性，可略过最后一个自旋的更新
						for i in range(n-1):
							# 计算第i个自旋的耦合强度Ii
							Ii = -beta*np.matmul(J[i], Mrt)
							Mrt[i] = out_m(Ii)
						# {mi}更新后计算H_CUT
						Hrt = H_CUT(Mrt, J, n, edges)

						# 仅保存目前求解到的最优解，节约内存资源
						if Hrt < Sol[rt, 0]:
							Sol[rt, 0] = Hrt
							Sol[rt, 1:] = Mrt

					# 为了方便随时查看结果，每完成一次rt保存一次。
					# 存储路径和文件名示例：‘3 p-bit solutions/n=100/d=0.1/p-bit Sol-n=100-d=0.1-0-beta=0.1.csv’
					fn_save = path_save + '/p-bit Sol-n=' + str(n) + '-d=' + str(d) + '-' + str(g) + '-beta=' + str(beta) + '.csv'
					np.savetxt(fn_save, Sol, fmt='%d', delimiter=',')

t1 = time.time()

T0 = t1 - t0
word0 = 'The p-computing time is: ' + str(T0) + ' s.\n'
fn_time = 'The p-computing time.txt'
with open(fn_time, 'a') as a:
	a.write(word0)
