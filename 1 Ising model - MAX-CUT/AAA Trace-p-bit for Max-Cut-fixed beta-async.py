'''
UTF-8
这个程序可以作为FPGA编程的参考

极为简单的Probabilistic computing算法
mi按顺序执行异步更新, 耦合系数beta在计算过程中不变
'''
# 这是p-bit迭代轨迹版，用于看迭代轨迹、画示例图等，使用时建议固定MAX-CUT图[n, d, g]
from numba import jit
import os
import numpy as np
# 画图用
import matplotlib.pyplot as plt

# 根据Ii输出m，类似于激活函数
def out_m(Ii):
	# p-bit建模，其中xc为sigmod函数中心，k为缩放系数
	# P-V关系为：P=1/(1+np.exp(-k*(Vx-xc)))
	xc = 0.7
	k = 48.68

	# 先将逻辑输入Ii转化为器件写入电压Vx
	Vx = 5*Ii/k + xc
	# 设置Vx上下限，保护器件
	if Vx > (xc + 6/k):
		Vx = xc + 6/k
	elif Vx < (xc - 6/k):
		Vx = xc - 6/k
	# 假设写入电压四舍五入精确到0.01V
	Vx = np.round(Vx, decimals=2)

	# 以下直到return部分实际上由自旋p-bit器件执行
	
	# mi=1的最终写入概率
	x = k*(Vx-xc)
	P = 1/(1 + np.exp(-x))         # 可将这一行替换为更贴合p-bit器件的建模
	P = np.round(P, decimals=1)    # 假设概率P四舍五入精确到0.1

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

# 输入[nodes, density, graphs]。
Ns = [100]
# 生成[0.1, 1.0)，step为0.1的数组，即[0.1, 0.2, ..., 0.9]，‘decimals=1’表示组元精确到小数点后一位
Ds = [0.5]    # np.round(np.arange(0.1, 1.0, 0.1), decimals=1)
graphs = np.arange(0, 10, dtype=int)
# 输入的耦合系数0<=beta<=1即可，beta=0时为纯随机(P=0.5)求解，beta>=1时mi将转变为确定性输出，
Betas = np.round(np.arange(0.0, 1.05, 0.05), decimals=2)
# 输入重复运行次数，看轨迹的话runtimes小一点
runtimes = 5

# load Global_mins (n=20-100, d=0.5)
fn_min = '1 Optimal solutions/Global_min n=20-100 d=0.5.csv'
Global_mins = np.loadtxt(open(fn_min,"rb"), delimiter=",", skiprows=0, dtype=int)

for n in Ns:
	# 输入最大迭代次数，可以根据问题规模n调节
	iterations = int(20*n)
	# 生成iteration index数组
	ite_index = np.arange(iterations, dtype=int) + 1
	# 将一维数组转换为二维数组的一行，-1 表示自动计算该维度的大小
	ite_index_row = ite_index.reshape(1, -1)

	for d in Ds:
		# 计算edges
		edges = 0.5*n*(n-1)*d
		for g in graphs:
			# 提取global_min
			global_min = Global_mins[int(n/20-1)][g]
			for beta in Betas:
				# J读取路径
				path_load = '0 MAX-CUT Graphs/n=' + str(n) + '/d=' + str(d)
				# 求解结果保存路径，可以灵活设置
				path_save = '3.1 p-bit trace/n=' + str(n) + '/g=' + str(g)
				mkdir(path_save)

				# load J
				fn_load = path_load + '/J' + str(n) + '-d=' + str(d) + '-' + str(g) + '.csv'
				J = np.loadtxt(open(fn_load,"rb"), delimiter=",", skiprows=0)

				# 生成矩阵H_trace，用于存储Hamiltonian更新轨迹
				H_trace = np.zeros((runtimes, iterations))
				# 调整图片长和宽(单位：英寸)，同时将figure清零
				plt.figure(figsize=(5, 2.6))
				
				for rt in range(runtimes):
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
						H_trace[rt, ite] = H_CUT(Mrt, J, n, edges)

					# 存储路径和文件名示例：‘3.1 p-bit trace/n=100/p-bit trace-n=100-d=0.1-0-beta=0.1.csv’
					fn_save = path_save + '/p-bit trace-n=' + str(n) + '-d=' + str(d) + '-' + str(g) + '-beta=' + str(beta) + '.csv'
					# 将ite_index和H_trace合并后保存。沿着行方向（axis=0）将一维数组（已转换为二维数组的一行）添加到二维数组中，最后转置.T
					Trace = np.concatenate((ite_index_row, H_trace), axis=0).T
					np.savetxt(fn_save, Trace, fmt='%d', delimiter=',')

					# 画出轨迹并保存图片
					fn_fig = path_save + '/p-bit trace-n=' + str(n) + '-d=' + str(d) + '-' + str(g) + '-beta=' + str(beta) + '.png'

					# 画出trace
					plt.plot(ite_index, H_trace[rt, :], lw=0.5)
					# 加入水平辅助线
					plt.axhline(y=global_min, color='r', linestyle='--')
					# 添加图片题目和x, y轴标签
					plt.title('P-comuputing solutions of 100-node MAX-CUT Graphs')
					plt.xlabel('Iteration')
					plt.ylabel('H_cut')
					# 调整边距
					plt.subplots_adjust(left=0.17, bottom=0.19)
					# 保存图片(dpi：每英寸像素点)
					plt.savefig(fn_fig, dpi=600)
				plt.close()
