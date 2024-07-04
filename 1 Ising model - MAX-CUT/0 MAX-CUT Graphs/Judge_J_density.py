'''
UTF-8
用于检查生成的随机MAX-CUT图
'''
# 矩阵计算等尽量使用numpy，优化好，速度快
import numpy as np
	
# 输入待生成MAX-CUT图的nodes(多个)，注意n需要整除2/d.
Ns = [20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
# 输入待生成MAX-CUT图的density(多个)
# 生成[0.1, 0.9]，总数为num=9个的平均取样的数组，即[0.1, 0.2, ..., 0.9]，‘decimals=1’表示组元精确到小数点后一位
Ds = np.round(np.linspace(0.1, 0.9, num=9), decimals=1)
# 每种[n, d]对应生成graphs个不同的随机图
graphs = 10
fn_err = 'The error set (density).txt'

for n in Ns:
	for d in Ds:
		path = 'n=' + str(n) + '/d=' + str(d)
		print(path)
		# 计算并输出edges，用于与后面生成的J对比
		edges = int(0.5*n*(n-1)*d)
		
		# 检查edges
		for g in range(graphs):
			# 读取保存J的.csv文件，分隔符统一用','，命名示例：'n=100/d=0.1/J100-d=0.1-0.csv'
			g0 ='J' + str(n) + '-' + 'd=' + str(d) + '-' + str(g) + '.csv'
			fn_graph_density = path + '/' + g0
			J = np.loadtxt(open(fn_graph_density, 'rb'), delimiter=',', skiprows=0)

			# edge数正确时输出为True。部分情况下会出现错误，比如n不能整除2/d
			count0 = np.count_nonzero(J)
			count = int(count0/2)
			if count != edges:
				# 将错误写入txt文件
				err_edges = 'Edges error: ' + str(g0) + ' .\n'
				print(err_edges)
				with open(fn_err, 'a') as a:
					a.write(err_edges)
		
		for i in range(graphs):
			for j in range(graphs):
				if i > j:
					# 读取文件路径
					gi = 'J' + str(n) + '-' + 'd=' + str(d) + '-' + str(i) + '.csv'
					gj = 'J' + str(n) + '-' + 'd=' + str(d) + '-' + str(j) + '.csv'
					fni = path + '/' + gi
					fnj = path + '/' + gj

					# 读取Ji和Jj文件，‘rb’代表只读
					Ji = np.loadtxt(open(fni, 'rb'), delimiter=',', skiprows=0)
					Jj = np.loadtxt(open(fnj, 'rb'), delimiter=',', skiprows=0)

					# 判断Ji和Jj是否全同
					if (Ji == Jj).all():
						# 将错误写入txt文件
						err_J = 'Identity error:' + str(gi) + ' & ' +str(gj) + '.\n'
						print(err_J)
						with open(fn_err, 'a') as a:
							a.write(err_J)
		
print('This programe is Done.')
