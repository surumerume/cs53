#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#=======================
#ボディを解くための2次元有限要素法プログラム
#簡単な波動方程式を解いてみる
#ut=v
#vt=uxx
#u:境界でゼロ固定
#12 13 14 15 
# 6  7  8 11
# 3  4  5 10
# 0  1  2  9
#4に外力を入れてみようの巻
###########################
#Todo list
#-グラフから自動動画生成 プロセスからバッチファイルを叩く
#-メッシュ読み込み stlファイルから読み込む？
#--境界判定および境界の区分け
###########################
import numpy as np
from math import fabs
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from fem_string import Femwave

#三角形の面積を返す関数
def calc_area(p1, p2, p3):
    print(p2[0])
    return (1.0/2.0)*fabs( (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) )

#=====初期設定=====#
t=0.0 #初期時刻
tmax=1.0 #終了時刻
rate=44100 #サンプリングレート
dt=1.0/rate #時間刻み
step=0 #現在のステップ
fps = 60 
graph_num = 0
#ボディの係数
rho = 350
aaa = 2.9e-3
D = 100e6
R=7.0
input_point = 4
#***あとでメッシュデータから読み込むinitのようなメソッドを作る
#ポイント行列 節点番号と座標
xy = np.array( [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
            [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], 
            [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
            [3.0, 0.0], [3.0, 1.0], [3.0, 2.0],
            [0.0, 3.0], [1.0, 3.0], [2.0, 3.0], [3.0, 3.0]] )
#連結性行列
triangles = np.array( [[0,1,3], [1,4,3], [4,1,2], [2,5,4], [3,4,6], [4,7,6], [4,5,7], [7,5,8],
            [2,9,5], [9,10,5], [5,10,8], [8,10,11], [6,7,12], [12,7,13], [7,8,13], [13,8,14], [8,11,14], [14,11,15]])
#境界にあるかどうかの行列
BC = np.array( [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1] )
#節点と要素のそれぞれの数
number_of_point = xy.shape[0]
print(number_of_point)
print(BC.shape[0])
number_of_element = triangles.shape[0]
#各サイズ
size = (number_of_point-np.sum(BC))*2
u_size = number_of_point-np.sum(BC)
v_size = number_of_point-np.sum(BC)
#係数行列
#全体
co_left = np.zeros( (size, size) )
print(co_left)
co_right = np.zeros( (size, size) )
#固定端|微分なし*微分なし
print(u_size)
M1 = np.zeros( (number_of_point-np.sum(BC),number_of_point-np.sum(BC)) )
print(M1)
#固定端|微分あり*微分あり
M2 = np.zeros( (number_of_point-np.sum(BC),number_of_point-np.sum(BC)) )
#外力
ex_f = np.zeros( (size) )
#ベクトル
uvn = np.zeros(size)
#uvn[0] = 1.0
un = np.zeros(number_of_point) #グラフ書き出し用
right = np.zeros(size)
#グラフ設定用
output_plt_file_name = 'body_result/plot.plt'
plt_file = open(output_plt_file_name, 'w')
plt_file.write('#plot.plt\n')
plt_file.write('set xrange [0:3]\n')
plt_file.write('set yrange [0:3]\n')
plt_file.write('set zrange [-0.05:0.05]\n')
plt_file.write('set terminal png\n')
plt_file.write('set dgrid3d 4,4\n')
plt_file.write('set hidden3d\n')
#value_lim = np.linspace(-0.001, 0.001, 30, endpoint=True) 

#===弦の初期設定（とゼロステップ目出力）===#
fem = Femwave("result",rate,0.65,1140,0.5188e-6,
        60.97,8.1e-7,6.4e-4,5.4e9,0.171e-12)
#エネルギー未対応
#fem.calc_energy()
#fem.calc_tension()
fem.set_fps(fps)
fem.output_txt_result()
fem.make_u_graph()
#張力
tension = 60.97
#tension = sin(pi/360*step)
#uxの境界条件取得
ux_bc = fem.get_ux_bc()
#ux_bc = sin(pi/360*step)

#=====係数行列計算=====#
#---各行列についてローカルからグローバルを作る
for k in range(number_of_element):
    print("element")
    print(k)
    area = calc_area(xy[triangles[k][0]],xy[triangles[k][1]],xy[triangles[k][2]])
    print("area")
    print(area)
    #ローカル要素行列計算
    #b=[y(1)-y(2),y(2)-y(0),y(0)-y(1)]/(2*area)
    #c=[x(2)-x(1),x(0)-x(2),x(1)-x(0)]/(2*area)
    b = np.array( [ xy[triangles[k][1]][1] - xy[triangles[k][2]][1], xy[triangles[k][2]][1] - xy[triangles[k][0]][1], xy[triangles[k][0]][1] - xy[triangles[k][1]][1] ] ) /(2.0*area)
    c = np.array( [ xy[triangles[k][2]][0] - xy[triangles[k][1]][0], xy[triangles[k][0]][0] - xy[triangles[k][2]][0], xy[triangles[k][1]][0] - xy[triangles[k][0]][0] ] ) /(2.0*area)
    M1_local = area/2.0 * np.array( [[2,1,1], [1,2,1], [1,1,2]] )
    M2_local = area/3.0 * np.array( [[b[0]*b[0]+c[0]*c[0], b[0]*b[1]+c[0]*c[1], b[0]*b[2]+c[0]*c[2]],
                                    [b[1]*b[0]+c[1]*c[0], b[1]*b[1]+c[1]*c[1], b[1]*b[2]+c[1]*c[2]],
                                    [b[2]*b[0]+c[2]*c[0], b[2]*b[1]+c[2]*c[1], b[2]*b[2]+c[2]*c[2]]] )
    print(b)
    print(c)
    #loc2glb
    print("loc2glb")
    for i in range(3):
        for j in range(3):
            #今回はどちらも境界条件固定なので境界ならグローバルには足さない
            if BC[triangles[k][i]]==0 and BC[triangles[k][j]]==0: 
                print("i")
                loc2glb_i = triangles[k][i]-np.sum(BC[0:triangles[k][i]])
                print(loc2glb_i)
                print("j")
                loc2glb_j = triangles[k][j]-np.sum(BC[0:triangles[k][j]])
                print(loc2glb_j)
                M1[loc2glb_i][loc2glb_j] += M1_local[i][j]
                #M1 += M1_local[i][j]
                M2[loc2glb_i][loc2glb_j] += M2_local[i][j]
                #M2 += M2_local[i][j]
#---組み合わせて全体の係数行列を作る
#係数
co1 = D*aaa**2/(rho*dt)
#---co_matrix---
#---co_left---[u(n+1) v(n+1)]
# u(n+1)(固) v(n+1)(固)
# | (1/dt)M1  (-1/2)M1 |
# |  co1*M2   (1/dt)M1 |
#---co_right---[u(n) v(n)]
#   u(n)(固)  v(n)(固)
# | (1/dt)M1   (1/2)M1 |
# | -co1*M2   (1/dt)M1 |
#---左辺係数行列計算---#
#memo:hstack=列結合,vstack=行結合
co_left = np.vstack( (np.hstack((1.0/dt*M1, -1.0/2.0*M1)), np.hstack((co1*M2, 1.0/dt*M1))) )
#---右辺係数行列計算---#
co_right = np.vstack( (np.hstack((1.0/dt*M1, 1.0/2.0*M1)), np.hstack((-co1*M2, 1.0/dt*M1))) )

#===mainloop===#
while step*dt<tmax:
    print('#'+str(step))
    #---1.ボディの更新,弦の境界決定---#
    #右辺計算
    #外力処理:今回は4に入れてみる
    ex_f[input_point-np.sum(BC[0:input_point])] = -1.0/(rho*aaa)*tension*ux_bc 
    right = np.zeros( (size) )
    right = np.dot(co_right,uvn) + ex_f
    #解く
    uvn = np.linalg.solve(co_left,right)
    #弦の境界設定
    print(uvn)
    print(uvn[input_point-np.sum(BC[0:input_point])])
    fem.set_un_bc(uvn[input_point-np.sum(BC[0:input_point])])
    #---2.弦を1ステップ進める---#
    step = fem.simulate_one_step()
    #step += 1
    #境界更新
    ux_bc = fem.get_ux_bc()
    #print(uvn[0])
    #---グラフ出力処理---#
    #unをグラフ用に整理
    cnt = 0
    for i in range(number_of_point):
        if BC[i] == 0:
            un[i] = uvn[cnt]
            cnt += 1
    print(un)
    if step%1000==0:
        #gnuplotでグラフを作成
        output_txt_file_name = 'body_result/' + str(graph_num).zfill(4) + '.txt'
        f = open(output_txt_file_name, 'w')
        for i in range(number_of_point):
            f.write(str(xy[i][0]) + ' ' + str(xy[i][1]) + ' ' + str(un[i]) + '\n')
        f.close
        plt_file.write('set output"' + str(graph_num).zfill(4) + '.png"\n')
        plt_file.write('splot "' + str(graph_num).zfill(4) + '.txt" w lp\n')
        graph_num += 1
        '''
        #matplotlibの成れの果て
        for i in range(number_of_element):
            for j in range(3):
                x[i+j] = xy[triangles[i][j]][0]
                y[i+j] = xy[triangles[i][j]][1]
                graph_un[i+j] = un[triangles[i][j]]
        fig = plt.figure()
        ax = Axes3D(fig)
        #ax.plot_surface(x,y,un)
        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(0.0, 3.0)
        ax.set_zlim(-0.05, 0.05)
        #ax.scatter3D(x,y,un)
        ax.plot3D(x,y,graph_un)
        plt.show()
        plt.figure()
        #plt.gca().set_aspect('equal')
        #plt.tricontourf(x, y, triangles, un, cmap=plt.cm.rainbow, norm=plt.Normalize(vmax=0.5,vmin=-5.0))
        plt.tricontourf(x, y, triangles, un, vmin=-0.005, vmax=0.005)
        #plt.tricontourf(x, y, triangles, un)
        plt.colorbar()
        #plt.colorbar(ticks=value_lim)
        #plt.title('Contour plot of user-specified triangulation')
        #plt.xlabel('Longitude (degrees)')
        #plt.ylabel('Latitude (degrees)')
        plt.xlim([-0.5,3.5])
        plt.ylim([-0.5,3.5])
        #plt.zlim([0.0,3.0])
        plt.show()
        '''
