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
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from fem_string import Femwave
import os
import os.path
import subprocess

#三角形の面積を返す関数
def calc_area(p1, p2, p3):
    #print(p2[0])
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
#外力を入力する点（決め打ち）
input_point = 8 
#フォルダ
foldername = 'body_result'
if not os.path.exists(foldername):
    os.mkdir(foldername)
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
number_of_element = triangles.shape[0]

#volファイルを読み込んでみるテスト
vol_file = open('output_v4.vol', 'r')
#要素のところまで行を進める
while True:
    s = vol_file.readline() 
    if s.find('surfaceelementsgi')!=-1:
        break
vol_file.readline()
triangles_list = [] 
count = 0
while True:
    s = vol_file.readline()
    if s=='\n':
        break
    s1 = s.strip()
    s2 = s1.split()
    #5番目からポイントっぽい
    #print(s2)
    #surfnrで分別してみる（決め打ち）
    print(s2[0])
    if int(s2[0]) == 2:
        triangles_list.append(int(s2[5])-1) 
        triangles_list.append(int(s2[6])-1) 
        triangles_list.append(int(s2[7])-1) 
        count += 1
#print(triangles_list)
triangles = np.asarray(triangles_list)
triangles = np.reshape(triangles, (count,3))
#print(triangles)
#今度は座標
while True:
    s = vol_file.readline() 
    if s.find('points')!=-1:
        break
vol_file.readline()
xy_list = [] 
count = 0
while True:
    s = vol_file.readline()
    if s=='\n':
        break
    s1 = s.strip()
    s2 = s1.split()
    #0:x 1:z 2:y
    #print(s2)
    #一旦タプルで保存
    xy_list.append((float(s2[0]),float(s2[2]))) 
    #xy_list.append(float(s2[2])) 
    count += 1
print(xy_list)
#重複削除
#memo:sortedで順番が変わらないようにしている
xy_list = list(sorted(set(xy_list),key=xy_list.index))
print(xy_list)
xy_list_uni = []
for i in range(len(xy_list)):
    xy_list_uni.append(xy_list[i][0])
    xy_list_uni.append(xy_list[i][1])
xy = np.asarray(xy_list_uni)
xy = np.reshape(xy, (int(len(xy_list_uni)/2),2))
print('<---xy--->')
print(xy)
#節点と要素のそれぞれの数（更新）
number_of_point = xy.shape[0]
number_of_element = triangles.shape[0]
print(number_of_point)
print(number_of_element)

#境界を判別してみるテスト
BC = np.zeros( (number_of_point) )
edge_list = [] 
for i in range(number_of_element):
    for j in range(3):
        edge_list.append([triangles[i][j%3],triangles[i][(j+1)%3]])
print(edge_list)
#重複削除（一度しか出てこない辺＝境界！）
boundary_list = []
flag = 0
print('<-----boundary_list----->')
for i in range(len(edge_list)):
    l = edge_list[i]
    print('<---l--->')
    print(l)
    for j in range(len(edge_list)):
        if ((l[0]==edge_list[j][0] and l[1]==edge_list[j][1]) or (l[0]==edge_list[j][1] and l[1]==edge_list[j][0])) and i!=j:
            print(edge_list[j])
            flag = 1
            break
    if flag == 0:
        boundary_list.append(l)
        print('<---add--->')
        print (boundary_list)
    flag = 0
print (boundary_list)
#境界条件設定
for i in range(number_of_point):
    for j in range(len(boundary_list)):
        if i == boundary_list[j][0] or i == boundary_list[j][1]:
            BC[i] = 1
            break
print(BC)
#テスト終わり
#print(number_of_point)
#print(BC.shape[0])

#各サイズ
size = (number_of_point-np.sum(BC))*2
u_size = number_of_point-np.sum(BC)
v_size = number_of_point-np.sum(BC)
#係数行列
#全体
co_left = np.zeros( (size, size) )
#print(co_left)
co_right = np.zeros( (size, size) )
#固定端|微分なし*微分なし
#print(u_size)
M1 = np.zeros( (number_of_point-np.sum(BC),number_of_point-np.sum(BC)) )
#print(M1)
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
grid_num = str(sqrt(number_of_point))
plt_file = open(output_plt_file_name, 'w')
plt_file.write('#plot.plt\n')
plt_file.write('set xrange [-0.5:0.5]\n')
plt_file.write('set yrange [-0.5:0.5]\n')
plt_file.write('set zrange [-0.01:0.01]\n')
plt_file.write('set terminal png\n')
plt_file.write('set dgrid3d '+ grid_num +','+ grid_num + '\n')
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
#ux_bc = 0.0 
rho_s = 1140.0  
aaa_s = 0.5188e-6 

#uvn[8] = 1.0

#=====係数行列計算=====#
#---各行列についてローカルからグローバルを作る
print('<-----matrix----->')
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
    #print(b)
    #print(c)
    #loc2glb
    #print("loc2glb")
    for i in range(3):
        for j in range(3):
            #今回はどちらも境界条件固定なので境界ならグローバルには足さない
            if BC[triangles[k][i]]==0 and BC[triangles[k][j]]==0: 
                #print("i")
                loc2glb_i = triangles[k][i]-np.sum(BC[0:triangles[k][i]])
                #print(loc2glb_i)
                #print("j")
                loc2glb_j = triangles[k][j]-np.sum(BC[0:triangles[k][j]])
                #print(loc2glb_j)
                M1[loc2glb_i][loc2glb_j] += M1_local[i][j]
                #M1 += M1_local[i][j]
                M2[loc2glb_i][loc2glb_j] += M2_local[i][j]
                #M2 += M2_local[i][j]
#---組み合わせて全体の係数行列を作る
#係数
co1 = D*aaa**2/(rho*2)
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
print('<---co_left--->')
print(co_left)
#---右辺係数行列計算---#
co_right = np.vstack( (np.hstack((1.0/dt*M1, 1.0/2.0*M1)), np.hstack((-co1*M2, 1.0/dt*M1))) )
print('<---co_right--->')
print(co_right)

cnt = 0
for i in range(number_of_point):
    if BC[i] == 0:
        un[i] = uvn[cnt]
        cnt += 1
#gnuplotでグラフを作成
output_txt_file_name = foldername + '/' + str(graph_num).zfill(4) + '.txt'
f = open(output_txt_file_name, 'w')
for i in range(number_of_point):
    f.write(str(xy[i][0]) + ' ' + str(xy[i][1]) + ' ' + str(un[i]) + '\n')
f.close
plt_file.write('set output"' + str(graph_num).zfill(4) + '.png"\n')
plt_file.write('splot "' + str(graph_num).zfill(4) + '.txt" w lp\n')
graph_num += 1

#===mainloop===#
print('<-----mainloop----->')
while step*dt<tmax:
    #print('#'+str(step))
    #---1.ボディの更新,弦の境界決定---#
    #右辺計算
    #外力処理:input_pointのところに入力
    ex_f[input_point-np.sum(BC[0:input_point])] = -tension/(rho*aaa)*ux_bc 
    right = np.zeros( (size) )
    right = np.dot(co_right,uvn) + ex_f
    #解く
    uvn = np.linalg.solve(co_left,right)
    #弦の境界設定
    #print(uvn)
    #print(uvn[input_point-np.sum(BC[0:input_point])])
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
    #print(un)
    if step%1000==0:
        print('#'+str(step))
        print(un)
        #gnuplotでグラフを作成
        output_txt_file_name = foldername + '/' + str(graph_num).zfill(4) + '.txt'
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

#ffmpeg命令
cmdstring4 = ('gnuplot', 'plot.plt')
cmdstring = ('ffmpeg', '-r', '60', '-i', '%04d.png',
        '-qscale', '0', '-y', 'out.avi')
cmdstring2 = ('ffmpeg', '-i', 'out.avi',
        '-i', 'result.wav', '-y', 'result.avi')
cmdstring3 = ('ffmpeg', '-i', 'result.avi',
        '-movflags', 'faststart', '-vcodec', 'libx264',
        '-acodec', 'copy', '-y', 'result.mp4')
#'libmp3lame', '-ac', '1', '-ar', '44100',
        #'-ab', '256k', '-y', 'result.mp4')
os.chdir(foldername)
p = subprocess.Popen(cmdstring4)#, shell=True)
p.wait()
#subprocess.call("rm result.avi", shell=True)
p = subprocess.Popen(cmdstring)#, shell=True)
p.wait()
#p.kill()
#p = subprocess.Popen(cmdstring2)#, shell=True)
#p.wait()
#p.kill()
#p = subprocess.Popen(cmdstring3)#, shell=True)
#p.wait()
#p.kill()
os.chdir("..")
