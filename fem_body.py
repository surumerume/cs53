#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#=======================
#ボディを解くための2次元有限要素法プログラム
#簡単な波動方程式を解いてみる
#ut=v
#vt=uxx
#u:外側でゼロ固定，ホール側で解放
###########################
#Todo list
#-グラフから自動動画生成 プロセスからバッチファイルを叩く
#-loggingを使ってみる
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
from scipy.sparse import csc_matrix, linalg as sla
import logging
import logging.config
import yaml
import time

logging.config.dictConfig(yaml.load(open("config.yaml").read()))

logger = logging.getLogger('mainloop')
logger.info('program start')

program_start_time = time.time()

#三角形の面積を返す関数
def calc_area(p1, p2, p3):
    #print(p2[0])
    return (1.0/2.0)*fabs( (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) )

#=====初期設定=====#
t=0.0 #初期時刻
tmax=1.0 #終了時刻
rate=44100000 #サンプリングレート
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
input_point = 4 
#フォルダ
foldername = 'body_result'
if not os.path.exists(foldername):
    os.mkdir(foldername)
f_wav = open(foldername + '/wave_log.txt', 'w')
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

#自作のmeshファイルを読み込んでみるテスト
mesh_file = open('guitar_plane_1248.mesh', 'r')
#mesh_file = open('plane02.mesh', 'r')
l = mesh_file.readline().replace('\n','').split(',')
xy = np.asarray(l, dtype=np.float32)
xy = np.reshape(xy, (len(l)/2,2))
l = mesh_file.readline().replace('\n','').split(',')
triangles = np.asarray(l, dtype=np.int32)
triangles = np.reshape(triangles, (len(l)/3,3))
l = mesh_file.readline().replace('\n','').split(',')
BC = np.asarray(l, dtype=np.int32)
mesh_file.close()
logger.debug('xy : \n %s' % np.array2string(xy))
logger.debug('triangles : \n %s' % np.array2string(triangles))
logger.debug('BC : \n %s' % np.array2string(BC))
#節点と要素のそれぞれの数（更新）
number_of_point = xy.shape[0]
number_of_element = triangles.shape[0]
logger.debug('number of point : %d' % number_of_point)
logger.debug('number of element : %d' % number_of_element)

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
plt_file.write('set zrange [-0.1:0.1]\n')
plt_file.write('set terminal png\n')
#plt_file.write('set dgrid3d '+ grid_num +','+ grid_num + '\n')
#plt_file.write('set hidden3d\n')
#value_lim = np.linspace(-0.001, 0.001, 30, endpoint=True) 

#===弦の初期設定（とゼロステップ目出力）===#
#fem = Femwave("result",rate,0.65,1140,0.5188e-6,
#        60.97,8.1e-7,6.4e-4,5.4e9,0.171e-12)
fem = Femwave("result",rate,0.65,1140,0.5188e-6,
        60.97,0.0,0.0,5.4e9,0.171e-12)
#fem = Femwave("result",rate,10.65,1140,0.5188e-6,
#        60.97,0.0,0.0,5.4e9,0.171e-12)
#エネルギー未対応
#fem.calc_energy()
#fem.calc_tension()
fem.set_fps(fps)
fem.output_txt_result()
#fem.make_u_graph()
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
logger.info('start matrix')
for k in range(number_of_element):
    area = calc_area(xy[triangles[k][0]],xy[triangles[k][1]],xy[triangles[k][2]])
    logger.debug('element : %d, area : %f' , k, area)
    #ローカル要素行列計算
    #b=[y(1)-y(2),y(2)-y(0),y(0)-y(1)]/(2*area)
    #c=[x(2)-x(1),x(0)-x(2),x(1)-x(0)]/(2*area)
    b = np.array( [ xy[triangles[k][1]][1] - xy[triangles[k][2]][1], xy[triangles[k][2]][1] - xy[triangles[k][0]][1], xy[triangles[k][0]][1] - xy[triangles[k][1]][1] ] ) /(2.0*area)
    c = np.array( [ xy[triangles[k][2]][0] - xy[triangles[k][1]][0], xy[triangles[k][0]][0] - xy[triangles[k][2]][0], xy[triangles[k][1]][0] - xy[triangles[k][0]][0] ] ) /(2.0*area)
    M1_local = area/12.0 * np.array( [[2,1,1], [1,2,1], [1,1,2]] )
    M2_local = area * np.array( [[b[0]*b[0]+c[0]*c[0], b[0]*b[1]+c[0]*c[1], b[0]*b[2]+c[0]*c[2]],
                                    [b[1]*b[0]+c[1]*c[0], b[1]*b[1]+c[1]*c[1], b[1]*b[2]+c[1]*c[2]],
                                    [b[2]*b[0]+c[2]*c[0], b[2]*b[1]+c[2]*c[1], b[2]*b[2]+c[2]*c[2]]] )
    #print(b)
    #print(c)
    #<loc2glb>
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
csc_co_left = csc_matrix(co_left)
lu = sla.splu(csc_co_left)
logger.debug('<---co_left--->\n%s' % np.array2string(co_left))
#---右辺係数行列計算---#
co_right = np.vstack( (np.hstack((1.0/dt*M1, 1.0/2.0*M1)), np.hstack((-co1*M2, 1.0/dt*M1))) )
logger.debug('<---co_right--->\n%s' % np.array2string(co_right))
logger.info('end matrix')

cnt = 0
for i in range(number_of_point):
    if BC[i] == 0:
        un[i] = uvn[cnt]
        cnt += 1
#gnuplotでグラフを作成
#メッシュ色付け
output_txt_file_name = foldername + '/' + str(graph_num).zfill(4) + '_1.txt'
f = open(output_txt_file_name, 'w')
for i in range(number_of_element):
    ave = (un[triangles[i][0]]+un[triangles[i][1]]+un[triangles[i][2]])/3.0
    f.write(str(xy[triangles[i][0]][0]) + ' ' + str(xy[triangles[i][0]][1]) + ' ' + str(ave) + '\n')
    f.write(str(xy[triangles[i][1]][0]) + ' ' + str(xy[triangles[i][1]][1]) + ' ' + str(ave) + '\n')
    f.write('\n')
    f.write(str(xy[triangles[i][2]][0]) + ' ' + str(xy[triangles[i][2]][1]) + ' ' + str(ave) + '\n')
    f.write(str(xy[triangles[i][1]][0]) + ' ' + str(xy[triangles[i][1]][1]) + ' ' + str(ave) + '\n')
    f.write('\n')
    f.write('\n')
#三角形出力
output_txt_file_name = foldername + '/' + str(graph_num).zfill(4) + '_2.txt'
f = open(output_txt_file_name, 'w')
for i in range(number_of_element):
    for j in range(3):
        f.write(str(xy[triangles[i][j]][0]) + ' ' + str(xy[triangles[i][j]][1]) + ' ' + str(un[triangles[i][j]]) + '\n')
    f.write(str(xy[triangles[i][0]][0]) + ' ' + str(xy[triangles[i][0]][1]) + ' ' + str(un[triangles[i][0]]) + '\n')
    f.write('\n')
    f.write('\n')
'''
for i in range(number_of_point):
    f.write(str(xy[i][0]) + ' ' + str(xy[i][1]) + ' ' + str(un[i]) + '\n')
'''
f.close
plt_file.write('set output"' + str(graph_num).zfill(4) + '.png"\n')
plt_file.write('splot "' + str(graph_num).zfill(4) + '_1.txt" using 1:2:3:4 with pm3d, "' + str(graph_num).zfill(4) + '_2.txt" using 1:2:3 with lines lt -1 \n')
#plt_file.write('splot "' + str(graph_num).zfill(4) + '.txt" w lp\n')
graph_num += 1

#===mainloop===#
logger.info('start mainloop')
mainloop_start_time = time.time()
laptime_start = time.time()
while step*dt<tmax:
    #print('#'+str(step))
    #---1.ボディの更新,弦の境界決定---#
    #右辺計算
    #外力処理:input_pointのところに入力
    ex_f[input_point-np.sum(BC[0:input_point])] = -tension/(rho*aaa)*ux_bc 
    right = np.zeros( (size) )
    right = np.dot(co_right,uvn) + ex_f
    #解く
    uvn = lu.solve(right)
    #uvn = np.linalg.solve(co_left,right)
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
    if step%(rate/44100)==0:
        f_wav.write(str(uvn[input_point+2])+'\n')
    if step%(rate/60)==0:
        #unをグラフ用に整理
        cnt = 0
        for i in range(number_of_point):
            if BC[i] == 0:
                un[i] = uvn[cnt]
                cnt += 1
        #print(un)
        laptime = time.time() - laptime_start
        logger.info('%d #%d laptime:%f',graph_num, step, laptime)
        laptime_start = time.time()
        logger.debug('un\n%s' % np.array2string(un))
        #gnuplotでグラフを作成
        #メッシュ色付け
        output_txt_file_name = foldername + '/' + str(graph_num).zfill(4) + '_1.txt'
        f = open(output_txt_file_name, 'w')
        for i in range(number_of_element):
            ave = (un[triangles[i][0]]+un[triangles[i][1]]+un[triangles[i][2]])/3.0
            f.write(str(xy[triangles[i][0]][0]) + ' ' + str(xy[triangles[i][0]][1]) + ' ' + str(ave) + '\n')
            f.write(str(xy[triangles[i][1]][0]) + ' ' + str(xy[triangles[i][1]][1]) + ' ' + str(ave) + '\n')
            f.write('\n')
            f.write(str(xy[triangles[i][2]][0]) + ' ' + str(xy[triangles[i][2]][1]) + ' ' + str(ave) + '\n')
            f.write(str(xy[triangles[i][1]][0]) + ' ' + str(xy[triangles[i][1]][1]) + ' ' + str(ave) + '\n')
            f.write('\n')
            f.write('\n')
        #三角形出力
        output_txt_file_name = foldername + '/' + str(graph_num).zfill(4) + '_2.txt'
        f = open(output_txt_file_name, 'w')
        for i in range(number_of_element):
            for j in range(3):
                f.write(str(xy[triangles[i][j]][0]) + ' ' + str(xy[triangles[i][j]][1]) + ' ' + str(un[triangles[i][j]]) + '\n')
            f.write(str(xy[triangles[i][0]][0]) + ' ' + str(xy[triangles[i][0]][1]) + ' ' + str(un[triangles[i][0]]) + '\n')
            f.write('\n')
            f.write('\n')
        '''
        for i in range(number_of_point):
            f.write(str(xy[i][0]) + ' ' + str(xy[i][1]) + ' ' + str(un[i]) + '\n')
        '''
        f.close
        plt_file.write('set output"' + str(graph_num).zfill(4) + '.png"\n')
        plt_file.write('splot "' + str(graph_num).zfill(4) + '_1.txt" using 1:2:3:4 with pm3d, "' + str(graph_num).zfill(4) + '_2.txt" using 1:2:3 with lines lt -1 \n')
        graph_num += 1
        fem.output_txt_result()
        #fem.make_u_graph()

mainloop_time = time.time() - mainloop_start_time
logger.info('end mainloop time:%f', mainloop_time)
program_time = time.time() - program_start_time
logger.info('end program time:%f', program_time)

f_wav.close()

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
#p = subprocess.Popen(cmdstring)#, shell=True)
#p.wait()
#p.kill()
#p = subprocess.Popen(cmdstring2)#, shell=True)
#p.wait()
#p.kill()
#p = subprocess.Popen(cmdstring3)#, shell=True)
#p.wait()
#p.kill()
os.chdir("..")
'''
