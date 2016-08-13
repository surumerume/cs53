#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
cho_flag = 0 
end_flag = 0 
xy_list = [] 
triangles_list = [] 
coordinate = [0.0,0.0,0.0]

#stlファイル読み込み
#inputfilename = 'plane02_old32'
inputfilename = 'guitar_plane_1248'
stl_file = open(inputfilename + '.stl', 'r')
#要素のところまで行を進める
while True:
    s = stl_file.readline() 
    if s.find('outer loop')!=-1:
        break
    if s.find('endsolid vcg')!=-1:
        end_flag = 1 
        break
while end_flag==0:
    while True:
        s = stl_file.readline()
        print(s)
        if s.find('endloop')!=-1:
            print('<---end_of_loop--->')
            break
        s1 = s.strip()
        s2 = s1.split()
        #座標を取得
        for i in range(1,4):
            coordinate[i-1] = float(s2[i])/100.0
        print(coordinate)
        #重複確認と追加
        print(xy_list)
        for i in range(0,len(xy_list),2):
            if xy_list[i] == coordinate[0] and xy_list[i+1] == coordinate[2]:
                print(int(i/2))
                triangles_list.append(int(i/2))
                cho_flag = 1
                break
        if cho_flag == 0:
            xy_list.append(coordinate[0]) 
            xy_list.append(coordinate[2]) 
            triangles_list.append(int(len(xy_list)/2)-1) 
            print(xy_list)
            print(int(len(xy_list)/2)-1)
        cho_flag = 0
    #要素のところまで行を進める
    while True:
        s = stl_file.readline() 
        if s.find('outer loop')!=-1:
            print(s)
            print('<---outer_loop--->')
            break
        if s.find('endsolid vcg')!=-1:
            end_flag = 1 
            break
print(xy_list)
print(triangles_list)
stl_file.close()

xy = np.asarray(xy_list, dtype=np.float32)
xy = np.reshape(xy, (len(xy_list)/2,2))
triangles = np.asarray(triangles_list, dtype=np.int32)
triangles = np.reshape(triangles, (len(triangles_list)/3,3))
print(xy)
print(triangles)
#節点と要素のそれぞれの数
number_of_point = xy.shape[0]
number_of_element = triangles.shape[0]
print(number_of_point)
print(number_of_element)

#境界を判別してみるテスト
#複数境界区別バージョン！
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

#===境界条件設定
#各境界を辿っていき区別する
#境界の数については決め打ち
#今回は固定端と開放端なので決め打ちで0と1を振る
#---bc1
print('<-----bc1----->')
bc1_list = []
#始点
start_num = boundary_list[0][0]
pre_num = start_num 
num = boundary_list[0][1]
bc1_list.append(pre_num)
bc1_list.append(num)
#境界ひとつめ
while True:
    for i in range(len(boundary_list)):
        print(boundary_list[i])
        if boundary_list[i][0] == num and boundary_list[i][1] != pre_num:
            pre_num = num
            num = boundary_list[i][1]
            break
        if boundary_list[i][1] == num and boundary_list[i][0] != pre_num:
            pre_num = num
            num = boundary_list[i][0]
            break
    if num == start_num:
        break
    bc1_list.append(num)
print(bc1_list)
#---bc2
print('<-----bc2----->')
bc2_list = []
cho_flag = 0
#始点
for i in range(len(boundary_list)):
    for j in range(len(bc1_list)):
        if boundary_list[i][0] == bc1_list[j] or boundary_list[i][1] == bc1_list[j]:
            cho_flag = 1
    if cho_flag == 0:
        start_num = boundary_list[i][0]
        pre_num = start_num
        num = boundary_list[i][1]
        break
    cho_flag = 0
bc2_list.append(pre_num)
bc2_list.append(num)
#境界ふたつめ
while True:
    for i in range(len(boundary_list)):
        print(boundary_list[i])
        if boundary_list[i][0] == num and boundary_list[i][1] != pre_num:
            pre_num = num
            num = boundary_list[i][1]
            break
        if boundary_list[i][1] == num and boundary_list[i][0] != pre_num:
            pre_num = num
            num = boundary_list[i][0]
            break
    if num == start_num:
        break
    bc2_list.append(num)
print(bc2_list)
for i in range(number_of_point):
    for j in range(len(bc1_list)):
        if i == bc1_list[j]:
            BC[i] = 1
            break
print(BC)
#テスト終わり
#print(number_of_point)
#print(BC.shape[0])

#メッシュファイルとして書き出す
mesh_file = open(inputfilename + '.mesh', 'w')
text = ','.join(map(str,xy_list)) 
mesh_file.write(text + '\n')
text = ','.join(map(str,triangles_list)) 
mesh_file.write(text + '\n')
l = BC.tolist()
l = map(int,l)
print(l)
text = ','.join(map(str,l)) 
mesh_file.write(text + '\n')
