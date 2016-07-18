#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
cho_flag = 0 
end_flag = 0 
xy_list = [] 
triangles_list = [] 
coordinate = [0.0,0.0,0.0]

#stlファイル読み込み
stl_file = open('plane02_old.stl', 'r')
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
            coordinate[i-1] = float(s2[i])
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

#メッシュファイルとして書き出す
mesh_file = open('plane02.mesh', 'w')
text = ','.join(map(str,xy_list)) 
mesh_file.write(text + '\n')
text = ','.join(map(str,triangles_list)) 
mesh_file.write(text + '\n')
