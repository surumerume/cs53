#!i/usr/bin/env python
# -*- coding: UTF-8 -*-
#python player.py foldername
#foldername内のresult.txtからOpenGLで動画表示

import os
import sys
import time
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from math import sin, cos, radians
from math import sqrt, floor

def draw():
    global next_frame
    global time_start
    global u
    global meshdata, coordinates, u_vec, u_max, u_min
    global num_of_mesh
    global phi, psi #物体回転用
    global input_folder_name
    global size
    global light0pos
    global fps, fps_time
    global redisplay_flag
    #初期待ち時間(player立ち上がり時間考慮)
    wait_time = 500
    #==========フレーム更新|データ読み込み処理==========
    now = glutGet(GLUT_ELAPSED_TIME)
    if next_frame == 0 and now < wait_time:
        time_start = now
    elif (now - time_start) >= 1000.0/fps and next_frame != fps:
        redisplay_flag = 1
        #-----遅れている場合フレームを飛ばす-----
        if (now - time_start)/(1000.0/fps) >= 2:
            next_frame += int((now - time_start)/(1000.0/fps)) - 1
            if next_frame > fps:
                next_frame=fps
        #-----ファイル読み込み＆更新-----
        input_txt_file_name = input_folder_name + '/%04d.txt' % next_frame
        f = open(input_txt_file_name, 'r')
        #======ここからFEniCSのデータ読み込み=====
        #メッシュのデータ
        l = f.readline().replace('\n','').split(',')
        meshdata = np.asarray(l, dtype=np.int32)
        meshdata = np.reshape(meshdata, (len(l)/4,4))
        num_of_mesh = len(l)/4
        l = []
        #座標データ
        l = f.readline().replace('\n','').split(',')
        coordinates = np.asarray(l, dtype=np.float32)
        coordinates = np.reshape(coordinates, (len(l)/3,3))
        l = []
        #各点のデータ
        l = f.readline().replace('\n','').split(',')
        u_vec = np.asarray(l, dtype=np.float32)
        u_max = np.amax(u_vec)
        u_min = np.amin(u_vec)
        l = []
        f.close()
        #時間更新
        fps_time = now - time_start
        time_start = now
        if next_frame == 0:
            next_frame += 1
        elif next_frame < fps:
            next_frame += 1
    #=====描画処理=====
    if redisplay_flag == 1:
        #==========OpenGL==========
        #初期化
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        #深さオン（描画順から見える順に）
        glEnable(GL_DEPTH_TEST)
        #光源設定
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(1000, 1000, 1000, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        #物体拡大
        glScalef(size,size,size)
        #物体回転
        glRotatef(phi, 0.0, 0.0, 1.0)
        glRotatef(psi, 0.0, 1.0, 0.0)
        #=====FEniCS描画=====
        glBegin(GL_TRIANGLES)
        for i in range(num_of_mesh):
            for j in range(4): #三角錘
                for k in range(3):
                    #ポリゴン描画
                    red = u_vec[meshdata[i][j]] / u_max
                    blue = (u_max - u_vec[meshdata[i][j]]) / (u_max - u_min)
                    glColor3d(red, 0, blue)
                    glVertex3f(coordinates[meshdata[i][(k+j)%4]][0],coordinates[meshdata[i][(k+j)%4]][1],coordinates[meshdata[i][(k+j)%4]][2])
        glEnd()
        #-----フレーム表示-----
        glColor4f(0.1, 0.1, 0.1, 0.5)
        drawText('frame:' + str(next_frame),0.0,640.0-15)
        drawText('fps:' + str(1000.0/fps_time),0.0,640.0-30)
        drawText('time:' + str(next_frame/fps),0.0,640.0-45)
        glutSwapBuffers()
        redisplay_flag = 0

def keyboard(key, x, y):
    global size
    global redisplay_flag
    if key == GLUT_KEY_UP:
        size += 0.01
    if key == GLUT_KEY_DOWN and size > 0.0:
        size -= 0.01
    redisplay_flag = 1
    glutPostRedisplay()

def reshape(w, h):
    global light0pos
    glViewport(0,0,w,h)
    #視点設定
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    #glOrtho(-1,1,-1,1,0.1,50)
    gluPerspective(45, 1.0, 0.5, 100000) 
    #light1pos = np.array([5.0,3.0,0.0,1.0])
    #glLightfv(GL_LIGHT1,GL_POSITION,light1pos)

def idle():
    glutPostRedisplay()

def init():
    #==========OpenGL初期設定==========#
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_STENCIL)
    glutInitWindowSize(640, 640)
    glutCreateWindow("Soundmaker")
    #-----表裏の表示管理-----#
    #glEnable(GL_CULL_FACE)
    #glCullFace(GL_BACK)
    #-----視点光源設定-----#
    #glEnable(GL_LIGHTING)
    #glEnable(GL_LIGHT0)
    #glEnable(GL_LIGHT1)
    reshape(480,480)
    #-----光源設定-----#
    #green = np.array([0.0,1.0,0.0,1.0])
    #glLightfv(GL_LIGHT1,GL_DIFFUSE,green)
    #glLightfv(GL_LIGHT1,GL_SPECULAR,green)

def mouse(button, state, x, y):
    global prev_mouse
    global next_frame
    global time_start
    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            prev_mouse[0] = x
            prev_mouse[1] = y
            glutMotionFunc(motion)
        elif state == GLUT_UP:
            glutMotionFunc(None)
    elif button == GLUT_RIGHT_BUTTON and state == GLUT_UP:
        next_frame = 0
        time_start = glutGet(GLUT_ELAPSED_TIME) - 1000/60
        glutPostRedisplay()

#ドラッグでカメラ位置を回転
def motion(x, y):
    global prev_mouse
    global phi, psi
    global redisplay_flag
    phi = phi+(prev_mouse[1]-y)/2 
    psi = psi+(prev_mouse[0]-x)/2
    if phi > 360:
        phi = 0
    if psi > 360:
        psi = 0
    prev_mouse[0] = x
    prev_mouse[1] = y
    redisplay_flag = 1
    glutPostRedisplay()

def drawText(value, x, y):
    glWindowPos2f(x,y)
    for character in value:
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(character))
    
def main(foldername):
    global next_frame,time_start
    global input_folder_name
    global u
    global meshdata, coordinates, u_vec
    global size
    global light0pos
    global fps, fps_time 
    global redisplay_flag
    global phi, psi
    global prev_mouse
    #==========各種ファイル読み込み==========#
    if not os.path.exists(foldername):
        sys.exit ('Error !!')
    input_folder_name = foldername
    fps = 6 
    #==========変数設定==========#
    next_frame=0
    time_start=0
    fps_time = 1.0
    size = 1.0
    redisplay_flag = 0
    light0pos = np.array([0.0,0.5,0.5,1.0])
    phi = 0.0 
    psi = 0.0
    prev_mouse = np.zeros( (2) ) #マウスの座標記憶用
    #==========OpenGL処理==========#
    init()
    #-----関数設定-----#
    glutDisplayFunc(draw)
    glutReshapeFunc(reshape)
    glutMouseFunc(mouse)
    glutMotionFunc(None)
    glutIdleFunc(idle)
    glutKeyboardFunc(keyboard)
    glutSpecialFunc(keyboard)
    #-----ループ-----#
    glutMainLoop()

if __name__ == '__main__':
    main(sys.argv[1])
