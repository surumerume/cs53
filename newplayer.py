#!i/usr/bin/env python
# -*- coding: UTF-8 -*-
#python player.py foldername
#foldername内のresult.txtからOpenGLで動画表示

import os
import sys
import time
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from math import sin, cos, radians
from math import sqrt, floor

def draw():
    global next_frame
    global time_start
    global phi, psi #物体回転用
    global circle_num
    global u, first_center, last_center
    global wav_file
    global input_folder_name
    global size
    global light0pos
    global floor_normal_vec, trans_floor, d
    global shadow_matrix
    global fps, fps_time
    global redisplay_flag
    global string_num
    global meshdata, coordinates, u_vec, u_max, u_min
    global num_of_mesh
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
                next_frame = fps 
        #時間更新
        fps_time = now - time_start
        time_start = now
        if next_frame == 0:
            for i in range(string_num):
                wav_file[i].play()
                #pygame.time.wait(30)
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
        #===========弦の描画==========
        for num in range(string_num):
            glLoadIdentity()
            gluLookAt(0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            glTranslatef(-0.15,-0.02+0.01*num,0.0) 
            #物体拡大
            glScalef(size,size,size)
            #物体回転
            glRotatef(phi, 0.0, 0.0, 1.0)
            glRotatef(psi, 0.0, 1.0, 0.0)
            glLightfv(GL_LIGHT0,GL_POSITION,light0pos)
            #材質設定
            param_string = np.array([0.7,0.5,0.2,1.0])
            glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,param_string)
            #glMaterialfv(GL_FRONT_AND_BACK,GL_EMISSION,red)
            #弦の描画
            draw_string(u[num][next_frame-1],first_center[num],last_center[num])
        #行列プッシュ(弦の設定をpush)
        glPushMatrix()
        #==========床描画==========
        #[memo]
        #法線(a,b,c)で点P(x0,y0,z0)を通る平面の方程式はax+by+cz+d=0とかける
        #d = a*(-x0) + b*(-y0) + c*(-z0)
        #影の描画のためにここでステンシル値に1を書き込んでおく
        #==========================
        #glLoadIdentity()
        #物体回転
        #glRotatef(45, 0.0, 0.0, 1.0)
        #glRotatef(45, 0.0, 1.0, 0.0)
        #-----変数設定-----
        glPopMatrix()
        glPushMatrix()
        #物体移動
        glTranslatef(trans_floor[0], trans_floor[1], trans_floor[2])
        #材質設定
        #param_floor = np.array([1.0,1.0,1.0,1.0])
        #glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,param_floor)
        #床のステンシル1をつける
        glEnable(GL_STENCIL_TEST)
        glStencilFunc( GL_ALWAYS, 1, ~0) #常に通過
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE) #glStencilFuncの第二引数で置き換え
        #床描画
        #draw_floor(0)
        #==========影の描画==========
        #[memo]
        #床の部分（ステンシルバッファ1）のところに
        #弦の影を投影してステンシルバッファをインクリメント
        #============================
        #カラー・デプスバッファマスクをセットする
        #これで以下の内容のピクセルの色の値は書き込まれない
        glColorMask(0,0,0,0)
        glDepthMask(0)
        glLoadIdentity()
        #行列ポップ(弦の設定をpop)
        glPopMatrix()
        #行列プッシュ(弦の設定をpush)
        glPushMatrix()
        #変換行列に影の投影行列かける
        glMultMatrixf(shadow_matrix)
        #材質設定
        #param_floor = np.array([0.0,0.0,0.0,1.0])
        #glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,param_floor)
        #-----影の部分のステンシルバッファを2に-----
        glEnable(GL_STENCIL_TEST)
        glStencilFunc(GL_EQUAL, 1, ~0)
        glStencilOp(GL_KEEP, GL_KEEP, GL_INCR)
        glDisable(GL_DEPTH_TEST)
        for num in range(string_num):
            glTranslatef(0.0,0.05,0.0) 
            #draw_string(u[num][next_frame-1],first_center[num],last_center[num])
        glEnable(GL_DEPTH_TEST)
        #ビットマスクを解除
        glColorMask(1,1,1,1)
        glDepthMask(1)
        #-----影をつける-----
        #行列ポップ(弦の設定をpop)
        glPopMatrix()
        glTranslatef(trans_floor[0], trans_floor[1], trans_floor[2])
        glEnable(GL_STENCIL_TEST)
        glStencilFunc(GL_EQUAL, 2, ~0)
        glDisable(GL_DEPTH_TEST)
        #床描画
        #draw_floor(1)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_STENCIL_TEST)
        #===ギターボディ描画===
        glLoadIdentity()
        gluLookAt(0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        glTranslatef(-0.65,0.0,-0.1) 
        #物体拡大
        glScalef(size*0.001,size*0.001,size*0.001)
        #物体回転
        glRotatef(-90, 0.0, 0.0, 1.0)
        glRotatef(90, 1.0, 0.0, 0.0)
        glRotatef(phi, 0.0, 0.0, 1.0)
        glRotatef(psi, 0.0, 1.0, 0.0)
        glLightfv(GL_LIGHT0,GL_POSITION,light0pos)
        #材質設定
        param_string = np.array([0.6,0.2,0.0,1.0])
        glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,param_string)
        draw_guitar()
        #-----フレーム表示-----
        glColor4f(0.1, 0.1, 0.1, 0.5)
        '''
        drawText('frame:' + str(next_frame),0.0,640.0-15)
        drawText('fps:' + str(1000.0/fps_time),0.0,640.0-30)
        drawText('time:' + str(next_frame/fps),0.0,640.0-45)
        '''
        glutSwapBuffers()
        redisplay_flag = 0


def draw_guitar():
    global meshdata, coordinates, u_vec, u_max, u_min
    global num_of_mesh
    global light0pos
    #=====FEniCS描画=====
    #法線ベクトル自動正規化
    glEnable(GL_NORMALIZE)
    glBegin(GL_TRIANGLES)
    for i in range(num_of_mesh):
        for j in range(4): #三角錘
            #法線ベクトル
            v1 = coordinates[meshdata[i][(j+1)%4]] - coordinates[meshdata[i][j%4]]
            v2 = coordinates[meshdata[i][(j+2)%4]] - coordinates[meshdata[i][(j+1)%4]]
            normal = np.cross(v1,v2)
            light = light0pos[0:3]
            if np.dot(normal,light - coordinates[meshdata[i][j%4]]) < 0:
                    normal = -normal
            glNormal3f(normal[0],normal[1],normal[2])
            for k in range(3):
                #ポリゴン描画
                #red = u_vec[meshdata[i][j]] / u_max
                #blue = (u_max - u_vec[meshdata[i][j]]) / (u_max - u_min)
                #glColor3d(red, 0, blue)
                glVertex3f(coordinates[meshdata[i][(k+j)%4]][0],coordinates[meshdata[i][(k+j)%4]][1],coordinates[meshdata[i][(k+j)%4]][2])
    glEnd()
    #法線ベクトル自動正規化終わり
    glDisable(GL_NORMALIZE)

def draw_string(u, first_center, last_center):
    global circle_num
    #法線ベクトル自動正規化
    glEnable(GL_NORMALIZE)
    #-----最初の端っこ塞ぐ-----
    glBegin(GL_TRIANGLES)
    for i in range(circle_num-1):
        #法線ベクトル
        v1 = u[0][i] - first_center
        v2 = u[0][i+1] - u[0][i] 
        normal = np.cross(v1,v2)
        glNormal3f(normal[0],normal[1],normal[2])
        #ポリゴン描画
        glVertex3f(first_center[0],first_center[1],first_center[2])
        glVertex3f(u[0][i][0],u[0][i][1],u[0][i][2])
        glVertex3f(u[0][i+1][0],u[0][i+1][1],u[0][i+1][2])
    #法線ベクトル
    v1 = u[0][circle_num-1] - first_center
    v2 = u[0][0] - u[0][circle_num-1] 
    normal = np.cross(v1,v2)
    glNormal3f(normal[0],normal[1],normal[2])
    #ポリゴン描画
    glVertex3f(first_center[0],first_center[1],first_center[2])
    glVertex3f(u[0][circle_num-1][0],u[0][circle_num-1][1],u[0][circle_num-1][2])
    glVertex3f(u[0][0][0],u[0][0][1],u[0][0][2])
    glEnd()
    #-----弦の部分-----
    glBegin(GL_QUADS)
    for point in range(31-1): #各節点
        for i in range(circle_num-1): #円状
            #法線ベクトル
            v1 = u[point+1][i] - u[point][i] 
            v2 = u[point+1][i+1] - u[point+1][i] 
            normal = np.cross(v1,v2)
            glNormal3f(normal[0],normal[1],normal[2])
            #ポリゴン描画（反時計回り）
            glVertex3f(u[point][i][0],u[point][i][1],u[point][i][2])
            glVertex3f(u[point+1][i][0],u[point+1][i][1],u[point+1][i][2])
            glVertex3f(u[point+1][i+1][0],u[point+1][i+1][1],u[point+1][i+1][2])
            glVertex3f(u[point][i+1][0],u[point][i+1][1],u[point][i+1][2])
        #法線ベクトル
        v1 = u[point+1][circle_num-1] - u[point][circle_num-1] 
        v2 = u[point+1][0] - u[point+1][circle_num-1] 
        normal = np.cross(v1,v2)
        glNormal3f(normal[0],normal[1],normal[2])
        #ポリゴン描画
        glVertex3f(u[point][circle_num-1][0],u[point][circle_num-1][1],u[point][circle_num-1][2])
        glVertex3f(u[point+1][circle_num-1][0],u[point+1][circle_num-1][1],u[point+1][circle_num-1][2])
        glVertex3f(u[point+1][0][0],u[point+1][0][1],u[point+1][0][2])
        glVertex3f(u[point][0][0],u[point][0][1],u[point][0][2])
    glEnd()
    #-----最後の端っこ塞ぐ-----
    glBegin(GL_TRIANGLES)
    for i in range(circle_num-1):
        #法線ベクトル
        v1 = u[30][i+1] - last_center
        v2 = u[30][i] - u[30][i+1] 
        normal = np.cross(v1,v2)
        glNormal3f(normal[0],normal[1],normal[2])
        #ポリゴン描画
        glVertex3f(last_center[0],last_center[1],last_center[2])
        glVertex3f(u[30][i+1][0],u[30][i+1][1],u[30][i+1][2])
        glVertex3f(u[30][i][0],u[30][i][1],u[30][i][2])
    #法線ベクトル
    v1 = u[30][0] - last_center
    v2 = u[30][circle_num-1] - u[30][0] 
    normal = np.cross(v1,v2)
    glNormal3f(normal[0],normal[1],normal[2])
    #ポリゴン描画
    glVertex3f(last_center[0],last_center[1],last_center[2])
    glVertex3f(u[30][0][0],u[30][0][1],u[30][0][2])
    glVertex3f(u[30][circle_num-1][0],u[30][circle_num-1][1],u[30][circle_num-1][2])
    glEnd()
    #法線ベクトル自動正規化終わり
    glDisable(GL_NORMALIZE)

def draw_floor(shadow_flag):
    global floor_normal_vec
    #==========床描画==========
    #[memo]
    #法線(a,b,c)で点P(x0,y0,z0)を通る平面の方程式はax+by+cz+d=0とかける
    #d = a*(-x0) + b*(-y0) + c*(-z0)
    #==========================
    if shadow_flag == 1:
        #glEnable(GL_BLEND)
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glColor4f(1.0, 0.1, 0.1, 1.0)
        param_floor = np.array( [[0.3,0.3,0.3,1.0], [0.0,0.0,0.0,1.0]])
    else:
        param_floor = np.array( [[0.6,0.6,0.6,1.0], [0.3,0.3,0.3,1.0]])
    #タイル状に床を描画(したい)
    glEnable(GL_NORMALIZE)
    glBegin(GL_QUADS)
    glNormal3fv(floor_normal_vec)
    for i in range(-5,4):
        for j in range(-5,4):
            glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,param_floor[(i+j)%2])
            glVertex3f(i/10.0,0.0,j/10.0)
            glVertex3f(i/10.0,0.0,j/10.0+0.1)
            glVertex3f(i/10.0+0.1,0.0,j/10.0+0.1)
            glVertex3f(i/10.0+0.1,0.0,j/10.0)
    glEnd()
    #if shadow_flag == 1:
        #glDisable(GL_BLEND)
    glDisable(GL_NORMALIZE)
 
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
    gluPerspective(45, 1.0, 0.1, 20) 
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
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    #-----視点光源設定-----#
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    #glEnable(GL_LIGHT1)
    reshape(640,640)
    #-----光源設定-----#
    #green = np.array([0.0,1.0,0.0,1.0])
    #glLightfv(GL_LIGHT1,GL_DIFFUSE,green)
    #glLightfv(GL_LIGHT1,GL_SPECULAR,green)

def calc_shadow_matrix():
    global floor_normal_vec, trans_floor, d
    global shadow_matrix
    #-----投影行列計算-----#
    #[memo]
    #P'=MPとなるM
    #M = N^T L - L N^T
    #----------------------#
    #内積計算(N^T L)
    dot = (floor_normal_vec[0]*light0pos[0] + floor_normal_vec[1]*light0pos[1] + floor_normal_vec[2]*light0pos[2] + d*light0pos[3]) 
    #-----投影行列計算-----
    shadow_matrix = np.zeros( (16) )
    #1行目
    shadow_matrix[0] = -light0pos[0]*floor_normal_vec[0] + dot
    shadow_matrix[4] = -light0pos[0]*floor_normal_vec[1]
    shadow_matrix[8] = -light0pos[0]*floor_normal_vec[2]
    shadow_matrix[12] = -light0pos[0]*d
    #2行目
    shadow_matrix[1] = -light0pos[1]*floor_normal_vec[0]
    shadow_matrix[5] = -light0pos[1]*floor_normal_vec[1] + dot
    shadow_matrix[9] = -light0pos[1]*floor_normal_vec[2]
    shadow_matrix[13] = -light0pos[1]*d
    #3行目
    shadow_matrix[2] = -light0pos[2]*floor_normal_vec[0]
    shadow_matrix[6] = -light0pos[2]*floor_normal_vec[1]
    shadow_matrix[10] = -light0pos[2]*floor_normal_vec[2] + dot
    shadow_matrix[14] = -light0pos[2]*d
    #4行目
    shadow_matrix[3] = -light0pos[3]*floor_normal_vec[0]
    shadow_matrix[7] = -light0pos[3]*floor_normal_vec[1]
    shadow_matrix[11] = -light0pos[3]*floor_normal_vec[2]
    shadow_matrix[15] = -light0pos[3]*d + dot

def drawText(value, x, y):
    glWindowPos2f(x,y)
    for character in value:
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(character))
    
def main(foldername):
    global circle_num
    global first_center, last_center, u
    global prev_mouse
    global phi,psi
    global next_frame,time_start
    global input_folder_name
    global wav_file
    global size
    global light0pos
    global floor_normal_vec, trans_floor, d
    global shadow_matrix
    global fps, fps_time 
    global redisplay_flag 
    global string_num
    global meshdata, coordinates, u_vec, u_max, u_min
    global num_of_mesh
    #==========各種ファイル読み込み==========#
    input_folder_name = [] 
    if not os.path.exists(foldername[0]):
        sys.exit ('Error !!')
    #-----設定ファイル読み込み-----#
    input_setting_file_name = foldername[0] + '/setting.txt'
    f = open(input_setting_file_name, 'r')
    l = f.readline().replace('\n','').split(',')
    fps = int(l[0])
    circle_num = int(l[1])
    wav_file = []
    string_num=6
    first_center = np.zeros( (string_num, 3) )
    last_center = np.zeros( (string_num, 3) )
    u = np.zeros( (string_num, 61, 31, circle_num, 3) ) #各点につき円状に12つの点のそれぞれの座標
    for num in range(string_num):
        for frame in range(fps+1):
            #-----ファイル読み込み-----
            if not os.path.exists(foldername[num]):
                sys.exit ('Error !!')
            input_folder_name.append(foldername[num])
            input_txt_file_name = foldername[num] + '/' + str(frame) + '.txt'
            f = open(input_txt_file_name, 'r')
            #最初の端
            l = f.readline().replace('\n','').split(',')
            first_center[num] = np.asarray(l, dtype=np.float32)
            l = []
            #弦の部分
            for point in range(31):
                #1行ずつ読み込んで処理
                l = f.readline().replace('\n','').split(',')
                #中心周りの点を読み込み
                for i in range(circle_num):
                    for j in range(3):
                        u[num][frame][point][i][j] = l[3*i+j]
                l = []
            #最後の端
            l = []
            l = f.readline().replace('\n','').split(',')
            last_center[num] = np.asarray(l, dtype=np.float32)
            f.close()
        #-----wavファイルをpygameで読み込む-----#
        pygame.init()
        input_wav_file_name = foldername[num] + '/result.wav'
        wav_file.append(pygame.mixer.Sound(input_wav_file_name))
        wav_file[num].set_volume(0.1)
        #-----ファイル読み込み＆更新-----
        input_txt_file_name = 'guitar2.txt'
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
        '''
        #各点のデータ
        l = f.readline().replace('\n','').split(',')
        u_vec = np.asarray(l, dtype=np.float32)
        u_max = np.amax(u_vec)
        u_min = np.amin(u_vec)
        l = []
        '''
        f.close()
    #==========変数設定==========#
    prev_mouse = np.zeros( (2) ) #マウスの座標記憶用
    phi = 0.0 
    psi = 0.0
    next_frame=0
    time_start=0
    fps_time = 1.0
    size = 1.0
    redisplay_flag = 0
    light0pos = np.array([0.0,0.5,0.5,1.0])
    floor_normal_vec = np.array([0.0,1.0,0.0])
    #移動する座標
    trans_floor = np.array([0.0, -0.3, 0.0])
    d = floor_normal_vec[0]*(-trans_floor[0])+floor_normal_vec[1]*(-trans_floor[1]) + floor_normal_vec[2]*(-trans_floor[2]) 
    calc_shadow_matrix()
    #==========OpenGL処理==========#
    init()
    #-----関数設定-----#
    glutDisplayFunc(draw)
    glutReshapeFunc(reshape)
    glutIdleFunc(idle)
    glutMouseFunc(mouse)
    glutMotionFunc(None)
    glutKeyboardFunc(keyboard)
    glutSpecialFunc(keyboard)
    #-----ループ-----#
    glutMainLoop()

if __name__ == '__main__':
    main(sys.argv[1:])
