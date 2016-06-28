#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#ut=v
#vt=uxx-uxxxx-v+utxx
#q,Du,DDu,Du(-)
#境界条件考慮
#ボディに合わせて境界条件が変わる
import os
import os.path
import subprocess
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
#import pyaudio
import wave
from math import exp
from math import sin, cos
from math import sqrt 

class Femwave:
    def __init__(self,foldername,rate,length,density,section_area,tension,d1,d3,young,moment):
        self.foldername = foldername
        #各係数
        self.length=length
        self.density=density
        self.section_area=section_area
        self.tension=tension
        self.d1=d1
        self.d3=d3
        self.young=young
        self.moment=moment
        #その他設定
        self.M=30 #節点の数
        self.h=length/(self.M-1) #幅
        self.slop=1.0/self.h
        self.t=0.0 #初期時刻
        self.tmax=10.0 #終了時刻
        self.rate=rate #サンプリングレート
        self.dt=1.0/self.rate #時間刻み
        self.step=0 #現在のステップ
        self.size=6*(self.M-2)+6 #係数行列及びベクトルのサイズ
        self.pi = 3.1415
        #グラフ関連
        self.graph_num=0
        self.fig = plt.figure()
        self.txt_num=0
        #===係数行列の計算===#
        #---行列---#
        #ハット関数の積分結果格納用
        self.Ma = np.zeros( (self.M,self.M) )     #M*M 微分,微分なし
        self.Mb = np.zeros( (self.M,self.M) )     #M*M 微分なし,微分
        self.M1 = np.zeros( (self.M-2,self.M-2) ) #(M-2)*(M-2) 微分なし,微分なし
        self.M2 = np.zeros( (self.M-2,self.M) )   #(M-2)*M 微分,微分なし
        self.M3 = np.zeros( (self.M,self.M) )     #M*M 微分なし,微分なし
        self.M4 = np.zeros( (self.M,self.M-2) )   #M*(M-2) 微分,微分なし
        self.M5 = np.zeros( (self.M,self.M-2) )   #M*(M-2) 微分なし,微分
        self.M6 = np.zeros( (self.M-2,self.M) )   #(M-2)*M 微分なし,微分
        self.co_left = np.zeros( (self.size,self.size) )     #左辺(n+1)の係数行列
        self.co_right = np.zeros( (self.size,self.size) )    #右辺(n)の係数行列
        #---積分計算---#
        #(M)*(M)
        for i in range(self.M):
            for j in range(self.M):
                if i==j: #重なり
                    if i==0: #はじっこ
                        self.M3[i][j] = self.h/3.0
                        self.Ma[i][j] = -1.0/2.0
                        self.Mb[i][j] = -1.0/2.0
                    elif i==self.M-1: #はじっこ
                        self.M3[i][j] = self.h/3.0
                        self.Ma[i][j] = 1.0/2.0
                        self.Mb[i][j] = 1.0/2.0
                    else:
                        self.M3[i][j] = 2.0*self.h/3.0 
                        self.Ma[i][j] = 0.0
                        self.Mb[i][j] = 0.0
                elif i-1==j and i>=0: #かたっぽ 
                    self.M3[i][j] = self.h/6.0
                    self.Ma[i][j] = -1.0/2.0
                    self.Mb[i][j] = 1.0/2.0
                elif i+1==j and i<=self.M-1: #かたっぽ
                    self.M3[i][j] = self.h/6.0
                    self.Ma[i][j] = 1.0/2.0
                    self.Mb[i][j] = -1.0/2.0
                else:
                    self.M3[i][j] = 0.0
                    self.Ma[i][j] = 0.0
                    self.Mb[i][j] = 0.0
        self.M1[:,:] = self.M3[1:self.M-1,1:self.M-1]
        self.M2[:,:] = self.Ma[1:self.M-1,0:self.M]
        self.M4[:,:] = self.Ma[0:self.M,1:self.M-1] 
        self.M5[:,:] = self.Mb[0:self.M,1:self.M-1]
        self.M6[:,:] = self.Mb[1:self.M-1,0:self.M]
        #係数
        co1 = -(self.young*self.moment)/(self.density*self.section_area) 
        co2 = self.tension/(2.0*self.density*self.section_area) 
        co3 = 1.0/self.dt+self.d1/(2.0*self.density*self.section_area)
        co4 = 1.0/self.dt-self.d1/(2.0*self.density*self.section_area)
        co5 = self.d3/(self.dt*self.density*self.section_area)
        #---co_matrix---
        #---co_left---[u(n+1) v(n+1) q Du DDu Du(-)]
        # u(n+1)(M-2) v(n+1)(M-2) q(M)  Du(M)  DDu(M-2) Du(-)(M)
        # | (1/dt)M1  (-1/2)M1    0       0       0       0    |
        # |    0       co3*M1  co1*M2  co2*M2     0    co5*M2  |
        # |    0          0       M3      0    (1/2)M4    0    |
        # |    M5         0       0      -M3      0       0    |
        # |    0          0       0      -M6      M1      0    |
        # |    M5         0       0       0       0      -M3   |
        #---co_right---[u(n) v(n) q Du DDu Du(-)]
        #   u(n)(M-2) v(n)(M-2)  q(M)    Du(M)  DDu(M-2) Du(-)(M)
        # | (1/dt)M1   (1/2)M1    0       0       0       0    | 
        # |    0       co4*M1     0       0       0       0    |
        # |    0          0       0       0       0       0    |
        # |   -M5         0       0       0       0       0    |
        # |    0          0       0       0       0       0    |
        # |    M5         0       0       0       0       0    |
        #---左辺右辺係数行列計算---#
        for i in range(self.size):
            for j in range(self.size):
                if i<(self.M-2): #1行目
                    if j<(self.M-2): #1列目u
                        self.co_left[i][j] = (1.0/self.dt)*self.M1[i][j]
                        self.co_right[i][j] = (1.0/self.dt)*self.M1[i][j]
                    elif (self.M-2)<=j and j<2*(self.M-2): #2列目v
                        self.co_left[i][j] = (-1.0/2.0)*self.M1[i][j-(self.M-2)]
                        self.co_right[i][j] = (1.0/2.0)*self.M1[i][j-(self.M-2)]
                elif i<2*(self.M-2): #2行目
                    if (self.M-2)<=j and j<2*(self.M-2): #2列目v
                        self.co_left[i][j] = co3*self.M1[i-(self.M-2)][j-(self.M-2)]
                        self.co_right[i][j] = co4*self.M1[i-(self.M-2)][j-(self.M-2)]
                    elif 2*(self.M-2)<=j and j<3*(self.M-2)+2: #3列目q
                        self.co_left[i][j] = co1*self.M2[i-(self.M-2)][j-2*(self.M-2)]
                    elif 3*(self.M-2)+2<=j and j<4*(self.M-2)+4: #4列目Du
                        self.co_left[i][j] = co2*self.M2[i-(self.M-2)][j-(3*(self.M-2)+2)] 
                    elif 5*(self.M-2)+4<=j and j<6*(self.M-2)+6: #6列目Du(-) 
                        self.co_left[i][j] = co5*self.M2[i-(self.M-2)][j-(5*(self.M-2)+4)]
                elif i<3*(self.M-2)+2: #3行目
                    if 2*(self.M-2)<=j and j<3*(self.M-2)+2: #3列目
                        self.co_left[i][j] = self.M3[i-2*(self.M-2)][j-2*(self.M-2)]
                    elif 4*(self.M-2)+4<=j and j<5*(self.M-2)+4: #5列目
                        self.co_left[i][j] = (1.0/2.0)*self.M4[i-2*(self.M-2)][j-(4*(self.M-2)+4)]
                elif i<4*(self.M-2)+4: #4行目
                    if j<(self.M-2): #1列目
                        self.co_left[i][j] = self.M5[i-(3*(self.M-2)+2)][j]
                        self.co_right[i][j] = -self.M5[i-(3*(self.M-2)+2)][j]
                    elif 3*(self.M-2)+2<=j and j<4*(self.M-2)+4: #4列目
                        self.co_left[i][j] = -self.M3[i-(3*(self.M-2)+2)][j-(3*(self.M-2)+2)]
                elif i<5*(self.M-2)+4: #5行目
                    if 3*(self.M-2)+2<=j and j<4*(self.M-2)+4: #4列目
                        self.co_left[i][j] = -self.M6[i-(4*(self.M-2)+4)][j-(3*(self.M-2)+2)]
                    elif 4*(self.M-2)+4<=j and j<5*(self.M-2)+4: #5列目
                        self.co_left[i][j] = self.M1[i-(4*(self.M-2)+4)][j-(4*(self.M-2)+4)]
                elif i<6*(self.M-2)+6: #6行目
                    if j<(self.M-2): #1列目u
                        self.co_left[i][j] = self.M5[i-(5*(self.M-2)+4)][j]
                        self.co_right[i][j] = self.M5[i-(5*(self.M-2)+4)][j]
                    elif 5*(self.M-2)+4<=j and j<6*(self.M-2)+6: #6列目Du(-)
                        self.co_left[i][j] = -self.M3[i-(5*(self.M-2)+4)][j-(5*(self.M-2)+4)]
                else:
                   print("error")
        #その他行列
        self.uvn = np.zeros( (self.size) )               #右辺のベクトルuとv,q,Du,DDu,DU(-)の(n)
        self.un_bc = 0.0                                #u(n+1)のN+1の境界条件
        self.up_bc = 0.0                                #u(n)のN+1の境界条件
        self.vn_bc = 0.0                                #v(n+1)のN+1の境界条件
        self.vp_bc = 0.0                                #v(n)のN+1の境界条件
        self.right = np.zeros( (self.size) )             #右辺計算結果 
        self.un = np.zeros( (self.M) )                  #グラフ出力用
        self.wav_data = np.zeros( (self.tmax*self.rate) )    #wav書き出し用
        self.energy = np.zeros( (self.tmax*self.rate) )      #エネルギー
        self.time = np.zeros( (self.tmax*self.rate) )      #時間
        self.tension_log = np.zeros( (self.tmax*self.rate) ) #張力
        self.x = np.zeros( (self.M) )
        self.bc_vec = np.zeros( (self.size) )           #境界条件による項
        ###u初期値設定
        for i in range(self.M-2):
            self.uvn[i] = exp(-50*((i+1)*self.h-self.length/5)*((i+1)*self.h-self.length/5))#/50.0
            #self.uvn[i] = i*self.h/self.length + exp(-50*((i+1)*self.h-self.length/5)*((i+1)*self.h-self.length/5))#/50.0
        for i in range(self.M):
            self.x[i] = i*self.h
        self.un[1:self.M-1] = self.uvn[:self.M-2]
        self.un[self.M-1] = self.up_bc
        self.time[0] = self.t 
        #wav_data
        self.wav_data[self.step] = self.un[self.M/2]
        #self.make_u_graph()
        #self.calc_energy()

    #1ステップ進めて、現在のステップ数を返す
    def simulate_one_step(self):
        #B.C.更新
        #uの境界
        self.bc_vec[self.M-3] = self.h/6.0/self.dt*(-self.un_bc+self.up_bc) 
        #vの境界
        self.bc_vec[2*(self.M-2)-2] = self.h/6.0/2.0*(self.vn_bc+self.vp_bc) + self.h/6.0/self.dt*(self.vn_bc-self.vp_bc) 
        #Duの境界
        self.bc_vec[4*(self.M-2)+2] = 1.0/2.0*(self.un_bc+self.up_bc) 
        self.bc_vec[4*(self.M-2)+3] = 1.0/2.0*(self.un_bc+self.up_bc) 
        #右辺計算
        right = np.zeros( (self.size) )
        right = np.dot(self.co_right,self.uvn) + self.bc_vec
        #解く
        self.uvn = np.linalg.solve(self.co_left,right)
        self.un[1:self.M-1] = self.uvn[:self.M-2]
        self.un[self.M-1] = self.un_bc
        #wav用配列に書き出し
        self.wav_data[self.step] = self.un[self.M/2]
        #print ("{0}:{1}".format(t,wav_data[i]))
        #時間進行
        self.t += self.dt
        self.step += 1
        self.time[self.step] = self.step*self.dt
        return self.step

    def set_un_bc(self,value_bc):
        #境界更新
        self.up_bc = self.un_bc
        self.vp_bc = self.vn_bc
        self.un_bc = value_bc
        self.vn_bc = (self.un_bc-self.up_bc)/self.dt

    def get_ux_bc(self):
        #return self.uvn[4*(self.M-2)+3]
        return (self.un_bc - self.uvn[self.M-3])/self.h

    #uの変位を出力する関数
    def make_u_graph(self,visual_mode=0):
        #グラフ
        ax1 = self.fig.add_subplot(111)
        ax1.set_xlim(0,self.length)
        output_png_file_name = self.foldername + "/%04d.png" % self.graph_num 
        if visual_mode == 1:
            ax1.plot(self.x,self.un,"y")
            #ax1.set_ylim(-1,1)
            #ax1.text(0.01, 0.9, 'time = %.5f' % t, bbox=dict(facecolor='red'))
            ax1.set_ylim(-15,15)
            ax1.text(self.length/100, 13, 'time = %.5f' % self.t, bbox=dict(facecolor='red'))
            #背景色
            ax1.patch.set_facecolor('black')
            #目盛りを消す
            ax1.tick_params(labelbottom='off')
            ax1.tick_params(labelleft='off')
            #グラフを全体に広げて出力
            self.fig.savefig(output_png_file_name, bbox_inches="tight", pad_inches=0.0) 
        else:
            ax1.plot(self.x,self.un)
            ax1.set_xlabel("Position(m)", fontsize=14)
            ax1.set_ylabel("Amplitude(m)", fontsize=14)
            ax1.tick_params(labelsize=14)
            ax1.text(self.length/100, 0.9, 'time = %.5f' % self.t, fontsize=14)
            ax1.set_ylim(-1,1)
            self.fig.savefig(output_png_file_name) 
        self.fig.clf()
        self.graph_num += 1

    #エネルギー計算
    def calc_energy(self):
        eun = np.zeros( (self.M-2) )              #現在のu
        v = np.zeros( (self.M-2) )                #現在のv
        du = np.zeros( (self.M) )                 #du
        ddu = np.zeros( (self.M-2) )              #ddu
        #エネルギー
        eun[:] = self.uvn[:self.M-2]
        v[:] = self.uvn[self.M-2:2*(self.M-2)]
        du = np.linalg.solve(self.M3,np.dot(self.M5,eun))
        ddu = np.linalg.solve(self.M1,np.dot(self.M6,du))
        self.energy[self.step] = np.dot(np.dot(self.M1,v),v)/2.0 + np.dot(np.dot(self.M3,du),du)*self.tension/(self.density*self.section_area*2.0) + np.dot(np.dot(self.M1,ddu),ddu)*self.young*self.moment/(self.density*self.section_area*2.0)
        #print ("{0}".format(energy[i]))

    def make_energy_graph(self):
        #エネルギーのグラフ
        ax1 = self.fig.add_subplot(111)
        ax1.set_xlim(0,1.0)
        ax1.set_xlabel("Time(s)", fontsize=14)
        ax1.set_ylabel("Energy", fontsize=14)
        ax1.tick_params(labelsize=14)
        ax1.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax1.plot(self.time, self.energy)
        output_png_file_name = self.foldername + "/energy.png" 
        self.fig.savefig(output_png_file_name) 
        self.fig.clf()

    #張力計算
    def calc_tension(self):
        #deltaL計算
        deltaL = 0.0
        tun = np.zeros( (self.M) ) #現在のu
        tun[1:self.M-1] = self.uvn[:self.M-2]
        for i in range(self.M-1):
            deltaL += sqrt(self.h**2+(tun[i+1]-tun[i])**2)
        deltaL = deltaL - self.length
        #張力
        self.tension_log[self.step] = self.tension + self.young*self.section_area*deltaL/self.length 

    #張力出力
    def output_tension(self):
        output_txt_file_name = self.foldername + '/tension_log.txt'
        f = open(output_txt_file_name, 'a')
        for i in range(self.tension_log.size):
            f.write(str(self.tension_log[i])+'\n')
        f.close

    def make_wav_graph(self):
        #wavのグラフ
        ax1 = self.fig.add_subplot(111)
        ax1.set_xlim(0,1.0)
        ax1.set_xlabel("Time(s)", fontsize=14)
        ax1.set_ylabel("Amplitude(m)", fontsize=14)
        ax1.tick_params(labelsize=14)
        ax1.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax1.plot(self.time, self.wav_data)
        output_png_file_name = self.foldername + "/wav.png" 
        self.fig.savefig(output_png_file_name) 
        self.fig.clf()

    def output_wav(self):
        self.wav_data = (self.wav_data/np.amax(self.wav_data))*32767
        self.wav_data = np.asarray(self.wav_data, dtype=np.int16)
        #print ("{0}".format(wav_data))
        output_wav_file_name = self.foldername + "/result.wav"
        scipy.io.wavfile.write(output_wav_file_name,self.rate,self.wav_data)

    #OpenGLを使うために座標データを吐き出しておく関数
    def output_txt_result(self):
        circle_num=6
        #=====設定ファイル出力=====
        if self.txt_num == 0:
            output_txt_file_name = self.foldername + '/setting.txt'
            f = open(output_txt_file_name, 'a')
            f.write(str(self.fps)+','+str(circle_num))
            f.close
        #=====フレームファイル出力=====
        output_txt_file_name = self.foldername + '/' + str(self.txt_num) + '.txt'
        f = open(output_txt_file_name, 'a')
        #[memo]
        #s = s + 'hoge'はオーバーヘッドがあって遅いらしいから
        #こうするといいらしい(伝聞)
        l = []
        #最初の点の真ん中
        #x
        l.append(str(self.x[0]-self.length/2.0))
        #y
        l.append(str(self.un[0]))
        #z
        l.append('0.0')
        text = ','.join(l) 
        f.write(text+'\n')
        l = []
        for i in range(self.M): #各点について
            for j in range(circle_num): #円状に点を配置
                #x
                l.append(str(self.x[i]-self.length/2.0))
                #y
                l.append(str(self.un[i]+sqrt(self.section_area/self.pi)*sin(2*self.pi*j/circle_num)))
                #z
                l.append(str(sqrt(self.section_area/self.pi)*cos(2*self.pi*j/circle_num)))
            text = ','.join(l) 
            f.write(text+'\n')
            l = []
        #最後の点の真ん中
        #x
        l.append(str(self.x[self.M-1]-self.length/2.0))
        #y
        l.append(str(self.un[self.M-1]))
        #z
        l.append('0.0')
        text = ','.join(l) 
        f.write(text+'\n')
        f.close
        self.txt_num += 1

    def set_fps(self,fps):
        self.fps = fps 

'''
def play_wav(foldername="result"):
    #フォルダチェック
    if not os.path.exists(foldername):
        sys.exit ('Error !!')

    #ファイルオープン
    input_wav_file_name = foldername + "/result.wav"
    wf = wave.open(input_wav_file_name, "r")

    # ストリームを開く
    p = pyaudio.PyAudio()
    stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True)

    # チャンク単位でストリームに出力し音声を再生
    chunk = 1024
    data = wf.readframes(chunk)
    while data != '':
        stream.write(data)
        data = wf.readframes(chunk)
    stream.close()
    p.terminate()
'''

def make_movie(foldername="result"):
    #フォルダチェック
    if not os.path.exists(foldername):
        sys.exit ('Error !!')
    #ffmpeg命令
    cmdstring = ('ffmpeg.exe', '-r', '60', '-i', '%04d.png',
            '-qscale', '0', '-y', 'out.avi')
    cmdstring2 = ('ffmpeg.exe', '-i', 'out.avi',
            '-i', 'result.wav', '-y', 'result.avi')
    cmdstring3 = ('ffmpeg.exe', '-i', 'result.avi',
            '-movflags', 'faststart', '-vcodec', 'libx264',
            '-acodec', 'copy', '-y', 'result.mp4')
    #'libmp3lame', '-ac', '1', '-ar', '44100',
            #'-ab', '256k', '-y', 'result.mp4')
    os.chdir(foldername)
    #subprocess.call("rm result.avi", shell=True)
    p = subprocess.Popen(cmdstring)#, shell=True)
    p.wait()
    p.kill()
    p = subprocess.Popen(cmdstring2)#, shell=True)
    p.wait()
    p.kill()
    p = subprocess.Popen(cmdstring3)#, shell=True)
    p.wait()
    p.kill()
    os.chdir("..")

def play_movie(foldername='result'):
    #windowsでしか動きません
    if os.name == 'nt':
        #windows media playerで再生を試みるテスト
        #(wmpを閉じるまでsoundmakerは止まる)
        #player_path = "C:/Program Files (x86)/Windows Media Player/wmplayer.exe"
        #video_path = os.path.abspath(os.path.dirname(__file__)) + '/' + foldername+'/result.avi'
        #subprocess.call([player_path, video_path])

        #関連付けされたプログラムでファイルを開く
        #絶対パス
        os.startfile(os.path.abspath(os.path.dirname(__file__)) + '/' + foldername+'/result.avi')


if __name__=="__main__":
    make_wav()
    #play_wav()
    make_movie()
