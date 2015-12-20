#!i/usr/bin/env python
# -*- coding: UTF-8 -*-
#newplayer呼び出し用soundmaker

import Tkinter as tk
#import pyglet
from femwave import Femwave
import femwave
import os
import sys
import subprocess
import time
import numpy as np
from PIL import Image
from PIL import ImageOps

def call_play_movie():
    global string_num
    global length_value
    global density_value
    global section_area_value
    global tension_value
    global d1_value
    global d3_value
    global young_value
    global moment_value
    #global start,end,time
    ###ボタンラベル変更
    play_movie_button.configure(text='please wait')
    set_value(string_num)
    foldername = []
    for i in range(6):
        #フォルダの名前
        foldername.append(str(length_value[i]) + str(density_value[i]) + str(section_area_value[i]) + \
                str(tension_value[i]) + str(d1_value[i]) + str(d3_value[i]) + str(young_value[i]) + str(moment_value[i]))
        #フォルダがなければシミュレートして動画作成
        if not os.path.exists(foldername[i]):
            simulate(foldername[i],i)
    #動画再生
    #femwave.play_movie(foldername)
    #OpenGL（予定）
    ###############
    #メモ
    #shellから別のプログラムを呼んでそこからGLUTの処理を行えばよい
    #巻き添え終了阻止
    ###############
    #ffmpeg命令
    cmdstring = ('python', 'newplayer.py', foldername[0],foldername[1],foldername[2],foldername[3],foldername[4],foldername[5])
    p = subprocess.Popen(cmdstring)
    p.wait()
    #p.kill()
    #ボタンラベル変更
    play_movie_button.configure(text='play movie')

def show_energy_graph():
    #フォルダの名前
    foldername = str(length.get()) + str(density.get()) + str(section_area.get()) + \
            str(tension.get()) + str(d1.get()) + str(d3.get()) + str(young.get()) + str(moment.get())
    #フォルダがなければシミュレートして動画作成
    if not os.path.exists(foldername):
        simulate(foldername)
    #関連付けされたプログラムでファイルを開く
    #絶対パス
    os.startfile(os.path.abspath(os.path.dirname(__file__)) + '/' + foldername+'/energy.png')

def simulate(foldername,num):
    global length_value
    global density_value
    global section_area_value
    global tension_value
    global d1_value
    global d3_value
    global young_value
    global moment_value
    #start = time.time()
    os.mkdir(foldername)
    rate = 44100
    tmax = 1.0
    step = 0
    fps = 60 
    #初期設定（とゼロステップ目出力）
    fem = Femwave(foldername,rate,length_value[num],density_value[num],section_area_value[num],
            tension_value[num],d1_value[num],d3_value[num],young_value[num],moment_value[num])
    #fem.make_u_graph()
    fem.calc_energy()
    fem.calc_tension()
    fem.set_fps(fps)
    fem.output_txt_result()
    #メインループ
    while step<rate*tmax-1: 
        step = fem.simulate_one_step()
        fem.calc_energy()
        fem.calc_tension()
        if step%(rate/fps) == 0:
            fem.output_txt_result()
            #fem.make_u_graph()
            per = str(step/rate*100)
            #ボタンラベル変更
            #play_movie_button.configure(text=per + '%')
            #f0.update
    #その他ファイル出力 
    #fem.make_u_graph()
    fem.make_energy_graph()
    #fem.make_wav_graph()
    fem.output_tension()
    fem.output_wav()
    fem.output_txt_result()
    #femwave.make_movie(foldername)
    #end = time.time()
    #time = end - start
    #print ("{0}".format(time))
    his = str(length_value[num]) + '|' + str(density_value[num]) + '|' + str(section_area_value[num]) + '|' + \
            str(tension_value[num]) + '|' + str(d1_value[num]) + '|' + str(d3_value[num]) + '|' + \
            str(young_value[num]) + '|' + str(moment_value[num]) + '\n'# + '|' + str(time)
    add_history(his)
 
def call_play_wav():
    ###ボタンラベル変更
    play_wav_button.configure(text='please wait')

    ###ファイル作成/再生
    foldername = str(length.get()) + str(density.get()) + str(section_area.get()) + \
            str(tension.get()) + str(d1.get()) + str(d3.get()) + str(young.get()) + str(moment.get())
    if not os.path.exists(foldername):
        os.mkdir(foldername)
        femwave.make_wav(foldername,length.get(),density.get(),section_area.get(),
                tension.get(),d1.get(),d3.get(),young.get(),moment.get())
        femwave.make_movie(foldername)
        his = str(length.get()) + '|' + str(density.get()) + '|' + str(section_area.get()) + '|' + \
                str(tension.get()) + '|' + str(d1.get()) + '|' + str(d3.get()) + '|' + str(young.get()) + '|' + str(moment.get())# + '|' + str(time)
        add_history(his)
    femwave.play_wav(foldername)
    
    ###ボタンラベル変更
    play_wav_button.configure(text='play wav')

def set_default():
    #初期値
    length.set(0.65)
    density.set(1140)
    section_area.set(0.5188e-6)
    tension.set(60.97)
    d1.set(8.1e-7)
    d3.set(6.4e-4)
    young.set(5.4e9)
    moment.set(0.171e-12)

def change_value(name,value):
    if name=="density":
        density.set(density.get()+value)
    elif name=="section_area":
        section_area.set(section_area.get()+value)
    elif name=="tension":
        tension.set(tension.get()+value)
    elif name=="d1":
        d1.set(d1.get()+value)
    elif name=="d3":
        d3.set(d3.get()+value)
    elif name=="young":
        young.set(young.get()+value)
    elif name=="moment":
        moment.set(moment.get()+value)
    elif name=="length":
        length.set(length.get()*pow(2,value*1.0/12))

def init_history():
    #ファイルを開き一行ずつhistoryへ
    try:
        f = open('history.txt', 'r') 
    except IOError:
        f = open('history.txt', 'w')
        f.write('')
        return 0
    for line in f.readlines():
        history.insert('end', line)
    history.see('end')

def add_history(add):
    #重複確認
    f = open('history.txt', 'r') 
    for line in f.readlines():
        if add == line:
            return 0
    #historyに追加
    history.insert('end', add)
    history.see('end')
    #history.txtに追加
    f = open('history.txt', 'a')
    f.write(add)
    f.close

def set_history():
    set_value = str(history.get(history.curselection()))
    set_list = set_value.split("|")
    length.set(set_list[0])
    density.set(set_list[1])
    section_area.set(set_list[2])
    tension.set(set_list[3])
    d1.set(set_list[4])
    d3.set(set_list[5])
    young.set(set_list[6])
    moment.set(set_list[7])

def set_value(string_num):
    global length_value
    global density_value
    global section_area_value
    global tension_value
    global d1_value
    global d3_value
    global young_value
    global moment_value
    length_value[string_num] = length.get() 
    density_value[string_num] = density.get()
    section_area_value[string_num] = section_area.get()
    tension_value[string_num] = tension.get()
    d1_value[string_num] = d1.get()
    d3_value[string_num] = d3.get()
    young_value[string_num] = young.get()
    moment_value[string_num] = moment.get()

def get_value(string_num):
    global length_value
    global density_value
    global section_area_value
    global tension_value
    global d1_value
    global d3_value
    global young_value
    global moment_value
    length.set(length_value[string_num])
    density.set(density_value[string_num])
    section_area.set(section_area_value[string_num])
    tension.set(tension_value[string_num])
    d1.set(d1_value[string_num])
    d3.set(d3_value[string_num])
    young.set(young_value[string_num])
    moment.set(moment_value[string_num])

def change_string(next_num):
    global string_num
    set_value(string_num)
    get_value(next_num)
    for i in range(6):
        button_text = 'select string' + str(i+1)
        string_button[i].configure(text=button_text)
    string_button[next_num].configure(text='now selected')
    string_num = next_num

def load_setting_file():
    global string_num
    filename = tk.filedialog.askopenfilename(filetypes = [('Text Files', '.txt')])
    num = 0
    try:
        f = open(filename, 'r') 
    except IOError:
        return 0
    for line in f.readlines():
        set_list = line.split("|")
        length.set(set_list[0])
        density.set(set_list[1])
        section_area.set(set_list[2])
        tension.set(set_list[3])
        d1.set(set_list[4])
        d3.set(set_list[5])
        young.set(set_list[6])
        set_value(num)
        num += 1
        if num >= 6:
           break 
    get_value(string_num)

def save_setting_file():
    global string_num
    global length_value
    global density_value
    global section_area_value
    global tension_value
    global d1_value
    global d3_value
    global young_value
    filename = tk.filedialog.asksaveasfilename(filetypes = [('Text Files', '.txt')])
    set_value(string_num)
    f = open(filename, 'w')
    for num in range(6):
        add = str(length_value[num]) + '|' + str(density_value[num]) + '|' + str(section_area_value[num]) + '|' + \
                str(tension_value[num]) + '|' + str(d1_value[num]) + '|' + str(d3_value[num]) + '|' + \
                str(young_value[num]) + '|' + str(moment_value[num]) + '\n'# + '|' + str(time)
        f.write(add)
    f.close

##############################################
#GUI
##############################################
#親ウィンドウ
root = tk.Tk()
root.title("soundmaker")
#root.resizable(width='FALSE', height='FALSE')
root.geometry("900x640")

###弦の選択ボタン
fstring = tk.Frame(root)
string_button = []
string_button.append(tk.Button(fstring, text='now selected', command=lambda:change_string(0)))
string_button[0].pack(side='left')
string_button.append(tk.Button(fstring, text='select string2', command=lambda:change_string(1)))
string_button[1].pack(side='left')
string_button.append(tk.Button(fstring, text='select string3', command=lambda:change_string(2)))
string_button[2].pack(side='left')
string_button.append(tk.Button(fstring, text='select string4', command=lambda:change_string(3)))
string_button[3].pack(side='left')
string_button.append(tk.Button(fstring, text='select string5', command=lambda:change_string(4)))
string_button[4].pack(side='left')
string_button.append(tk.Button(fstring, text='select string6', command=lambda:change_string(5)))
string_button[5].pack(side='left')

fstring.place(relwidth=1.0, relheight=0.05)

string_num = 0 
length_value = np.array([0.65,0.65,0.65,0.65,0.65,0.65])
density_value = np.array([1140,1140,1140,1140,1140,1140])
section_area_value = np.array([0.5188e-6,0.5188e-6,0.5188e-6,0.5188e-6,0.5188e-6,0.5188e-6,0.5188e-6])
tension_value = np.array([60.97,60.97,60.97,60.97,60.97,60.97])
d1_value = np.array([8.1e-7,8.1e-7,8.1e-7,8.1e-7,8.1e-7,8.1e-7])
d3_value = np.array([6.4e-4,6.4e-4,6.4e-4,6.4e-4,6.4e-4,6.4e-4])
young_value = np.array([5.4e9,5.4e9,5.4e9,5.4e9,5.4e9,5.4e9]) 
moment_value = np.array([0.171e-12,0.171e-12,0.171e-12,0.171e-12,0.171e-12,0.171e-12])

###スライドバー
f0 = tk.Frame(root)

fs = tk.Frame(f0)

#length
fs7 = tk.Frame(fs)
len_plus = tk.Button(fs7, text='+', command=lambda:change_value("length",1))
len_minus = tk.Button(fs7, text='-', command=lambda:change_value("length",-1))
length = tk.Scale(fs7, label='弦の長さ', orient='h', from_=0.0, to=2.0, resolution=0.0001)
len_minus.pack(side='left')
len_plus.pack(side='right')
length.pack(fill='both')
fs7.pack(fill='both')

#density
fs0 = tk.Frame(fs)
den_plus = tk.Button(fs0, text='+', command=lambda:change_value("density",1))
den_minus = tk.Button(fs0, text='-', command=lambda:change_value("density",-1))
density = tk.Scale(fs0, label='弦の密度', orient='h', from_=0.0, to=10000, resolution=1.0)
den_minus.pack(side='left')
den_plus.pack(side='right')
density.pack(fill='both')
fs0.pack(fill='both')

#section_area
fs1 = tk.Frame(fs)
sec_plus = tk.Button(fs1, text='+', command=lambda:change_value("section_area",1.0e-10))
sec_minus = tk.Button(fs1, text='-', command=lambda:change_value("section_area",-1.0e-10))
section_area = tk.Scale(fs1, label='弦の断面積', orient='h', from_=0.0, to=1.0e-5, resolution=1.0e-10)
sec_minus.pack(side='left')
sec_plus.pack(side='right')
section_area.pack(fill='both')
fs1.pack(fill='both')

#tension
fs2 = tk.Frame(fs)
ten_plus = tk.Button(fs2, text='+', command=lambda:change_value("tension",1.0e-2))
ten_minus = tk.Button(fs2, text='-', command=lambda:change_value("tension",-1.0e-2))
tension = tk.Scale(fs2, label='弦の張力', orient='h', from_=0.0, to=10000, resolution=1.0e-2)
ten_minus.pack(side='left')
ten_plus.pack(side='right')
tension.pack(fill='both')
fs2.pack(fill='both')

#d1
fs3 = tk.Frame(fs)
d1_plus = tk.Button(fs3, text='+', command=lambda:change_value("d1",1.0e-8))
d1_minus = tk.Button(fs3, text='-', command=lambda:change_value("d1",-1.0e-8))
d1 = tk.Scale(fs3, label='空気抵抗による振動の減衰係数', orient='h', from_=0.0, to=1.0e-2, resolution=1.0e-8)
d1_minus.pack(side='left')
d1_plus.pack(side='right')
d1.pack(fill='both')
fs3.pack(fill='both')

#d3
fs4 = tk.Frame(fs)
d3_plus = tk.Button(fs4, text='+', command=lambda:change_value("d3",1.0e-5))
d3_minus = tk.Button(fs4, text='-', command=lambda:change_value("d3",-1.0e-5))
d3 = tk.Scale(fs4, label='粘弾性による振動の減衰係数', orient='h', from_=0.0, to=1.0e-3, resolution=1.0e-5)
d3_minus.pack(side='left')
d3_plus.pack(side='right')
d3.pack(fill='both')
fs4.pack(fill='both')

#young
fs5 = tk.Frame(fs)
young_plus = tk.Button(fs5, text='+', command=lambda:change_value("young",100000))
young_minus = tk.Button(fs5, text='-', command=lambda:change_value("young",-100000))
young = tk.Scale(fs5, label='弦のヤング率', orient='h', from_=0.0, to=1.0e10, resolution=100000)
young_minus.pack(side='left')
young_plus.pack(side='right')
young.pack(fill='both')
fs5.pack(fill='both')

#moment
fs6 = tk.Frame(fs)
moment_plus = tk.Button(fs6, text='+', command=lambda:change_value("moment",1.0e-15))
moment_minus = tk.Button(fs6, text='-', command=lambda:change_value("moment",-1.0e-15))
moment = tk.Scale(fs6, label='弦の曲げモーメント', orient='h', from_=0.0, to=1.0e-10, resolution=1.0e-15)
moment_minus.pack(side='left')
moment_plus.pack(side='right')
moment.pack(fill='both')
fs6.pack(fill='both')

fs.pack(fill='both', padx=10, pady=14, expand=1)#place(relx=0.03, rely=0.02, relwidth=0.9, relheight=0.85)

set_default()

fb = tk.Frame(f0)
play_movie_button = tk.Button(fb, text='play movie', command=call_play_movie)
play_movie_button.pack(fill='both')
#show_energy_graph_button = tk.Button(fb, text='show energy graph', command=show_energy_graph)
#show_energy_graph_button.pack(fill='both')
exit_button = tk.Button(fb, text='Exit', command=sys.exit)
exit_button.pack(fill='both')

fb.pack(fill='both', expand=1)#place(rely=0.85, relwidth=1.0, relheight=0.2)

f0.place(rely=0.05, relwidth=0.5, relheight=0.95)

###history
#設定
f1 = tk.Frame(root)
label_his = tk.Label(f1, text='history')
history = tk.Listbox(f1)
sb_his = tk.Scrollbar(f1, orient='v', command=history.yview)
history.configure(yscrollcommand = sb_his.set)
#配置
label_his.place(relwidth=1.0, relheight=0.05)
history.place(rely=0.05, relwidth=0.95, relheight=0.95)
sb_his.place(relx=0.95, rely=0.05, relwidth=0.05, relheight=0.95)
f1.place(relx=0.5, rely=0.05, relwidth=0.5, relheight=0.75)

f2 = tk.Frame(root)
set_history_button = tk.Button(f2, text='set', command=set_history)
set_history_button.place(relwidth=0.5, relheight=0.4)
default_button = tk.Button(f2, text='default', command=set_default)
default_button.place(relx=0.5,relwidth=0.5, relheight=0.4)
load_setting_file_button = tk.Button(f2, text='load setting file', command=load_setting_file)
load_setting_file_button.place(rely=0.4, relwidth=1.0, relheight=0.3)
save_setting_file_button = tk.Button(f2, text='save setting file', command=save_setting_file)
save_setting_file_button.place(rely=0.7, relwidth=1.0, relheight=0.3)
f2.place(relx=0.5, rely=0.8, relwidth=0.5, relheight=0.2)

init_history()

root.mainloop()
