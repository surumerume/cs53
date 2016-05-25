#!i/usr/bin/env python
# -*- coding: UTF-8 -*-
#張力ひとつ前との差分版
from dolfin import *
import numpy as np
import scipy.io.wavfile
#from petsc4py import PETSc
#from mpi4py import MPI

#mesh = UnitSquareMesh(10, 10)
#mesh = UnitCubeMesh(10, 10, 10)

#弦のメッシュ及び関数空間
mesh_s = IntervalMesh(30,0,1)
#mesh_s = Mesh("Technology_44.xml.gz")
Vs_a = FunctionSpace(mesh_s, "Lagrange", 2)
Vs_b = FunctionSpace(mesh_s, "Lagrange", 2)
Vs = Vs_a*Vs_b

#ボディのメッシュ及び関数空間
mesh_b = Mesh("Technology_44.xml.gz")
Vb_a = FunctionSpace(mesh_b, "Lagrange", 1)
Vb_b = FunctionSpace(mesh_b, "Lagrange", 1)
Vb  = Vb_a*Vb_b

t=0.0 #初期時刻
tmax=1.0 #終了時刻
rate=44100 #サンプリングレート
wav_data = np.zeros( (tmax*rate) )    #wav書き出し用
dt=1.0/rate #時間刻み
#dt = 0.60*mesh.hmin()
print(dt)

#comm = mpi_comm_world() 
#mpiRank = MPI.rank(comm)
#print(mpiRank)

input_txt_file_name = 'tension_log.txt'
f = open(input_txt_file_name, 'r')
line = f.readline()
l=[]
while line:
    l.append(line) 
    line = f.readline()
tension_log = np.asarray(l, dtype=np.float32)
f.close

f = open('wave_log.txt', 'w')

def boundary(x, on_boundary):
    return on_boundary

#dbc = DirichletBC(Vs, Constant(("0.", "0.")), boundary)

#ボディの初期値設定(おまじない)
class VolumeInitialCondition(Expression):
    def eval(self, value, x):
        #r = sqrt((x[0] - 0.5)**2 + (x[1] - 0.5)**2 + (x[2] - 0.5)**2)
        #r = sqrt((x[0] - 0.5)**2 + (x[1] - 0.5)**2)
        #value[1] = exp(-sqrt((x[0]-1.5)**2+(x[1]-0.5)**2+(x[2]+1.5)**2)**2)
        #value[1] = exp(-0.0001*sqrt((x[0]-0.0)**2+(x[1]-100.0)**2+(x[2]+150.0)**2)**2)
        #value[1] = -2/0.2*r*exp(-(r/0.2)**2)
        #value[1] = 1 + x[0]*x[0]
        value[1] = 0.0
        value[0] = 0.0
    def value_shape(self):
        return (2,)

dbc = DirichletBC(Vs, Constant(("0.", "0.")), boundary)

#===ボディのシミュレーション===#
#u0 = 1/100 * x[0]*x[0] + alpha*x[1]*x[1]
#s=2
#T0 = Expression('1/(2*3.1415*s*s) * exp(-(x[0]*x[0] + x[1]*x[1])/(2*s*s)) *100*t',
#                        s=s, t=0)
tension = float(tension_log[0] - tension_log[tension_log.size-1])
T0 = Expression('exp(-0.001*((x[0]-0.0)*(x[0]-0.0)+(x[1]-100.0)*(x[1]-100.0)+(x[2]+150.0)*(x[2]+150.0)))*tension', tension=tension)

(vb, ub)     = TrialFunctions(Vb)
(psib, phib) = TestFunctions(Vb)

wb_prev = interpolate(VolumeInitialCondition(), Vb)
vb_prev, ub_prev = wb_prev.split()
wav_data[0] = wb_prev.vector().array()[3128*2]
f.write(str(wav_data[0])+'\n')
#T1 = interpolate(T0, Vb)
T1 = interpolate(T0, Vb_a)
rho = 350
aaa = 2.9e-3
D = 100e6

lhs1 = (2.0*ub-dt*vb)*phib*dx
rhs1 = (2.0*ub_prev+dt*vb_prev)*phib*dx
lhs2 = 2.0*vb*psib*dx + D*aaa**2/rho*1000*1000*1000*1000*dt*inner(grad(ub), grad(psib))*dx
rhs2 = 2.0*vb_prev*psib*dx - D*aaa**2/rho*1000*1000*1000*1000*dt*inner(grad(ub_prev), grad(psib))*dx + 1.0/(rho*aaa)*T1*psib*dx

A = assemble(lhs1 + lhs2)
b = assemble(rhs1 + rhs2)
#dbc.apply(A)
#dbc.apply(b)

wb = Function(Vb)
solve(A, wb.vector(), b)
output_file = File("scalar_results.pvd", "compressed")
wave = wb.split()[1]
wave.rename("WaveFunction", wave.name())
#output_file << (wave, t)
print("hoge")
up = plot(ub_prev, elevate=75.0 , range_min=-0.03, range_max=0.03)
interactive()
up.write_png('png/0000')
print("hoge")
A = assemble(lhs1 + lhs2)
b = assemble(rhs1 + rhs2)
#print(A.array()[0][0])
#print(b)
#+str(mpiRank))

print("mainloop")
for step in range(44100-1):
    # store previous values, increment time step
    t += dt
    #wb_prev.assign(w)
    #v_prev, u_prev = wb_prev.split()
    vbt, ubt = wb.split(deepcopy = True)
    vb_prev.assign(vbt)
    ub_prev.assign(ubt)
    '''
    if step == 20:
        for i in range(wb.vector().array().size):
            if wb.vector().array()[i] != 0.0:
                print(str(i) + ':' + str(wb.vector().array()[i]))
    '''
    wav_data[step+1] = wb.vector().array()[3128*2]
    f.write(str(wav_data[step+1])+'\n')
    # assemble forms and solve
    tension = float(tension_log[step+1]-tension_log[step])
    T0 = Expression('exp(-0.001*((x[0]-0.0)*(x[0]-0.0)+(x[1]-100.0)*(x[1]-100.0)+(x[2]+150.0)*(x[2]+150.0)))*tension', tension=tension)
    T1 = interpolate(T0, Vb_a)
    lhs1 = (2.0*ub-dt*vb)*phib*dx
    rhs1 = (2.0*ub_prev+dt*vb_prev)*phib*dx
    lhs2 = 2.0*vb*psib*dx + D*aaa**2/rho*1000*1000*1000*1000*dt*inner(grad(ub), grad(psib))*dx
    rhs2 = 2.0*vb_prev*psib*dx - D*aaa**2/rho*1000*1000*1000*1000*dt*inner(grad(ub_prev), grad(psib))*dx + 1.0/(rho*aaa)*T1*psib*dx
    A = assemble(lhs1 + lhs2)
    b = assemble(rhs1 + rhs2)
    #print(A)
    #print(b)
    #dbc.apply(A)
    #dbc.apply(b)
    solve(A, wb.vector(), b)
    # save information
    wave = wb.split()[1]
    wave.rename("WaveFunction", wave.name())
    #output_file << (wave, t)
    if(step%1000==0):
        print(step)
        print(wav_data[step+1])
        #up = plot(ub_prev, range_min=-0.1, range_max=0.1)
        up = plot(ub_prev)#, range_min=-0.1, range_max=1.2)
        up.write_png('png/' + str(step/1000+1).zfill(4)) # + '_' + str(mpiRank))
        #interactive()

f.close()
wav_data = (wav_data/np.amax(wav_data))*32767
wav_data = np.asarray(wav_data, dtype=np.int16)
#print ("{0}".format(wav_data))
output_wav_file_name = "result.wav"
scipy.io.wavfile.write(output_wav_file_name,rate,wav_data)
