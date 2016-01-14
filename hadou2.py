#!i/usr/bin/env python
# -*- coding: UTF-8 -*-
from dolfin import *
import numpy as np
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

#弦の初期値設定(おまじない)
class VolumeInitialCondition2(Expression):
    def eval(self, value, x):
        #value[1] = 0.0
        value[1] = exp(-50*(x[0]-0.5)**2)/50.0
        #self.uvn[i] = exp(-50*((i+1)*self.h-self.length/5)*((i+1)*self.h-self.length/5))/50.0
        value[0] = 0.0
    def value_shape(self):
        return (2,)

dbc = DirichletBC(Vs, Constant(("0.", "0.")), boundary)

'''
#===弦のシミュレーション===#
(vs, us)     = TrialFunctions(Vs)
(psis, phis) = TestFunctions(Vs)

ws_prev = interpolate(VolumeInitialCondition2(), Vs)
vs_prev, us_prev = ws_prev.split()

#up = plot(us_prev)
up = plot(us_prev, range_min=-0.025, range_max=0.025)
interactive()

lhss1 = (2.0*us-dt*vs)*phis*dx
rhss1 = (2.0*us_prev+dt*vs_prev)*phis*dx
lhss2 = 2.0*vs*psis*dx + dt*inner(grad(us), grad(psis))*dx + dt*inner(grad(grad(us)),grad(grad(psis)))*dx
rhss2 = 2.0*vs_prev*psis*dx - dt*inner(grad(us_prev), grad(psis))*dx - dt*inner(grad(grad(us_prev)),grad(grad(psis)))*dx

for step in range(44100):
    # assemble forms and solve
    As = assemble(lhss1 + lhss2)
    bs = assemble(rhss1 + rhss2)
    dbc.apply(As)
    dbc.apply(bs)
    ws = Function(Vs)
    solve(As, ws.vector(), bs)
    # store previous values, increment time step
    t += dt
    vst, ust = ws.split(deepcopy = True)
    vs_prev.assign(vst)
    us_prev.assign(ust)
    # save information
    wave = ws.split()[1]
    wave.rename("WaveFunction", wave.name())
    #output_file << (wave, t)
    if((step+1)%2000==0):
        print(step+1)
        #up = plot(us_prev)#, range_min=-0.1, range_max=1.2)
        up = plot(us_prev, range_min=-0.025, range_max=0.025)
        #up.write_png('png/' + str(step/1000+1).zfill(4)) # + '_' + str(mpiRank))
        interactive()
'''

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
#T1 = interpolate(T0, Vb)
T1 = interpolate(T0, Vb_a)

#Dt = Constant(dt/2.)
lhs1 = (2.0*ub-dt*vb)*phib*dx
rhs1 = (2.0*ub_prev+dt*vb_prev)*phib*dx
lhs2 = 2.0*vb*psib*dx + dt*inner(grad(ub), grad(psib))*dx
rhs2 = 2.0*vb_prev*psib*dx - dt*inner(grad(ub_prev), grad(psib))*dx + T1*psib*dx

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
up = plot(ub_prev, elevate=75.0) #, range_min=-0.1, range_max=0.5)
A = assemble(lhs1 + lhs2)
b = assemble(rhs1 + rhs2)
print(A.array()[0][0])
print(b)
interactive()
up.write_png('png/0000')
#+str(mpiRank))

for step in range(44100):
    # store previous values, increment time step
    t += dt
    #wb_prev.assign(w)
    #v_prev, u_prev = wb_prev.split()
    vbt, ubt = wb.split(deepcopy = True)
    vb_prev.assign(vbt)
    ub_prev.assign(ubt)
    # assemble forms and solve
    tension = float(tension_log[step+1]-tension_log[step])
    T0 = Expression('exp(-0.001*((x[0]-0.0)*(x[0]-0.0)+(x[1]-100.0)*(x[1]-100.0)+(x[2]+150.0)*(x[2]+150.0)))*tension', tension=tension)
    T1 = interpolate(T0, Vb_a)
    lhs1 = (2.0*ub-dt*vb)*phib*dx
    rhs1 = (2.0*ub_prev+dt*vb_prev)*phib*dx
    lhs2 = 2.0*vb*psib*dx + dt*inner(grad(ub), grad(psib))*dx
    rhs2 = 2.0*vb_prev*psib*dx - dt*inner(grad(ub_prev), grad(psib))*dx + T1*psib*dx
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
        up = plot(ub_prev)#, range_min=-0.1, range_max=1.2)
        up.write_png('png/' + str(step/1000+1).zfill(4)) # + '_' + str(mpiRank))
        #interactive()
