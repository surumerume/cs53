#!i/usr/bin/env python
# -*- coding: UTF-8 -*-
from dolfin import *
import numpy as np

#mesh = UnitSquareMesh(10, 10)
#mesh = UnitCubeMesh(10, 10, 10)
mesh = Mesh("Technology_44.xml.gz")
Va = FunctionSpace(mesh, "Lagrange", 1)
Vb = FunctionSpace(mesh, "Lagrange", 1)
V  = Va*Vb

t=0.0 #初期時刻
tmax=1.0 #終了時刻
rate=44100 #サンプリングレート
dt=1.0/rate #時間刻み
#dt = 0.60*mesh.hmin()
print(dt)

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

#dbc = DirichletBC(V, Constant(("0.", "0.")), boundary)

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

#u0 = 1/100 * x[0]*x[0] + alpha*x[1]*x[1]
#s=2
#T0 = Expression('1/(2*3.1415*s*s) * exp(-(x[0]*x[0] + x[1]*x[1])/(2*s*s)) *100*t',
#                        s=s, t=0)
tension = float(tension_log[0])
T0 = Expression('exp(-0.001*((x[0]-0.0)*(x[0]-0.0)+(x[1]-100.0)*(x[1]-100.0)+(x[2]+150.0)*(x[2]+150.0)))*tension', tension=tension)

(v, u)     = TrialFunctions(V)
(psi, phi) = TestFunctions(V)

w_prev = interpolate(VolumeInitialCondition(), V)
v_prev, u_prev = w_prev.split()
#T1 = interpolate(T0, V)
T1 = interpolate(T0, Va)

#Dt = Constant(dt/2.)
lhs1 = (2.0*u-dt*v)*phi*dx
rhs1 = (2.0*u_prev+dt*v_prev)*phi*dx
lhs2 = 2.0*v*psi*dx + dt*inner(grad(u), grad(psi))*dx
rhs2 = 2.0*v_prev*psi*dx - dt*inner(grad(u_prev), grad(psi))*dx + T1*psi*dx

A = assemble(lhs1 + lhs2)
b = assemble(rhs1 + rhs2)
#dbc.apply(A)
#dbc.apply(b)

w = Function(V)
solve(A, w.vector(), b)
output_file = File("scalar_results.pvd", "compressed")
wave = w.split()[1]
wave.rename("WaveFunction", wave.name())
#output_file << (wave, t)
up = plot(u_prev)#, range_min=-0.1, range_max=1.2)
interactive()
#up.write_png('png/0000')

for step in range(5000):
    # store previous values, increment time step
    t += dt
    #w_prev.assign(w)
    #v_prev, u_prev = w_prev.split()
    vt, ut = w.split(deepcopy = True)
    v_prev.assign(vt)
    u_prev.assign(ut)

    # assemble forms and solve
    tension = float(tension_log[step+1])
    T0 = Expression('exp(-0.001*((x[0]-0.0)*(x[0]-0.0)+(x[1]-100.0)*(x[1]-100.0)+(x[2]+150.0)*(x[2]+150.0)))*tension', tension=tension)
    T1 = interpolate(T0, Va)
    lhs1 = (2.0*u-dt*v)*phi*dx
    rhs1 = (2.0*u_prev+dt*v_prev)*phi*dx
    lhs2 = 2.0*v*psi*dx + dt*inner(grad(u), grad(psi))*dx
    rhs2 = 2.0*v_prev*psi*dx - dt*inner(grad(u_prev), grad(psi))*dx + T1*psi*dx
    A = assemble(lhs1 + lhs2)
    b = assemble(rhs1 + rhs2)
    #dbc.apply(A)
    #dbc.apply(b)
    solve(A, w.vector(), b)

    # save information
    wave = w.split()[1]
    wave.rename("WaveFunction", wave.name())
    #output_file << (wave, t)
    if(step%100==0):
        up = plot(u_prev)#, range_min=-0.1, range_max=1.2)
    #up.write_png('png/' + str(step+1).zfill(4))
        interactive()
