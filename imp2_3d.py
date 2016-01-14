from dolfin import *
import numpy as np

# Create mesh and define function space
#mesh = UnitSquareMesh(6, 4)
#mesh = UnitCubeMesh(6, 4, 5)
mesh = Mesh("Technology_44.xml.gz")
V = FunctionSpace(mesh, 'Lagrange', 1)

alpha = 3 
beta = 1.2 
c = 2
u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1]',
                        alpha=alpha)
#u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
#                        alpha=alpha, beta=beta, t=0)
#u2 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + c*x[2]*x[2]',
#                        alpha=alpha, beta=beta, t=0, c=c)
u2 = Constant(1)
u0.t = 0

def u0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u0, u0_boundary)

u_1 = interpolate(u0, V)

plot(u_1)
interactive()
#file = File('poisson.pvd')
#file << u_1

dt = 0.3 #time step
u = TrialFunction(V)
phi = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)
#f = Constant(0)

a = u*phi*dx + dt*inner(nabla_grad(u), nabla_grad(phi))*dx
L = u_1*phi*dx

A = assemble(a)   # assemble only once, before the time stepping

w = Function(V)   # the unknown at a new time level
T = 2.0             # total simulation time
t = dt

#output
count = 0
output_txt_file_name = 'result/%04d.txt' % count

def outputData(output_txt_file_name, cells,coordinates,u_vec):#,dof_to_vertex_map_values):
    f = open(output_txt_file_name, 'a')
    #meshdata
    l = []
    for i in range(len(cells)):
        for j in range(len(cells[0])):
            l.append(cells[i][j]) 
    text = ','.join(map(str,l)) 
    f.write(text+'\n')
    #coordinates
    l = []
    for i in range(len(coordinates)):
        for j in range(len(coordinates[0])):
            l.append(coordinates[i][j])
    text = ','.join(map(str,l)) 
    f.write(text+'\n')
    '''
    #value of u
    value = np.zeros( (len(u_vec)) )
    for i in range(len(u_vec)):
        value[dof_to_vertex_map_values[i]] = u_vec[i] 
    l = value.tolist()
    text = ','.join(map(str,l)) 
    f.write(text+'\n')
    '''
    f.close()

while t <= T:
    b = assemble(L)
    u0.t = t
    bc.apply(A, b)
    solve(A, w.vector(), b)

    t += dt
    u_1.assign(w)

    #output
    count += 1 
    output_txt_file_name = 'result/%04d.txt' % count
    #outputData(output_txt_file_name, mesh.cells(), mesh.coordinates(), u_1.vector().array())#, dof_to_vertex_map(V))

    # Plot solution and mesh
    plot(u_1)
    #plot(mesh)
    # Dump solution to file in VTK format
    #file = File('poisson.pvd')
    #file << u
    # Hold plot
    interactive()
