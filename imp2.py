from dolfin import *
import numpy as np

# Create mesh and define function space
mesh = UnitSquareMesh(6, 4)
#mesh = UnitCubeMesh(6, 4, 5)
V = FunctionSpace(mesh, 'Lagrange', 1)

alpha = 3
beta = 1.2
u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                        alpha=alpha, beta=beta, t=0)
u0.t = 0

def u0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u0, u0_boundary)

u_1 = interpolate(u0, V)

dt = 0.3 #time step
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)

a = u*v*dx + dt*inner(nabla_grad(u), nabla_grad(v))*dx
L = (u_1 + dt*f)*v*dx

A = assemble(a)   # assemble only once, before the time stepping

u = Function(V)   # the unknown at a new time level
T = 2             # total simulation time
t = dt

#output
count = 0
output_txt_file_name = 'result/%04d.txt' % count
'''
for (i, cell) in enumerate(cells(V.mesh())):
    print "Global dofs associated with cell %d: " % i,
    print V.dofmap().cell_dofs(i)
    print "The Dof coordinates:",
    print V.dofmap().tabulate_coordinates(cell)
'''

def outputData(output_txt_file_name, cells,coordinates,u_vec,dof_to_vertex_map_values):
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
    #value of u
    value = np.zeros( (len(u_vec)) )
    for i in range(len(u_vec)):
        value[dof_to_vertex_map_values[i]] = u_vec[i] 
    l = value.tolist()
    text = ','.join(map(str,l)) 
    f.write(text+'\n')
    f.close()

outputData(output_txt_file_name, mesh.cells(), mesh.coordinates(), u_1.vector().array(), dof_to_vertex_map(V))
print dof_to_vertex_map(V)
dtv = dof_to_vertex_map(V)
vtd = vertex_to_dof_map(V)

'''
coords = mesh.coordinates()
x = u_1.vector()
dofs_at_vertices = x[dof_to_vertex_map(V)]

# let's iterate over vertices
for v in vertices(mesh):
    print 'vertex index', v.index()
    print 'at point', v.point().str()
    print 'at coordinates', coords[v.index()]
    print 'dof', dofs_at_vertices[v.index()]
'''

u_vec = u_1.vector().array()
X = V.dofmap().tabulate_all_coordinates(mesh)
X.resize((V.dim(), 2))

print 'dof index | dof coordinate |  dof value'
for i, (x, v) in enumerate(zip(X, u_vec)):
    print i, x, v, dtv[i], u_vec[dtv[i]]

#print V.dofmap().cell_dofs()

File("0000.xml") << u_1

while t <= T:
    b = assemble(L)
    u0.t = t
    bc.apply(A, b)
    solve(A, u.vector(), b)

    t += dt
    u_1.assign(u)

    #output
    count += 1 
    output_txt_file_name = 'result/%04d.txt' % count
    outputData(output_txt_file_name, mesh.cells(), mesh.coordinates(), u_1.vector().array(), dof_to_vertex_map(V))

    '''
    # Plot solution and mesh
    plot(u_1)
    plot(mesh)
    # Dump solution to file in VTK format
    file = File('poisson.pvd')
    file << u
    # Hold plot
    interactive()
    '''
