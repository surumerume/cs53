from dolfin import *

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

print mesh.cells()
print mesh.coordinates()

print u_1.vector().array()

while t <= T:
    b = assemble(L)
    u0.t = t
    bc.apply(A, b)
    solve(A, u.vector(), b)

    t += dt
    u_1.assign(u)

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
