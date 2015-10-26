from dolfin import *

CRANK_NICOLSON       = 0.5
theta = Constant(CRANK_NICOLSON)

mesh = UnitCubeMesh(10, 10, 10)
Va = FunctionSpace(mesh, "Lagrange", 1)
Vb = FunctionSpace(mesh, "Lagrange", 1)
V  = Va*Vb

t  = 0.
dt = 0.60*mesh.hmin()

def boundary(x, on_boundary):
    return on_boundary

dbc = DirichletBC(V, Constant(("0.", "0.")), boundary)

class VolumeInitialCondition(Expression):
    def eval(self, value, x):
        r = sqrt((x[0] - 0.5)**2 + (x[1] - 0.5)**2 + (x[2] - 0.5)**2)
        value[0] = exp(-(r/0.2)**2)
        value[1] = -2/0.2*r*exp(-(r/0.2)**2)
    def value_shape(self):
        return (2,)

(du, u)           = TrialFunctions(V)
(test_du, test_u) = TestFunctions(V)

w_prev = interpolate(VolumeInitialCondition(), V)
du_prev, u_prev = w_prev.split()

Dt = Constant(dt/2.)
lhs1 = test_u*(u - Dt*du)*dx
rhs1 = test_u*(u_prev + Dt*du_prev)*dx
lhs2 = test_du*du*dx + Dt*inner(grad(u), grad(test_du))*dx
rhs2 = test_du*du_prev*dx - Dt*inner(grad(test_du), grad(u_prev))*dx

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
plot(u_prev)
interactive()

for step in range(20):
    # store previous values, increment time step
    t += dt
    #w_prev.assign(w)
    #du_prev, u_prev = w_prev.split()
    dut, ut = w.split(deepcopy = True)
    du_prev.assign(dut)
    u_prev.assign(ut)

    # assemble forms and solve
    A = assemble(lhs1 + lhs2)
    b = assemble(rhs1 + rhs2)
    #dbc.apply(A)
    #dbc.apply(b)
    solve(A, w.vector(), b)

    # save information
    wave = w.split()[1]
    wave.rename("WaveFunction", wave.name())
    #output_file << (wave, t)
    plot(u_prev)
    interactive()
