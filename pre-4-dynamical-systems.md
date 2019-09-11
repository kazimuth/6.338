# Introduction to Continuous Dynamics General Linear Behavior and the Linearization of Nonlinear Systems By Chris Rackauckas
abstract:
    linear systems: completely determined by eigenvalues of matrix of coefficients
    Stable Manifold + Hartman Grobman theorems: linearize nonlinear systems to understand
    applications

X = (x1 ... xn)^T
system of differential equations for X:
    X' = F(t, X) = (f1(x1 .. xn) .. fn (x1 .. xn))^T

* we're discussing a single vector in a phase space, not a field

autonomous systems (no explicit time dependence):
    F(t, X) denoted F(X)

solution X(t) to the system satisfies X'(t) = F(X(t))
    intuitive sense: derivative of solution must correspond plugging solution into system of difeqs
    pick a starting point, have behavior of system for all time

flow: \phi : R x R -> R
    \phi(t, X0) is the coordinates of the object that started at X0 at time t

Equilibrium point X0: F(X0) = 0
Any time a system gets to an equilibrium point, it will stay there forever.

To understand systems, understand equilibrium
Different types of behavior around equilibrium for linear systems

linear system: each f_i can be written as a linear combination of the variables, f_i(X) = ai1 x1 + ... + xin xn
    more intuitively: define matrix A = ( (a11 .. an1)^T ... (a1n .. ann)^T )
    powerful formulation

exponential of a matrix: for nxn matrix , [equation i dont wanna write]

theorem: let A: nxn; then unique solution of IVP X' = AX with X(0) = X0 is `X(t) = e^{At} X0`

* where At is matrix scaling?

proposition: if det A != 0, a unique equilibrium point exists
    because equilibrium point X0 has `A X0 = 0`, has unique solution when `det A != 0`

proposition: if det A = 0, there exists a straight line of equilibrium through the origin
    because for det A = 0, null space must be at least a line (possibly higher-dimensional)

theorem: suppose V1 .. Vn linearly independent eigenvectors, \l1 .. \ln eigenvalues.
    then X' = AX has solution X(t) = \alpha1 exp(\l1 t) V1 + .. + \alphan expr(\ln t) Vn.

because: write matrix in eigenbasis {V1,...,Vn} to obtain diagonalized matrix of eigenvalues. guess solution, check, hey it works.

* "write matrix in eigenbasis": what does that mean?

... close reading ends

* note: eigenvectors of rotation matrix are complex (but must always give real outputs for real matrix)

* imagine 1x1 matrix with negative value: flips inputs backwards! eigenvector 1, eigenvalue -1

don't want repeated eigenvalues...

"most"  matrices have distinct eigenvalues...




