import minimization as mymin
import numpy as np
import matplotlib.pyplot as plt

Np = 100
Nf = 4
X = np.random.randn(Np,Nf)
w = np.random.randn(Nf,1)
y = X@w #column vector y



m = mymin.SolveLLS(y,X) #instantiate the object
m.run() #run LLS
m.print_result('LLS') #print the result (inherited method)
m.plot_what('LLS') #plot w_hat (inherited method)

Nit = 150
gamma = 1e-3
g = mymin.SolveGrad(y,X)
g.run(gamma,Nit)
g.print_result('Gradient Algorithm')
g.plot_what('Gradient Algorithm')
g.plot_err('Gradient Algorithm')

NitSteep = 150
s = mymin.SolveSteepDesc(y,X)
s.run(NitSteep)
s.print_result('Steepest Descent')
s.plot_what('Steepest Descent')
s.plot_err('Steepest Descent')

plt.show()