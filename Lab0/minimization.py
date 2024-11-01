import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SolveMinProbl:
    """
    parent class where we define methods to print and plot vector w_hat
    """
    def __init__(self, y = np.ones((3,1)) , X = np.eye(3)):  #np.eye(3) identity matrix 3x3
        self.matr = X  # matrix X (known its the matrix where our data are)
        self.y = y  # column vector y (known , regressand )
        # self.Np = y.shape[0]  # number of rows (number of people)   dont know why prof uses y instead of X
        self.Np = X.shape[0]  # number of rows (number of people)
        self.Nf = X.shape[1]  # number of columns (number of features)
        self.what = np.zeros((self.Nf, 1), dtype=float)  # inizialization of the column vector w_hat to be found where we store the solution (weight of features)
        self.min = 0  # square norm of error ( initially set to zero)
        return

    def plot_what(self, title='Solution'):
        what = self.what  # retrive what
        n = np.arange(self.Nf)  #self.Nf elements starting from 0
        plt.figure()
        plt.plot(n, what)
        plt.xlabel('n')
        plt.ylabel('w_hat(n)')
        plt.title(title)
        plt.grid()
        #plt.show()
        return

    def print_result(self, title):
        vett = np.array(self.what)
        print(title, ':' , end='')
        # print('The optimum weight vector is: ', self.what.T)  #to print a row vector
        print('The optimum weight vector is: ', vett.T)  # to print a row vector
        return

    def plot_yhat(self , title , y_hat , y , xlabel , ylabel):
        n = np.arange(len(y_hat))
        plt.figure(figsize=(12, 6))
        plt.plot(n, y_hat, color='r' , label = 'y_hat')
        plt.plot(n , y , color= 'b' , label='y ' , alpha=0.6)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        return

    def plot_error_histogram(self, e_tr , e_te , title):
        M = np.max([np.max(e_tr), np.max(e_te)])
        m = np.min([np.min(e_tr), np.min(e_te)])
        common_bins = np.arange(m, M, (M - m) / 50)
        e = [e_tr , e_te]
        plt.figure(figsize=(6,4))
        plt.hist(e,bins=common_bins,density=True , histtype='bar' , label=['training' , 'test'])
        plt.xlabel(r'$e=y-\^y$')
        plt.ylabel(r'$P(e$ in bin$)$')
        plt.legend()
        plt.grid()
        plt.title(title)
        plt.tight_layout()
        return

    def print_performance(self, E_tr , E_te , y_tr , y_te , y_hat_tr , y_hat_te , nome):
        E_tr_max = E_tr.max()
        E_tr_min = E_tr.min()
        E_tr_mu = E_tr.mean()
        E_tr_sig = E_tr.std()
        E_tr_MSE = np.mean(E_tr ** 2)
        R2_tr = 1 - E_tr_MSE / (np.var(y_tr))
        c_tr = np.mean((y_tr - y_tr.mean()) * (y_hat_tr - y_hat_tr.mean())) / (y_tr.std() * y_hat_tr.std())
        E_te_max = E_te.max()
        E_te_min = E_te.min()
        E_te_mu = E_te.mean()
        E_te_sig = E_te.std()
        E_te_MSE = np.mean(E_te ** 2)
        R2_te = 1 - E_te_MSE / (np.var(y_te))
        c_te = np.mean((y_te - y_te.mean()) * (y_hat_te - y_hat_te.mean())) / (y_te.std() * y_hat_te.std())
        cols = ['min err', 'max err', 'mean err', 'err std', 'MSE', 'R^2', 'corr_coeff']
        rows = ['Training', 'test']
        p = np.array([
            [E_tr_min, E_tr_max, E_tr_mu, E_tr_sig, E_tr_MSE, R2_tr, c_tr],
            [E_te_min, E_te_max, E_te_mu, E_te_sig, E_te_MSE, R2_te, c_te],
        ])

        results = pd.DataFrame(p, columns=cols, index=rows)
        results.name = nome
        print(f"{results.name}:")
        print(results)
        print()



class SolveLLS(SolveMinProbl):
    """
    subclass to solve minimization problem with Linear Least Squares method
    """
    def run(self):
        X = self.matr # retrive the known matrix X
        y = self.y #retrive the know vector y
        w_hat = np.linalg.inv(X.T@X)@X.T@y # evaluate w_hat (solution)
        self.what = w_hat # store the solution in self.what so that other methods can use it
        self.min = np.linalg.norm(X@w_hat - y )**2  #square norm of error
        return


class SolveGrad(SolveMinProbl):
    """
        subclass to solve minimization problem with Gradient Algorithm
    """
    def run(self, gamma = 1e-3 , Nit = 100): #hyperparameters gamma and nummber of iterations
        self.err = np.empty((0,2) , dtype= float) # empty array with two columns
        self.gamma = gamma #learning rate
        self.Nit = Nit #number of iterations
        X = self.matr #retrive X
        y = self.y #retrive y
        w = np.random.randn(self.Nf, 1) #random start point
        for i in range(Nit):
            grad = 2*X.T@(X@w-y) #gradient of current value
            w = w - gamma*grad # update of w
            sqerr = np.linalg.norm(X@w - y)**2 #square norm of the error
            self.err = np.append(self.err , np.array([[i,sqerr]]), axis= 0)
        self.what = w # store w in what
        self.min = sqerr #store sqerr in self.min
        return

    def plot_err(self, title = 'Square error'):
         err = self.err
         plt.figure()
         plt.plot(err[:,0],err[:,1])
         plt.xlabel('iteration step')
         plt.ylabel('square norm')
         plt.title(f"Square Error {title}")
         plt.grid()
         #plt.show()
         return


class SolveSteepDesc(SolveMinProbl):
    """
     subclass to solve minimization problem with the Steepest Descent method
    """

    def run(self, Nit = 100):
        self.err = np.empty((0, 2), dtype=float)
        X = self.matr
        y = self.y.reshape(-1, 1)
        w = np.random.randn(self.Nf,1)
        HessMatr =  2*(X.T@X)
        grad = 2 * X.T @ (X @ w - y)
        i = 0
        sqerr = np.linalg.norm(X @ w - y) ** 2
        while i < Nit and grad.T @ HessMatr @ grad > 1e-10:
            self.err = np.append(self.err, np.array([[i, sqerr]]), axis=0)
            optGamma = (np.linalg.norm(grad) ** 2) / (grad.T @ HessMatr @ grad)
            w = w - optGamma * grad
            sqerr = np.linalg.norm(X @ w - y) ** 2  # square norm of the error
            grad = 2 * X.T @ (X @ w - y)
            i += 1
        # self.what = w
        self.what = [item[0] for item in w] #solo per formattarli meglio
        self.min = sqerr
        return


    def plot_err(self, title = 'Square error'):
         err = self.err
         plt.figure()

         plt.plot(err[:,0],err[:,1])
         plt.xlabel('iteration step')
         plt.ylabel('square norm')
         plt.title(f"Square Error {title}")
         plt.grid()
         #plt.show()
         return



