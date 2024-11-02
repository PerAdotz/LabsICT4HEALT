import pandas as pd
from Lab0 import minimization as mymin
import numpy as np
import matplotlib.pyplot as plt
from Lab02_Parkinson.Lab02_functions import plot_regression_line

def plot_what(w_hat, title ,regressors):
    Nf = len(regressors)
    nn = np.arange(Nf)
    plt.figure(figsize=(6, 4))
    plt.plot(nn, w_hat, '-o')
    ticks = nn
    plt.xticks(ticks, regressors, rotation=90)
    plt.ylabel(r'$\^w(n)$')
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    # plt.savefig(f'./{title}.png')
    plt.draw()


pd.set_option('display.precision', 3)
X = pd.read_csv('parkinsons_updrs_av.csv') # read the dataset; x is a Pandas dataframe
features = list(X.columns) #list of features in the dataset

Np,Nc = X.shape #Np = number of rows/ptients Nc=number Nf of regressors + 1 (regressand total UPDRS is included)
# print(X.describe().T) # gives the statistical description of the content of each column
# print(X.info())  to have a look at the dataset

seed = 343420
# shuffle da data
Xsh = X.sample(frac=1, replace=False , random_state=seed , axis=0 , ignore_index=True)
# 50% of dataset for training
Ntr = int(Np*0.5)
Nte = Np - Ntr

X_tr = Xsh[0:Ntr]
#evaluate mean ans standard deviation for the training data only
mm = X_tr.mean()
ss = X_tr.std()
#mean and standard deviation of total UPDRS
my = mm['total_UPDRS']
sy = ss['total_UPDRS']

Xsh_norm = (Xsh-mm)/ss  #normalized training data
ysh_norm = Xsh_norm['total_UPDRS'] #regressand only

dropMotor = False
if dropMotor:
    Xsh_norm = Xsh_norm.drop(['total_UPDRS', 'subject#', 'Jitter:DDP', 'Shimmer:DDA' , "motor_UPDRS"], axis=1)
else:
    Xsh_norm = Xsh_norm.drop(['total_UPDRS', 'subject#', 'Jitter:DDP', 'Shimmer:DDA'], axis=1)  # regressors only

regressors=list(Xsh_norm.columns)
Nf = len(regressors) # number of regressors

Xsh_norm=Xsh_norm.values # from dataframe to Ndarray
ysh_norm=ysh_norm.values # from dataframe to Ndarray

X_tr_norm=Xsh_norm[0:Ntr] # regressors for training phase
X_te_norm=Xsh_norm[Ntr:] # regressors for test phase

y_tr_norm=ysh_norm[0:Ntr] # regressand for training phase
y_te_norm=ysh_norm[Ntr:] #regressand for test phase

#%% de-normalize data to put UPDRS in a scale understanable by the doctor
y_tr=y_tr_norm*sy+my
y_te=y_te_norm*sy+my

#LLS regression (recognize collinarity due to X.T @ X )
m = mymin.SolveLLS(y_tr_norm,X_tr_norm) #instantiate the object
m.run() #run LLS
w_hat_LLS = m.what
plot_what(w_hat_LLS,'LLS-Optimized weights',regressors)
y_hat_tr_norm_LLS=X_tr_norm@w_hat_LLS #normalized estimation of total UPDRS for training set
y_hat_te_norm_LLS=X_te_norm@w_hat_LLS #normalized estimation of total UPDRS for test set

#de-normalize data LLS
y_hat_tr_LLS=y_hat_tr_norm_LLS*sy+my
y_hat_te_LLS=y_hat_te_norm_LLS*sy+my

m.plot_yhat('y_hat y comparison on test set LLS', y_hat_te_LLS , y_te , "patients" , "UPDRS")
print("LLS final error:" , m.min)

E_tr_LLS = y_tr - y_hat_tr_LLS
E_te_LLS = y_te - y_hat_te_LLS

m.plot_error_histogram(E_tr_LLS , E_te_LLS , "LLS-Error histograms")
m.print_performance(E_tr_LLS,E_te_LLS,y_tr,y_te,y_hat_tr_LLS,y_hat_te_LLS, "LLS performance")

plot_regression_line(y_te , y_hat_te_LLS , "LLS-test")



# STEEPEST DESCENT regression
NitSteep = 1000
s = mymin.SolveSteepDesc(y_tr_norm,X_tr_norm)
s.run(NitSteep)
s.plot_err('Steepest Descent')
w_hat_SD = s.what
# w_hat_SD = [item[0] for item in s.what] #solo per formattarli meglio
plot_what(w_hat_SD,'STEEPEST DESCENT-Optimized weights',regressors)
y_hat_tr_norm_SD = X_tr_norm@w_hat_SD
y_hat_te_norm_SD = X_te_norm@w_hat_SD

#de-normalize data STEEPEST DESCENT
y_hat_tr_SD=y_hat_tr_norm_SD*sy+my
y_hat_te_SD=y_hat_te_norm_SD*sy+my

s.plot_yhat('y_hat y comparison on test set Steepest Descent', y_hat_te_SD , y_te , "patients" , "UPDRS")
print("Steepest descent final error:" , s.min)


E_tr_SD = y_tr - y_hat_tr_SD
E_te_SD = y_te - y_hat_te_SD

s.plot_error_histogram(E_tr_SD , E_te_SD , "Steepest Descent -Error histograms")
s.print_performance(E_tr_SD,E_te_SD,y_tr,y_te,y_hat_tr_SD,y_hat_te_SD, "Steepest Descent performance")

plot_regression_line(y_te , y_hat_te_SD , "SD-test")


# plt.show()

