import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Lab02_functions import *
from Lab0 import minimization as mymin


pd.set_option('display.precision', 3)
X = pd.read_csv('parkinsons_updrs_av.csv') # read the dataset; x is a Pandas dataframe
features = list(X.columns) #list of features in the dataset

Np,Nc = X.shape #Np = number of rows/ptients Nc=number Nf of regressors + 1 (regressand total UPDRS is included)

seed = 343420
# shuffle da data
Xsh = X.sample(frac=1, replace=False , random_state=seed , axis=0 , ignore_index=True)
# 75% of dataset for training (validation + true training)
Ntr_ttd = int(Np*0.5)
Ntr_val = int(Np*0.25)
Nte = Np - ( Ntr_val + Ntr_ttd )


X_tr_ttd = Xsh[0:Ntr_ttd]  #true training data
#evaluate mean ans standard deviation for the true training data only
mm = X_tr_ttd.mean()
ss = X_tr_ttd.std()
#mean and standard deviation of total UPDRS
my = mm['total_UPDRS']
sy = ss['total_UPDRS']

Xsh_norm = (Xsh-mm)/ss  #normalized data
ysh_norm = Xsh_norm['total_UPDRS'] #regressand only


Xsh_norm = Xsh_norm.drop(['total_UPDRS', 'subject#' ,'Jitter:DDP', 'Shimmer:DDA'], axis=1)


regressors=list(Xsh_norm.columns)
Nf = len(regressors) # number of regressors


Xsh_norm=Xsh_norm.values # from dataframe to Ndarray
ysh_norm=ysh_norm.values # from dataframe to Ndarray

X_tr_ttd_norm=Xsh_norm[0:Ntr_ttd] # regressors for true training data
X_tr_val_norm=Xsh_norm[Ntr_ttd: (Ntr_ttd + Ntr_val)] # regressors for validation data
X_te_norm=Xsh_norm[(Ntr_ttd + Ntr_val):] # regressors for test phase

y_tr_ttd_norm=ysh_norm[0:Ntr_ttd] # regressand for true training
y_tr_val_norm = ysh_norm[Ntr_ttd: (Ntr_ttd + Ntr_val)] # regressand for validation data
y_te_norm=ysh_norm[(Ntr_ttd + Ntr_val):] #regressand for test phase

#de-normalize regressand to put UPDRS in a scale understanable by the doctor (useful later)
y_tr_ttd=y_tr_ttd_norm*sy+my
y_tr_val=y_tr_val_norm*sy+my
y_te=y_te_norm*sy+my


#Finding optimal K in Validation Data
Kmax = 75
Kmin = 3
mse_vect = []  #vector containing all MSE for range of K
k_vect = np.arange(Kmin,Kmax)  #range of K to evaluate
for k in k_vect:
    # err_vect = []
    y_hat_vect = []  #vector containing all y_hat from the validation data
    for x in X_tr_val_norm:
        y_hat_norm = find_K_closest(k, x, X_tr_ttd_norm , y_tr_ttd_norm )
        y_hat_vect.append(y_hat_norm)

    E_val = y_tr_val_norm - y_hat_vect  #evaluate error of y_hat
    MSE = np.mean(np.array(E_val) ** 2)
    mse_vect.append(MSE)

K_opt = np.argmin(mse_vect) + Kmin  #optimal K, remember you are not starting from 0 but from Kmin
print('Optimal K:',K_opt)
# print('Optimal MSE:' , mse_vect[(K_opt - Kmin)]) #MSE associated with the optimal K

#plot MSE over K for validation data
plt.figure(figsize=(12, 6))
plt.plot(k_vect, mse_vect, color='r')
plt.plot(K_opt, mse_vect[(K_opt-Kmin)]  , '-o' , color='b' , label='Optimal K')
plt.xlabel('K')
plt.ylabel('MSE')
plt.legend()
plt.grid()
plt.title('Variation of validation MSE over K ')
#---------------------------------------------------------------------------------------------------------#

# LLS with optimal K over Test Data
y_hat_te_norm = []
for x in X_te_norm:
    y_hat = find_K_closest(K_opt, x, X_tr_ttd_norm, y_tr_ttd_norm)
    y_hat_te_norm.append(y_hat)

y_hat_te = np.array(y_hat_te_norm) * sy + my

E_test = y_te - y_hat_te

print_performance(E_test,y_te , y_hat_te, 'Performance on Test Set with optimal K')
plot_regression_line(y_te,y_hat_te, 'LLS-test with optimal K')
plot_error_histogram(E_test , 'LLS error with optimal K')



#----------------------------------------------------------------------------------------------------------------

#Comparison with LLS Regression
Ntr  = int(Np*0.75)
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

Xsh_norm = Xsh_norm.drop(['total_UPDRS', 'subject#' ,'Jitter:DDP', 'Shimmer:DDA'], axis=1)

regressors=list(Xsh_norm.columns)
Nf = len(regressors) # number of regressors

Xsh_norm=Xsh_norm.values # from dataframe to Ndarray
ysh_norm=ysh_norm.values # from dataframe to Ndarray

X_tr_norm=Xsh_norm[0:Ntr] # regressors for training data
X_te_norm=Xsh_norm[Ntr:] # regressors for test phase

y_tr_norm=ysh_norm[0:Ntr] # regressand for training
y_te_norm=ysh_norm[Ntr:] #regressand for test phase

y_te=y_te_norm*sy+my

m = mymin.SolveLLS(y_tr_norm,X_tr_norm) #instantiate the object
m.run() #run LLS
w_hat_LLS = m.what
y_hat_te_norm_LLS=X_te_norm@w_hat_LLS #normalized estimation of total UPDRS for test set

#de-normalize data LLS
y_hat_te_LLS=y_hat_te_norm_LLS*sy+my

E_te_LLS = y_te - y_hat_te_LLS

plot_error_histogram(E_te_LLS , "LLS-Error histograms")
print_performance(E_te_LLS,y_te,y_hat_te_LLS, "LLS performance")
plot_regression_line(y_te , y_hat_te_LLS , "LLS-test")

plt.show()