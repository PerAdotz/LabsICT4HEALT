import numpy as np
import matplotlib.pyplot as plt

Np = 5 #number of rows (patients)
Nf = 4 #number of column (features)
np.random.seed(3) # seed to generate the random values
A = np.random.randn(Np,Nf)
w  = np.random.randn(Nf,1) #column vector with weights

y = A@w #multiplication row * column so shape of y is (Np,1)

# lets pretend we dont know w and apply the linear least square method LLS

ATA = A.T@A #A transpose times A , to have the transpose of a matrix X write X.T
ATAinv = np.linalg.inv(ATA)

ATy = A.T@y

w_hat = ATAinv@ATy

#w_hat = np.linalg.inv(A.T@A)@A.T@y # all code in one line
#w_hat = np.linalg.pinv(A)@y # all code in one line with pseudo inverse

print("Original w",w.T)
print("Estimated w",w_hat.T)

e = y - A@w_hat
print("Error vector:",e.T)  #numerical errors inside computer, it should be 0 but it is not
print("Square norm:", np.linalg.norm(e)**2)

plt.figure()
plt.plot(w , 'x' , label='w')
plt.plot(w_hat , '+' , label='w_hat')
plt.xlabel('n')
plt.ylabel('w(n)')
plt.legend()
plt.grid()
plt.title('Comparison between w and w_hat')
plt.show()


