import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_regression_line(y_te, y_hat_te, title):
  plt.figure(figsize=(6, 4))
  plt.plot(y_te, y_hat_te, '.')
  v = plt.axis()
  plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
  plt.xlabel(r'$y$')
  plt.ylabel(r'$\^y$')
  plt.grid()
  plt.title(title)
  plt.tight_layout()

def print_performance(E, y, y_hat, nome):
  E_max = E.max()
  E_min = E.min()
  E_mu = E.mean()
  E_sig = E.std()
  E_MSE = np.mean(E ** 2)
  R2 = 1 - E_MSE / (np.var(y))
  c = np.mean((y - y.mean()) * (y_hat - y_hat.mean())) / (y.std() * y_hat.std())
  cols = ['min err', 'max err', 'mean err', 'err std', 'MSE', 'R^2', 'corr_coeff']
  # rows = ['Results']
  p = np.array([E_min, E_max, E_mu, E_sig, E_MSE, R2, c]).reshape(1, -1)
  results = pd.DataFrame(p, columns=cols)
  results.name = nome
  print(f"{results.name}:")
  print(results.to_string(index=False))
  print()


def distance(x1, x2):
  dist = np.sqrt(np.sum((x1-x2)**2))
  return dist


def find_K_closest(k, x , Xtr , ytr):
  Np, Nf = Xtr.shape
  identity_matrix = np.eye(Nf)
  delta = 10e-8
  closest = []
  for i , x_tr in enumerate(Xtr):
    dist = distance(x,x_tr)
    closest.append((dist,i))

  closest.sort(key= lambda n: n[0])
  K_closest = closest[0:k]
  indexes = [elm[1] for elm in K_closest]
  A = Xtr[indexes , :]
  y = ytr[indexes]
  w_hat = np.linalg.inv(A.T @ A + delta*identity_matrix) @ A.T @ y
  y_hat = x.T @ w_hat
  return y_hat


def plot_error_histogram(e,title):
  M = np.max(e)
  m = np.min(e)
  common_bins = np.arange(m, M, (M - m) / 50)
  plt.figure(figsize=(6, 4))
  plt.hist(e, bins=common_bins, density=True, histtype='bar')
  plt.xlabel(r'$e=y-\^y$')
  plt.ylabel(r'$P(e$ in bin$)$')
  plt.grid()
  plt.title(title)
  plt.tight_layout()
  return