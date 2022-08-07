import math, copy
import numpy as np
import matplotlib.pyplot as plt

# Data Set
x_train = np.array([1.0, 2.0, 2.5, 2.5, 3.0, 4.0])
y_train = np.array([300.0, 500.0, 600.0, 620.0, 800.0, 850.0])

# Cost Function
def compute_cost(x, y, w, b):
  m = x.shape[0]
  cost = 0
  for i in range(m):
    f_wb = w*x[i]+b
    cost = cost + (f_wb-y[i])**2
  total_cost = (1/(2*m))*cost
  
  # For computing the gradient
def compute_gradient(x, y, w, b):
  m = x.shape[0]
  dj_dw = 0
  dj_db = 0
  for i in range(m):
    f_wb = w*x[i]+b
    dj_dw += (f_wb-y[i])*x[i]
    dj_db += f_wb-y[i]
  dj_dw = dj_dw/m
  return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, gradient_function):
    b = b_in
    w = w_in
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)
        # Update Parameters using equation (3) above
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
    return w, b
  
  w_init = 0
b_init = 0
lrate = 2.5e-2
iter = 1000
w_fin, b_fin = gradient_descent(x_train, y_train, w_init, b_init, lrate, iter, compute_gradient)

# Prediction
print(f"1000 sqft house prediction {w_fin*1.0 + b_fin:0.1f} Thousand dollars")
print(f"2500 sqft house prediction {w_fin*2.5 + b_fin:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_fin*2.0 + b_fin:0.1f} Thousand dollars")

# Visualization
def fun_wb(x, y, w, b):
  m = x.shape[0]
  f_wb = np.zeros(m)
  for i in range(m):
    y_hat = w*x[i]+b
    f_wb[i] = y_hat
  return f_wb
plt.scatter(x_train, y_train, c='r')
fin_f_wb = fun_wb(x_train, y_train, w_fin, b_fin)
plt.plot(x_train, fin_f_wb, c='b')
plt.xlabel("Area in 1000 sq ft")
plt.ylabel("Price in $1000")
plt.show()
