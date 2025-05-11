import numpy as np
from matplotlib import pyplot as plt
import sympy as sp
from sympy import Eq, symbols, solve

class func:
    def __init__(self):
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.d = 0.0


x1, x2, x3, x4, y1, y2, y3, y4 = map(float, input("Enter x1, x2, x3, x4, y1, y2, y3, y4: ").split())
#pts = np.array([(x1,y1), (x2,y2), (x3,y3), (x4,y4)])

f1 = func()
f2 = func()
f3 = func()

f1.a, f1.b, f1.c, f1.d = sp.symbols('a1 b1 c1 d1')
f2.a, f2.b, f2.c, f2.d = sp.symbols('a2 b2 c2 d2')
f3.a, f3.b, f3.c, f3.d = sp.symbols('a3 b3 c3 d3')

a1, b1, c1, d1 = sp.symbols('a1 b1 c1 d1')
a2, b2, c2, d2 = sp.symbols('a2 b2 c2 d2')
a3, b3, c3, d3 = sp.symbols('a3 b3 c3 d3')

eq1 = sp.Eq(a1 * x1*x1*x1 + b1 * x1*x1 + c1 * x1 + d1, y1)
eq2 = sp.Eq(a1 * x2*x2*x2 + b1 * x2*x2 + c1 * x2 + d1, y2)
eq3 = sp.Eq(a2 * x2*x2*x2 + b2 * x2*x2 + c2 * x2 + d2, y2)
eq4 = sp.Eq(a2 * x3*x3*x3 + b2 * x3*x3 + c2 * x3 + d2, y3)
eq5 = sp.Eq(a3 * x3*x3*x3 + b3 * x3*x3 + c3 * x3 + d3, y3)
eq6 = sp.Eq(a3 * x4*x4*x4 + b3 * x4*x4 + c3 * x4 + d3, y4)

eq7 = sp.Eq(3*a1*x2*x2 + 2*b1*x2 + c1, 3*a2*x2*x2 + 2*b2*x2 + c2)
eq8 = sp.Eq(3*a2*x3*x3 + 2*b2*x3 + c2, 3*a3*x3*x3 + 2*b3*x3 + c3)

eq9  = sp.Eq(6*a1*x2 + 2*b1, 6*a2*x2 + 2*b2)
eq10 = sp.Eq(6*a3*x3 + 2*b3, 6*a2*x3 + 2*b2)

con1, con2 = sp.symbols('con1 con2') #gotta declare constants as symbols

eq11 = sp.Eq(6*a1*x1 + 2*b1, con1)
eq12 = sp.Eq(6*a3*x4 + 2*b3, con2)
solutions = sp.solve((eq1, eq2, eq3, eq4, eq5, eq6,
                      eq7, eq8, eq9, eq10, eq11, eq12),
                     (a1, b1, c1, d1,
                      a2, b2, c2, d2,
                      a3, b3, c3, d3))

print(solutions)

j =0
sigma = 0
f1_arr_x = []
f1_arr_curv = []
while j <= 100:
    x_val = x1 + j * (x2 - x1) / 100
    f1_arr_x.append(x_val)
    f1_arr_curv.append((6*solutions[a1]*f1_arr_x[j] + 2*solutions[b1])**2/(1+(6*solutions[a1]*f1_arr_x[j] + 2*solutions[b1])**2)**3)
    sigma = sigma + f1_arr_curv[j]
    j+=1
    print(j-1)

while j <= 200:
    f1_arr_x.append(x2 + (j-100) * (x3 - x2) / 100)
    f1_arr_curv.append(((6*solutions[a2]*f1_arr_x[j] + 2*solutions[b2])**2/(1+(6*solutions[a2]*f1_arr_x[j] + 2*solutions[b2])**2)**3))
    sigma = sigma + f1_arr_curv[j]
    j+=1
    print(j-1)

while j <= 300:
    f1_arr_x.append(x3 + (j-200) * (x4 - x3) / 100)
    f1_arr_curv.append(((6*solutions[a3]*f1_arr_x[j] + 2*solutions[b3])**2/(1+(6*solutions[a3]*f1_arr_x[j] + 2*solutions[b3])**2)**3))
    sigma = sigma + f1_arr_curv[j]
    j+=1
    print(j-1)
    

print(sigma)

sigma_func = sp.lambdify((con1, con2), sigma, "numpy")

def numerical_gradient(f, x, y, eps=1e-6):
    df_dx = (f(x + eps, y) - f(x - eps, y)) / (2 * eps)
    df_dy = (f(x, y + eps) - f(x, y - eps)) / (2 * eps)
    return df_dx, df_dy

con1_val = 1.0  # starting point
con2_val = 100.0
learning_rate = 0.01  # step size
tolerance = 1e-6  # how close to zero we want gradient
max_iterations = 100000

for i in range(max_iterations):
    grad1, grad2 = numerical_gradient(sigma_func, con1_val, con2_val)
    
    grad_norm = np.sqrt(grad1**2 + grad2**2)
    
    if grad_norm <tolerance:
        print(f"Converged at iteration {i}")
        break
    
    con1_val = con1_val - learning_rate * grad1
    con2_val = con2_val - learning_rate * grad2

else:
    print("Did not converge within max iterations.")

print(f"Solution: con1 = {con1_val}, con2 = {con2_val}")
print(f"Final gradient: {grad1}, {grad2}")

eq11_2 = sp.Eq(6*a1*x1 + 2*b1, con1_val)
eq12_2 = sp.Eq(6*a3*x4 + 2*b3, con2_val)

solutions = sp.solve((eq1, eq2, eq3, eq4, eq5, eq6,
                      eq7, eq8, eq9, eq10, eq11_2, eq12_2),
                     (a1, b1, c1, d1,
                      a2, b2, c2, d2,
                      a3, b3, c3, d3))

print(solutions)

i = 0
f1_arr_x = np.zeros(3001)
f1_arr_y = np.zeros(3001)
while i <= 1000:
    f1_arr_x[i] = x1 + i * (x2 - x1) / 1000
    f1_arr_y[i] = solutions[a1] * f1_arr_x[i]**3 + solutions[b1] * f1_arr_x[i]**2 + solutions[c1] * f1_arr_x[i] + solutions[d1]
    i += 1

while i <= 2000:
    f1_arr_x[i] = x2 + (i-1000) * (x3 - x2) / 1000
    f1_arr_y[i] = solutions[a2] * f1_arr_x[i]**3 + solutions[b2] * f1_arr_x[i]**2 + solutions[c2] * f1_arr_x[i] + solutions[d2]
    i += 1

while i <= 3000:
    f1_arr_x[i] = x3 + (i-2000) * (x4 - x3) / 1000
    f1_arr_y[i] = solutions[a3] * f1_arr_x[i]**3 + solutions[b3] * f1_arr_x[i]**2 + solutions[c3] * f1_arr_x[i] + solutions[d3]
    i += 1


plt.plot(f1_arr_x, f1_arr_y, label='curve', color='b')  

# Add labels and title
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Simple Line Plot')

# Show legend
plt.legend()

# Show the plot
plt.show()