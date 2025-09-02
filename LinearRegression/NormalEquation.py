# formula: w = (Xt.X)^-1.(Xt.y)
# X : [n X m] n rows, m columns
# Xt : [m X n] m rows, n columns
# y : [n X 1] n rows, 1 column
# (Xt.X) : [m X n].[n X m] = [m X m] m columns, m rows
# (Xt.y) : [m X n].[n X 1] = [m X 1] m rows, 1 column
# (Xt.X)^-1 : [m X m]^-1 = [m X m] mrows, m columns
# w = (Xt.X)^-1.(Xt.y) : [m X m].[m X 1]= [m X 1] m rows, 1 column of weights
# important functions 1. transpose, 2. inverse, 3. dot product
import numpy as np
def transpose(x):
    a = len(x)
    b = len(x[0])
    trans = []
    for i in range(b):
        temp = []
        for j in range(a):
            temp.append(x[j][i])
        trans.append(temp)
    return trans
# def dot(x,y):
#     a1 = len(x)
#     b1 = len(x[0])
#     a2 = len(y)
#     b2 = len(y[0])
#     if b1!=a2 :
#         print("Dot product or Multiplication not possible")
#     else :
def dot(xt,x):
    """ dot product of xt and x given only x. vector vs vector"""
    dott=[]
    for j in range(len(xt)):
        sumo  = []
        for i in range(len(xt)):
            k1 = xt[j]
            k2 = xt[i]
            k3 = [a * b for a,b in zip(k1,k2)]
            sum = 0
            for kk in k3:
                sum += kk
            sumo.append(sum)
        dott.append(sumo)
    return dott

def dott(x,y):
    """ matrix x and vector y dot product"""
    result = []
    for row in x:
        dot = sum(a*b for a,b in zip(row, y))
        result.append(dot)
    return result

def inverse(A):
    return np.linalg.inv(A)

X = 2*np.random.rand(100, 1)
X_b = np.c_[np.ones((100, 1)),X]
y = 4 + 3* X + np.random.rand(100, 1)
t1 = dot(transpose(X_b),X_b)
t11 = inverse(t1)
print(t1)
t2 = dott(transpose(X_b),y)
print(t2)
theta = dott(t11,t2)
print("Theta values are : ",theta)
# testing time:
X_test1 = np.random.rand(25,1)*10
X_test = np.c_[np.ones((25, 1)),X_test1]
print(X_test)
ans = X_test.dot(theta)
print(ans)


