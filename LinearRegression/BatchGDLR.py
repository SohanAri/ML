#liner regression baby.
#import numpy as np
# d = np.arange(2,220,3)
# f = (d*3)+4
# x = d
# y = f
# x_train = x[:50]
# y_train = y[:50]
# x_test = x[50:x.size]
# y_test = y[50:y.size]
# m = 1.5
# c = 1.5
# iter = 400
# alpha = 0.00009
# for _ in range(iter):
#     y_pred = m*x_train+c
#     der_m = ((x_train*(y_pred - y_train)).sum())/25
#     der_c = ((y_pred-y_train).sum())/25
#     m = m - alpha*der_m
#     c = c - alpha*der_c
#     cst = (((y_pred-y_train)**2).sum())/50
#     if _%10 ==0 :
#         print("Iteration number: ",_)
#         print("m value : ",m)
#         print("c value : ",c)
#         print("cost value : ",cst)
# # testing time:
# y_test_pred = m*x_test+c
# error = (((y_test_pred-y_test)**2).sum())/y_test.size
# print("The mean Squared Error says: ",error)
# practice done now all reals
#ippudu chustav ra raja code ante ento
import numpy as np
#print(train," ",test)
# train 75 percent test 25

def generate_data():
    x = 2*np.random.rand(100, 1)
    y = 4 + 3* x + np.random.rand(100, 1)
    #return x,y
    # can directly return x,y. But for normalization (helps in conversion because x values are larges takes time to converge) i used below code, it converges at 1300(0.009 lerning rate) , where as with standard x value it converges at 100000(0.000005 learning rate)(and also here m and c converges at different rates so , we need to keep two different alpha for m and c).
    xmean = np.mean(x)
    xstd = np.std(x)
    xnor = (x-xmean)/xstd
    return xnor,y

def train_test_split(x,y,test_size=0.2):
    test = int((test_size)*x.size)
    train = x.size-test
    x_train = x[:train]# here we are choosing the first few , we can also choose random
    x_test = x[train:]
    y_train = y[:train]
    y_test = y[train:]
    return x_train,y_train,x_test,y_test

def compute_cost(y_true, y_pred):
    cost = (((y_true - y_pred)**2).sum())/y_true.size
    return cost

def predict(x ,m ,c):
    y = m*x + c
    return y

def compute_gradients(x, y, y_pred):
    m_grad = (((x*(y_pred-y)).sum())*2)/y.size
    c_grad = (((y_pred-y).sum())*2)/y.size
    return m_grad,c_grad

def update_parameters(m, c, grad_m, grad_c, learning_rate):
    m = m - (learning_rate*grad_m)
    c = c - (learning_rate*grad_c*(10))
    #c = c - (learning_rate*grad_c*(299))
    return m,c

def initialize_parameters():
    return 0,0

def train(x_train, y_train, learning_rate, iterations):
    m,c = initialize_parameters()
    for _ in range(iterations):
        y_pred = predict(x_train, m, c)
        cm, cc = compute_gradients(x_train,y_train,y_pred)
        m,c = update_parameters(m,c,cm,cc,learning_rate)
        loss = compute_cost(y_train,y_pred)
        if _%20==0:
            print("iteration number: ",_)
            print("Value of m: ",m)
            print("Value of c: ",c)
            print("Cost : ",loss)
    return m,c

def evaluate(x_test, y_test,m,c):
    y_pred = predict(x_test, m, c)
    Cost = compute_cost(y_test,y_pred)
    print("The cost on testing data is: ",Cost)

def main():
    x,y = generate_data()
    learning_rate= 0.009
    #learning_rate= 0.000005
    iterations = 1111
    x_train,y_train,x_test,y_test = train_test_split(x,y,0.25) # changes the default value 0.2
    m,c = train(x_train, y_train, learning_rate,iterations)
    test_cost = evaluate(x_test,y_test,m,c)

if __name__== "__main__" : main()


