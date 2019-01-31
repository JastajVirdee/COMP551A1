# p3.1
# Closed form vs gradient descent evaluations
import linear_regression as lr
import loader_a1 as load
import numpy as np
import timeit

def mse(y, w, x):
    xw = x @ w
    size, _ = y.shape
    #print(np.shape(xw))
    #print(xw[0])
    e = 0
    for i in range(len(y)):
        e += (y[i] - xw[i])**2
    return e/size

train, val, test = load.load_data()

X_train, y_train = load.make_matrix_3_1(train)
y_train = np.array(y_train, ndmin=2)
y_train = np.transpose(y_train)

X_val, y_val = load.make_matrix_3_1(val)
y_val = np.array(y_val, ndmin=2)
y_val = np.transpose(y_val)

X_test, y_test = load.make_matrix_3_1(test)
y_test = np.array(y_test, ndmin=2)
y_test = np.transpose(y_test)

start_cf = timeit.default_timer()
cf_weights = lr.closed_form(X_train, y_train)
stop_cf = timeit.default_timer()

cf_train_err = mse(y_train, cf_weights, X_train)
cf_val_err = mse(y_val, cf_weights, X_val)
print('Closed form no text features:')
print('  Train err:', cf_train_err)
print('  Val err: ', cf_val_err)
print('  CF Time: ', stop_cf-start_cf)

wo = np.ones((4,1))
b = 1000000
n = 200
epsilon = 10**-7
start_gd = timeit.default_timer()
gd_weights = lr.gradient_descent(X_train,y_train,wo,b,n,epsilon)
stop_gd = timeit.default_timer()

gd_train_err = mse(y_train, gd_weights, X_train)
gd_val_err = mse(y_val, gd_weights, X_val)
print('\nGradient descent no text features:')
print('  Train err:', gd_train_err)
print('  Val err:', gd_val_err)
print('  GD Time: ', stop_gd-start_gd)



### part 2: top 60 words

X_train = load.make_matrix_60(train)[0]

X_val= load.make_matrix_60(val) [0]

X_test = load.make_matrix_60(test)[0]

start_cf = timeit.default_timer()
cf_weights = lr.closed_form(X_train, y_train)
stop_cf = timeit.default_timer()

cf_train_err = mse(y_train, cf_weights, X_train)
cf_val_err = mse(y_val, cf_weights, X_val)
print('\nClosed form top 60 words:')
print('  Train err:', cf_train_err)
print('  Val err: ', cf_val_err)
print('  CF Time: ', stop_cf-start_cf)

### part 3 all 160 words

X_train = load.make_matrix_160(train)[0]
X_val = load.make_matrix_160(val)[0]

start_cf = timeit.default_timer()
cf_weights = lr.closed_form(X_train, y_train)
stop_cf = timeit.default_timer()

cf_train_err = mse(y_train, cf_weights, X_train)
cf_val_err = mse(y_val,cf_weights,X_val)
print('\nClosed form all 160 words:')
print('  Train err:', cf_train_err)
print('  Val err: ', cf_val_err)
print('  CF Time: ', stop_cf-start_cf)

### part 4: two new features + top 60 words

X_train = load.make_matrix_60_newfeatures(train)[0]

X_val = load.make_matrix_60_newfeatures(val)[0]

X_test = load.make_matrix_60_newfeatures(test)[0]

start_cf = timeit.default_timer()
cf_weights = lr.closed_form(X_train, y_train)
stop_cf = timeit.default_timer()

cf_train_err = mse(y_train, cf_weights, X_train)
cf_val_err = mse(y_val, cf_weights, X_val)
print('\nClosed form top 60 words and new features:')
print('  Train err:', cf_train_err)
print('  Val err: ', cf_val_err)

### part 5: best performing model on the test set
cf_test_err = mse(y_test, cf_weights, X_test)
print('  Test err:', cf_test_err)
print('  CF Time: ', stop_cf-start_cf)