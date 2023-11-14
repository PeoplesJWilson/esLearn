from scipy.optimize import minimize
from scipy.special import softmax
from functools import partial
from numpy import ones, zeros, hstack, argmax, log

class LogisticRegression:

    def __init__(self, fit_intercept = True, method = "BFGS"):
        self.fit_intercept = fit_intercept
        self.method = method
        self._coef = None
        self._training_log = None
        self._num_classes = None

        
    def fit(self, X, Y):
        N = X.shape[0]

        if self.fit_intercept:
            X = hstack((ones((N,1)), X))
        
        p = X.shape[1]
        K = Y.shape[1]
        self._num_classes = K

        training_loss = partial(self.__loss, X, Y)
        training_log = minimize(training_loss, ones((K*p,)), method = self.method)

        self._training_log = training_log
        self._coef = training_log['x'].reshape((p,K))
    
    def predict_proba(self, x):
        if self.fit_intercept:
            N = x.shape[0]
            x = hstack((ones((x,1)), x))
        if self._coef is None:
            error_msg = "Error: must fit model before calling predict methods"
            print(error_msg)
            return error_msg

        return softmax(-x @ self._coef, axis = 1)
    
    def predict(self, x):
        N = x.shape[0]
        K = self._num_classes

        y_prob = self.predict_proba(x)
        index = argmax(y_prob, axis = 1)

        y_preds  = zeros((N,K))
        for i in range(N):
            y_preds[i,index[i]] = 1

        return y_preds

    def __loss(self, X, Y, beta):
        K = Y.shape[1]
        p = X.shape[1]
        beta = beta.reshape((p,K))

        P = softmax(-X @ beta, axis = 1)
        L = -log(P)*Y
        return L.sum(axis=1).sum()