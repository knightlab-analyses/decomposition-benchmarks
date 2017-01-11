from __future__ import division


try:
    import numpy as np
except ImportError:
    print('Unable to import numpy.')
try:
    from scipy.linalg import sqrtm, inv
except ImportError:
    print('Unable to import scipy.')

import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm, inv

try:
    from pylab import plt
except ImportError:
    print('Unable to import pylab. R_pca.plot_fit() will not work.')
    
    
class eval_metric(object): 


    '''''
    This contains the evaluation metrics for all impuation methods
    '''''


    #RMSE of two matricies 
    def mse(x1, x2, axis=0):

        x1 = np.asanyarray(x1)
        x2 = np.asanyarray(x2)
        return np.mean((x1-x2)**2, axis=axis)
    @staticmethod
    def rmse(x1, imputed, axis=0):

        x1 = np.asanyarray(x1)
        imputed = np.asanyarray(imputed)
        X=np.matrix(np.sqrt(mse(x1, imputed, axis=axis)))
        return abs(X.mean())

    #Frobenius norm of two matricies
    @staticmethod
    def for_error(A,imputed):
        C=A-imputed
        C=LA.norm(C, 'fro')
        A=LA.norm(A, 'fro')
        X=np.matrix(C/A)
        return (X.mean())


    #measure of the convergence of two matricies (will approach zero as they converge)
    def ortho(w):
        return w.dot(inv(sqrtm(w.T.dot(w))))
    @staticmethod
    def omega_error(A,imputed):
        A = np.matrix(A)
        P=ortho(A)
        C=P*(A-imputed)
        C=LA.norm(C, 'fro')
        A=P*A
        A=LA.norm(A, 'fro')
        X=np.matrix(C/A)
        return (X.mean())

    #Aitchison stress
    @staticmethod
    def stress(A,imputed):
        ssm=0
        ssa=0
        for x,y in zip(np.nditer(a),np.nditer(b)):
            ssm_tmp=(x-y)**2
            ssm+=ssm_tmp
            ssa+=(x**2)
        X=ssm/ssa
        return X
