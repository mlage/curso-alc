import numpy as np
from matplotlib import pyplot as plt
from linear_regression import *

######################################

N_FUNCTIONS = 10            # dimensions
N_POINTS = 200              # domain discretization
ALPHA = 1e-03               # learning rate
BASE_FUNCTIONS = "cosines"  # "polynomials", "exponentials", or "cosines"
DOMAIN = [0.,1.]

######################################

def generate_dataset( basis, n, x=[0,1], add_noise=True ):
    F = np.zeros((2,n))
    F[0] = np.linspace(x[0],x[1],n)
    #F[1,n//2:]=1
    
    if add_noise:
        F[0,np.abs(F[0])<1e-12]=1e-12
        a = np.log10(np.abs(F[0]))
        F[0] += (np.random.rand(n)*4-2)*10**(a-1)
        F[0].sort()
    
    f_n = len(basis)
    f_ids = np.random.rand(np.random.randint(f_n-1)+1)*f_n
    f_ids = np.unique(f_ids.astype('uint32'))
    
    for i in f_ids:
        F[1] += basis[i](F[0])

    if add_noise:
        F[1,np.abs(F[1])<1e-12]=1e-12
        a = np.log10(np.abs(F[1]))
        F[1] += (np.random.rand(n)*4-2)*10**(a-1)

    return F

######################################

if __name__ == '__main__':

    functions = []
    if BASE_FUNCTIONS == "polynomials":
        functions = [ (lambda x, p=i: np.power(x,p)) for i in range(N_FUNCTIONS) ]
    elif BASE_FUNCTIONS == "exponentials":
        functions = [ (lambda x, p=i: np.exp(x*(p-N_FUNCTIONS/2))) for i in range(N_FUNCTIONS) ]
    elif BASE_FUNCTIONS == "cosines":
        functions = [ (lambda x, p=i: np.cos(x*2*np.pi*p*0.25)) for i in range(1,N_FUNCTIONS+1) ]
    else:
        print(f"ERROR. Invalid BASE_FUNCTIONS: {BASE_FUNCTIONS}")
        exit()

    D = generate_dataset(functions, N_POINTS)

    #v, _, _ = gradient_descent_graphic(D,functions,alpha=ALPHA)
    #_ = input('Press any key to exit.')

    print('---Gradient Descent---')
    vg, _, _ = gradient_descent(D,functions,alpha=ALPHA,print_frequency=1000,tol=1e-04)
    print('---Steepest Descent---')
    vs, _, _ = steepest_descent(D,functions,print_frequency=1000,tol=1e-04)
    print('---Newton Raphson---')
    vn, _, _ = newton_raphson(D,functions,print_frequency=1000,tol=1e-04)
    print('---Normal System---')
    vsys = solve_normal_system(D,functions)

    #x = np.linspace(DOMAIN[0],DOMAIN[1],N_POINTS)
    x = np.linspace(np.min(D[0]),np.max(D[0]),N_POINTS)

    plt.scatter(D[0],D[1]) # x, y
    
    fx = sum( w*f(x) for w,f in zip(vg,functions) )
    plt.plot(x,fx,'red',linewidth=2.5)
    fx = sum( w*f(x) for w,f in zip(vs,functions) )
    plt.plot(x,fx,'green',linewidth=2.0)
    fx = sum( w*f(x) for w,f in zip(vn,functions) )
    plt.plot(x,fx,'magenta',linewidth=1.5)
    fx = sum( w*f(x) for w,f in zip(vsys,functions) )
    plt.plot(x,fx,'#f5c211ff',linewidth=1.0)
    
    plt.legend(['dataset','gradient descent','steepest descent','newton-raphson', 'normal system'])
    
    #plt.figure(2)
    #plt.plot( [p*0.25 for p in range(1,N_FUNCTIONS+1) ], v, '--', marker='*' )
    plt.show()
    
######################################
