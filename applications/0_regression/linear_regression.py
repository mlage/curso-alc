import numpy as np
from matplotlib import pyplot as plt

######################################

def solve_normal_system(Y, basis):
    x = np.array( Y[0], dtype='float32' )
    y = np.array( Y[1], dtype='float32' )
    A = np.zeros( (x.size,len(basis)) , dtype='float32' )
    for i, f in enumerate(basis):
      A[:,i] = f(x)[:]
    AtA = A.T @ A
    Aty = A.T @ y
    return np.linalg.solve(AtA,Aty)
    
######################################

def gradient_descent(Y, basis, v=None, alpha=1e-04, tol=1e-04, maxit=100_000, print_frequency=100 ):
    v = v if v is not None else np.zeros(len(basis))
    d = np.zeros_like(v)
    x = Y[0]
    r = np.array(Y[1]) - sum( w*f(x) for w, f in zip(v, basis))
    delta_0 = np.linalg.norm(Y[1])
    delta = 1.0  
    for ii in range(maxit+1):
        d[:] = np.array( [ np.dot(r,f(x)) for f in basis ] )
        delta = np.linalg.norm(d) / delta_0
        if delta <= tol:
            print(f'Iteration {ii}, norm(grad) = {delta:.4e}')
            return v, ii, delta
        if (ii % print_frequency)==0:
            print(f'Iteration {ii}, norm(grad) = {delta:.4e}')
        v += alpha*d
        r -= alpha*sum( w*f(x) for w, f in zip(d, basis))
    return v, maxit, delta
    
######################################

def steepest_descent(Y, basis, v=None, tol=1e-04, maxit=100_000, print_frequency=100 ):
    v = v if v is not None else np.zeros(len(basis))
    d = np.zeros_like(v)
    s = np.zeros_like(v)
    x = Y[0]
    r = np.array(Y[1]) - sum( w*f(x) for w, f in zip(v, basis))
    q = np.zeros_like(r)
    delta_0 = np.linalg.norm(Y[1])
    delta = 1.0   
    for ii in range(maxit+1):
        d[:] = np.array( [ np.dot(r,f(x)) for f in basis ] )
        delta = np.linalg.norm(d) / delta_0
        if delta <= tol:
            print(f'Iteration {ii}, norm(grad) = {delta:.4e}')
            return v, ii, delta
        if (ii % print_frequency)==0:
            print(f'Iteration {ii}, norm(grad) = {delta:.4e}')
        q[:] = sum( w*f(x) for w, f in zip(d, basis))
        s[:] = np.array( [ np.dot(q,f(x)) for f in basis ] )
        alpha = np.dot(d,d) / np.dot(d,s)
        v += alpha*d
        r -= alpha*q
        
    return v, maxit, delta

######################################

def newton_raphson(Y, basis, v=None, tol=1e-04, maxit=100_000, print_frequency=100 ):
    v = v if v is not None else np.zeros(len(basis))
    d = np.zeros_like(v)
    x = Y[0]
    r = np.array(Y[1]) - sum( w*f(x) for w, f in zip(v, basis))
    J = np.zeros((v.size,v.size))
    delta_0 = np.linalg.norm(Y[1])
    delta = 1.0  
    for ii in range(maxit+1):
        d[:] = np.array( [ np.dot(r,f(x)) for f in basis ] )
        delta = np.linalg.norm(d) / delta_0
        if delta <= tol:
            print(f'Iteration {ii}, norm(grad) = {delta:.4e}')
            return v, ii, delta
        if (ii % print_frequency)==0:
            print(f'Iteration {ii}, norm(grad) = {delta:.4e}')
        for jj, fj in enumerate(basis):
            for kk in range(jj,len(basis)):
                fk = basis[kk]
                J[jj,kk] = np.dot(fj(x),fk(x))
                J[kk,jj] = J[jj,kk]
        p = np.linalg.solve(J,d)
        v += p
        r -= sum( w*f(x) for w, f in zip(p, basis))
    return v, maxit, delta
    
######################################

def gradient_descent_graphic(Y, basis, v=None, alpha=1e-04, tol=1e-04, maxit=100_000, animation_frequency=100, print_frequency=100 ):
    plt.ion()
    fig = plt.figure(1)
    plt.scatter(Y[0],Y[1])
    
    v = v if v is not None else np.zeros(len(basis))
    d = np.zeros_like(v)
    x = Y[0]
    r = np.array(Y[1]) - sum( w*f(x) for w, f in zip(v, basis))
    delta_0 = np.linalg.norm(Y[1])
    delta = 1.0
        
    fx = sum( w*f(x) for w,f in zip(v,basis) )
    line_plot = plt.plot(x,fx,'red')[0]
    
    for ii in range(maxit+1):
        d[:] = np.array( [ np.dot(r,f(x)) for f in basis ] )
        delta = np.linalg.norm(d) / delta_0
        if delta <= tol:
            print(f'Iteration {ii}, norm(grad) = {delta:.4e}')
            plt.ioff()
            return v, ii, delta
        if (ii % print_frequency)==0:
            print(f'Iteration {ii}, norm(grad) = {delta:.4e}')
        if (ii % animation_frequency)==0 or ii < animation_frequency:
            fx = sum( w*f(x) for w,f in zip(v,basis) )
            line_plot.set_ydata(fx)
            plt.draw()
            fig.canvas.flush_events()
            if ii < animation_frequency:
                _ = input(f'Iteration {ii}, norm(grad) = {delta:.4e}')
        v += alpha*d
        r -= alpha*sum( w*f(x) for w, f in zip(d, basis))
    
    fx = sum( w*f(x) for w,f in zip(v,basis) )
    line_plot.set_ydata(fx)
    plt.draw()
    fig.canvas.flush_events()
    plt.ioff()
    
    return v, maxit, delta
