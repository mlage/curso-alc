import numpy as np
from matplotlib import pyplot as plt

n_points = 100
x = np.linspace(0,6*np.pi,n_points,dtype='float64')

n_functions = 2 # não alterar! importante para a visualização
functions = [ ( lambda x, st=i: np.cos(x*st) ) for i in range(1,n_functions+1) ]
y = sum( np.random.rand()*f(x) for f in functions )
y_noise = y + 0.5*np.random.rand(n_points)-0.25

plt.ion()
fig1 = plt.figure(1)
plt.scatter(x,y_noise,s=8)
print('Dataset')
_ = input('')
for f in functions:
  plt.plot(x,f(x),linewidth=0.75,linestyle='--')
print('Funções-base')
_ = input('')


A = np.array( [ f(x) for f in functions ] ).T

AtA = A.T @ A
Aty = A.T @ y
Aty_noise = A.T @ y_noise

fig2 = plt.figure(2)
plt.arrow(0,0,AtA[0,0],AtA[1,0], color='red', linewidth=1.25, head_width=2, head_length=1)
plt.arrow(0,0,AtA[0,1],AtA[1,1], color='green', linewidth=1.25, head_width=2, head_length=1)
plt.arrow(0,0,Aty_noise[0],Aty_noise[1], color='blue', linewidth=1.25, head_width=2, head_length=1)
plt.arrow(0,0,Aty[0],Aty[1], color='magenta', linewidth=1.25, head_width=2, head_length=1)
plt.grid(True)
plt.legend(['1a coluna de AtA','2a coluna de AtA','Aty','Aty (sem noise)'])
print('AtA = Aty')
_ = input('')

weights = np.linalg.solve(AtA,Aty_noise)
y_fit = sum( w*f(x) for w,f in zip(weights,functions) )

plt.figure(fig1)
plt.plot(x,y_fit,linewidth=1.5)

plt.ioff()
print('Função ajustada (vem da solução do sistema normal)')
_ = input('')
