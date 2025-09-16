import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse

import LU
import steepest_descent
import conjugate_gradient

# tamanho do grid. atenção: matriz será nodes x nodes
rows = 20
cols = 20
nodes = rows*cols

# constrói matriz de adjacência de um grafo análogo a uma grade (grid). Assume numeração por linha (row major):
# Exemplo 3x4:
#  0 -- 1 -- 2 -- 3
#  |    |    |    |
#  4 -- 5 -- 6 -- 7
#  |    |    |    |
#  8 -- 9 --10 --11
adjacency = np.zeros((nodes,nodes),dtype='float32')
ii = np.arange(1,nodes)
jj = np.arange(0,nodes-1)
adjacency[ii,jj] = 1.
adjacency[jj,ii] = 1.
adjacency[ii[cols-1::cols],jj[cols-1::cols]] = 0.
adjacency[jj[cols-1::cols],ii[cols-1::cols]] = 0.
ii = np.arange(cols,nodes)
jj = np.arange(0,nodes-cols)
adjacency[ii,jj] = 1.
adjacency[jj,ii] = 1.

# monta matriz laplaciana de um grafo
# esta matriz descreve o equilíbrio em cada nó do grafo: "a soma de tudo que chegou e saiu do nó tem que zer 0"
degrees = np.sum(adjacency,axis=1)
laplacian = np.diag(degrees) - adjacency

# monta sistema de equações, "descartando" as bordas de cima e de baixo (primeira e última linhas)
# essas bordas serão usadas como condição de contorno (valores conhecidos nos nós destas linhas)
A = laplacian[cols:(rows-1)*cols,cols:(rows-1)*cols]

# monta lado direito: "sabendo o quanto chegou/saiu em alguns nós (primeira e última linha), o quanto deve ser 'compensado' pelo demais nós"
b = np.zeros((rows-2)*cols)
b[:] = -np.sum(laplacian[cols:(rows-1)*cols,:cols]*2.0,axis=1)-np.sum(laplacian[cols:(rows-1)*cols,-cols:]*-2.0,axis=1) # aplica x=2 em cima, x=-2 embaixo
#b[cols*(rows//2-rows//10):cols*(rows//2+rows//10):cols] = -0.1 # "vazamento" em uma das "bordas" do grafo (malha)

# resolve sistema Ax=b de formas variadas
x_LU = LU.solve(A,b)
x_SD = steepest_descent.solve(A,b,tol=1e-03,maxit=2000) # deve convergir lentamente. dependendo do tamanho do problema, não dará uma boa resposta após 2000 iterações.
x_SD_symm = steepest_descent.solve_symmetric(A,b,tol=1e-03,maxit=2000)
x_CG = conjugate_gradient.solve(A,b,tol=1e-04,maxit=2000) # funciona para matrizes simétricas positiva-definidas, que é o caso aqui.

#print("x_LU =",x_LU)
#print("x_SD =",x_SD)

plt.figure(1)
plt.imshow(x_LU.reshape(rows-2,cols),cmap='coolwarm')
plt.colorbar()

plt.figure(2)
plt.imshow(x_SD.reshape(rows-2,cols),cmap='coolwarm')
plt.colorbar()

plt.figure(3)
plt.imshow(x_SD_symm.reshape(rows-2,cols),cmap='coolwarm')
plt.colorbar()

plt.figure(4)
plt.imshow(x_CG.reshape(rows-2,cols),cmap='coolwarm')
plt.colorbar()

plt.show()
