import numpy as np

def EDO(t,n,state):
    '''(list,list,int,str,str) -> list
   Recebe um instante instante final (começa em 0)
    um o vetor com as condiçoes iniciais,
    um inteiro n que define o tamanho do passo,
    um string indicando o metodo a ser utilizado,
    e um string indicando o modelo a ser resolvido.
    Retorna a solução numerica da EDO
    '''
    phi = runge_kutta
    G = 6.67428*10**-11    

#Discretiza o intervalo de estudo
    dt = t/n

#Obtem os valores das massas, através do número de corpos 
    N = int(len(state)/7)
    
    m = state[-N:]


#Cria matriz que guarda os valores numericos calculados para cada x(tk+dt)
    x0 = state[:-N]
    i = len(x0)
    j = n + 1    
    x = np.zeros((i,j))
    x0 = np.transpose(np.array([x0]))
    x[:,0:1] = x0

#Laço que calcula e atribui os valores de x(tk+dt)
    tk = 0
    k = 0
    for k in range(n):
        x[:,k+1:k+2] = x[:,k:k+1] + dt*phi(x[:,k:k+1],m,tk,dt)
        tk = tk + dt
    return x


#Metodo Runge-Kutta classico#

def runge_kutta(x,m,tk,dt):
    '''(array,float,float,mod)-> array
	Propriedades:
	e(t) = O(dt^4)
	lambda*dt \in (-2.71,0)    
	4 estágios    
	1 passo
'''
#Define o modelo que sera utilizado
    f = stellar_system
    
    K1 = f(x,m,tk,dt)
    K2 = f(x+(dt/2)*K1,m,tk+(dt/2),dt)
    K3 = f(x+(dt/2)*K2,m,tk+(dt/2),dt)
    K4 = f(x+dt*K3,m,tk+dt,dt)
    phi = (1/6)*(K1 + 2*K2 + 2*K3 + K4)
    return phi


#Stellar system model#

def stellar_system(x,m,t,dt):
    '''(array,float,float)->Array
    F do problema de Cauchy \dot{y}=f(y,t), do problema de 3-corpos, criada a partir das leis de Newton.
    Dicionario semantico:
    rs := posicao do Sol
    vs := velocidade do Sol
    rt := posicao da Terra
    vt := velocidade da Terra
    rl := posicao da Lua
    vl := velocidade da Lua
    '''

#Define as constantes do problema
    G = 6.67428*10**-11


#Descobre o estado do sistema
    N = len(m) #Números de corpos
    r = np.zeros((N,3)) #Posições
    v = np.zeros((N,3)) #Velocidades
    for k in range(N):
        r[k] = np.array([x[6*k][0],x[6*k+1][0],x[6*k+2][0]])
        v[k] = np.array([x[6*k+3][0],x[6*k+4][0],x[6*k+5][0]])


        
#Lei da gravitação de Newton. Calcula o vetor f = (f1,f2,f3,f4,f5,f6) tal que dx/dt = f(x,tk)

    f = np.zeros((2*N,3))
    for k in range(N):
        f[2*k] = v[k]
        for l in range(0,k):
            f[2*k+1] += G*m[l]*(r[l]-r[k])/np.linalg.norm(r[l]-r[k])**3
        for l in range(k+1,N):
            f[2*k+1] += G*m[l]*(r[l]-r[k])/np.linalg.norm(r[l]-r[k])**3


#Converte o resultado em um vetor coluna
    F = []
    for k in range(2*N):
        F  += list(f[k])
    f = np.transpose(np.array([F]))
    return f
