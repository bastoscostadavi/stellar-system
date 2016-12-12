import numpy as np

def EDO(t,n,x0):
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

#Cria matriz que guarda os valores numericos calculados para cada x(tk+dt)
    i = len(x0)
    j = n + 1    
    x = np.zeros((i,j))
    x0 = np.transpose(np.array([x0]))
    x[:,0:1] = x0

#Laço que calcula e atribui os valores de x(tk+dt)
    tk = 0
    for k in range(n):
        x[:,k+1:k+2] = x[:,k:k+1] + dt*phi(x[:,k:k+1],tk,dt)
        tk = tk + dt
    return x        


#Metodo Runge-Kutta classico#

def runge_kutta(x,tk,dt):
    '''(array,float,float,mod)-> array
	Propriedades:
	e(t) = O(dt^4)
	lambda*dt \in (-2.71,0)    
	4 estágios    
	1 passo
'''
#Define o modelo que sera utilizado
    f = stellar_system
    
    K1 = f(x,tk,dt)
    K2 = f(x+(dt/2)*K1,tk+(dt/2),dt)
    K3 = f(x+(dt/2)*K2,tk+(dt/2),dt)
    K4 = f(x+dt*K3,tk+dt,dt)
    phi = (1/6)*(K1 + 2*K2 + 2*K3 + K4)
    return phi


#Stellar system model#

def stellar_system(x,t,dt):
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
    Ms = 2E30
    Mt = 6E24
    Ml = 7E22

#Descobre o estado do sistema
    rs = np.array([x[0][0],x[1][0],x[2][0]])
    vs = np.array([x[3][0],x[4][0],x[5][0]])
    rt = np.array([x[6][0],x[7][0],x[8][0]])
    vt = np.array([x[9][0],x[10][0],x[11][0]])
    rl = np.array([x[12][0],x[13][0],x[14][0]])
    vl = np.array([x[15][0],x[16][0],x[17][0]])

#Calcula o vetor f = (f1,f2,f3,f4,f5,f6) tal que dx/dt = f(x,tk)
    f1 = vs
    f2 = G*(Mt*(rt-rs)/np.linalg.norm(rt-rs)**3+Ml*(rl-rs)/np.linalg.norm(rl-rs)**3)
    f3 = vt
    f4 = G*(Ms*(rs-rt)/np.linalg.norm(rs-rt)**3+Ml*(rl-rt)/np.linalg.norm(rl-rt)**3)
    f5 = vl
    f6 = G*(Ms*(rs-rl)/np.linalg.norm(rs-rl)**3+Mt*(rt-rl)/np.linalg.norm(rt-rl)**3)

    f = np.array([list(f1)+list(f2)+list(f3)+list(f4)+list(f5)+list(f6)]).transpose()
    return f
