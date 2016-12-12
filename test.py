import numpy as np
from stellar_system import stellar_system


def three():
    '''sun,earth,moon
    rs,vs,re,ve,rm,vm,ms,me,mm
    '''
    G = 6.67428*10**-11
    r = [0,0,0,0,0,0,15E11,0,0,0,np.sqrt(G*2E30/15E11),0,15E11+4E8,0,0,0,np.sqrt(G*2E30/15E11)+np.sqrt(G*6E24/4E8),0,2E30,6E24,7E22]
    s = stellar_system(r)
    s.change(1E9,10000)
    s.trajectory()
    return s

def four():
    '''sun,earth,mars,moon
    '''
    G = 6.67428*10**-11
    r = [0,0,0,0,0,0,15E11,0,0,0,np.sqrt(G*2E30/15E11),0,23E11,0,0,0,np.sqrt(G*2E30/23E11),0,15E11+4E8,0,0,0,np.sqrt(G*2E30/15E11)+np.sqrt(G*6E24/4E8),0,2E30,6E24,6E23,7E22]
    s = stellar_system(r)
    s.change(1E9,10000)
    s.trajectory()

def five():
    '''sun,earth,mars,moon,mercury
    '''
    G = 6.67428*10**-11
    r = [0,0,0,0,0,0,15E11,0,0,0,np.sqrt(G*2E30/15E11),0,23E11,0,0,0,np.sqrt(G*2E30/23E11),0,15E11+4E8,0,0,0,np.sqrt(G*2E30/15E11)+np.sqrt(G*6E24/4E8),0,6E11,0,0,0,np.sqrt(G*2E30/6E11),0,2E30,6E24,6E23,7E22,3E23]
    s = stellar_system(r)
    s.change(1E9,10000)
    s.trajectory()

def four_plus_intruder():
    '''sun,earth,mars,moon,mercury
    '''
    G = 6.67428*10**-11
    r = [-15E11,0,0,0,0,0,-np.sqrt(15E11),np.sqrt(15E11),0,0,0,0,0,0,0,0,0,0,15E11,0,0,0,np.sqrt(G*2E30/15E11),0,23E11,0,0,-np.sqrt(G*2E30/23E11),0,0,15E11+4E8,0,0,0,np.sqrt(G*2E30/15E11)+np.sqrt(G*6E24/4E8),0,6E11,0,0,0,np.sqrt(G*2E30/6E11),0,2E30,2E30,2E30,6E24,6E23,7E22,3E23]
    s = stellar_system(r)
    s.change(1E2,10000)
    s.trajectory()

