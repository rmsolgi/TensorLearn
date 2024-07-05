import numpy as np


def divisors_finder(n):
    
    div_check=np.arange(1,int(n/2)+1)
    rem=np.remainder(n,div_check)
    divisors=np.where(rem==0)[0]+1

    return divisors