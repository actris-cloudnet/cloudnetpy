import numpy as np

def is_bit(n, k):
    """ element wise test of bit k (1,2,3...) on integer n """
    mask = 1 << k-1
    return (n & mask > 0)

                                                                                                                                           
def set_bit(n, k):
    """ set bit k (1,2,3..) on integer n """
    mask = 1 << k-1
    #n = np.bitwise_or(n, mask)
    n |= mask
    return n
