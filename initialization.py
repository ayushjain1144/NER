import numpy as np
import math

def random_initialization(l_prev, l):
    return np.random.randn(l_prev, l)

# uniform initialization between -e, e
# e = (6/ fan_in + fanout)^0.5
# fanIn = l_prev, fanOut = l 
def range_initializtion(l_prev, l):
    
    e = math.sqrt(6 / (l_prev + l))
    return np.random.uniform(-e, e, (l_prev, l))

# def main():
#     a  = random_initialization(3, 4)
#     b = range_initializtion(3, 4)

#     print(a)
#     print(b)

# if __name__ == "__main__":
#     main()