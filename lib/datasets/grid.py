import numpy as np
from lib.datasets.base import visualize_xs

def x_generator(params):
    x1s = np.linspace(-1., 1., params['N1']) # large steps
    x2s = np.linspace(-1., 1., params['N2']) # small steps
    
    grid = (params['N1'] != params['N2']) # partly fixes dups

    for i1 in range(params['N1']):
        for i2 in range(params['N2']):
            yield np.array([x1s[i1], x2s[i2]])
            if grid:
                yield np.array([x2s[i2], x1s[i1]])

if __name__ == "__main__":        
    visualize_xs(x_generator({'N1': 5, 'N2': 50}))
