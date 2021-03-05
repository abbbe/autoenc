import numpy as np
from lib.datasets.base import visualize_xs

def x_generator(params):
    for i in range(params['N']):
        yield np.random.uniform(low=-1.0, high=1.0, size=(2))

if __name__ == "__main__":        
    visualize_xs(x_generator(250))