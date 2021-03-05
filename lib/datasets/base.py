import numpy as np
import pandas as pd

# spawn gym environment
#from env import ArmRobotEnv
#env = ArmRobotEnv()
DATASETSDIR = "output/datasets"

from lib.envs.myenv import MyEnv
env = MyEnv()

def visualize_xs(x_gen):
    import matplotlib.pyplot as plt

    xs = list(x_gen)
    xs = np.array(xs)

    plt.suptitle("XS")
    plt.scatter(xs[:,0], xs[:,1], 1)
    plt.show()

def xy_generator(x_gen):
    for x in x_gen:
        env.step(x)
        env.render()
        y = env.get_image()

        yield x, y

store = pd.HDFStore('store.h5')
