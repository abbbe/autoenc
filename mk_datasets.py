import os
import numpy as np

from lib.config import DATASETS_DIR
from lib.envs.myenv import TwoBallsEnv

import numpy as np

def generate_rand_angles(params):
    N = params['N']
    return np.random.uniform(low=-1.0, high=1.0, size=(N, 2)) # FIXME number of joints is hardcoded

def generate_grid_angles(params):
    N1 = params['N1']
    a1s = np.linspace(-1., 1., N1) # FIXME number of joints and low/high angles are hardcoded
    if 'N2' in params:
        N2 = params['N2']
        a2s = np.linspace(-1., 1., N2)
        grid = True
    else:
        N2 = N1
        a2s = a1s
        grid = False

    N = N1 * N2
    angles = np.zeros((N, 2))

    i = 0
    for i1 in range(N1):
        for i2 in range(N2):
            angles[i] = [a1s[i1], a2s[i2]]
            i += 1
    return angles

def generate_images(env, angles):
    N = angles.shape[0]
    images = np.zeros((N, env.D, env.D, 1)) # FIXME number of channels is hardcoded

    for i in range(N):
        env.step(angles[i])
        env.render()
        image = env.get_image()

        images[i] = image

    return images

def save_dataset(name, angles, images):
    fname = os.path.join(DATASETS_DIR, ("%s.npz" % name))
    print("%d datapoints generated, saving in %s" % (angles.shape[0], fname))
    np.savez(fname, angles=angles, images=images)

def generate_and_save(gen_method, params, env, name_template):
    """
    Generates angles using given 'gen_method' with given 'params'
    Uses given 'env' to produce images corresponding to these angles
    Saves angles and images in npz file name formatted with the same 'params'
    """
    name = name_template.format(**params)
    #if df_name in store:
    #    print("'%s' exists, skipping" % df_name)
    #    #return

    print("Generating '%s' ..." % name)
    angles = gen_method(params)
    images = generate_images(env, angles)

    save_dataset(name, angles, images)

# spawn gym environment
#from env import ArmRobotEnv
#env = ArmRobotEnv()

env = TwoBallsEnv()
generate_and_save(generate_rand_angles, {'N': 1000},                env, 'twoballs_rand_1k')
generate_and_save(generate_grid_angles, {'N1': 100, 'N2': 1000},    env, 'twoballs_grid_{N1}_{N2}')
generate_and_save(generate_grid_angles, {'N1': 500},                env, 'twoballs_linspaced_250k')
#generate_and_save(generate_grid_angles, {'N1': 1000},               env, 'twoballs_linspaced_1m')
