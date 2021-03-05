#!/usr/bin/env - python

import os
import numpy as np
import pandas as pd

import lib.datasets.base as ds
import lib.datasets.rand as ds_rand
import lib.datasets.grid as ds_grid

from lib.envs.myenv import MyEnv
from lib.datasets.base import xy_generator, store

def generate(name_template, x_gen_params, x_gen_func):
    name = name_template.format(**x_gen_params)
    if name in store:
        print("'%s' exists, skipping" % name)
        return

    print("Generating '%s' ..." % name)

    xy_gen = xy_generator(x_gen_func(x_gen_params))
    store[name] = pd.DataFrame(xy_gen)

    print("Generation of '%s' is done" % (name))

generate('rand_1k', {'N': 1000}, ds_rand.x_generator)
generate('grid_{N1}_{N2}', {'N1': 10, 'N2': 1000}, ds_grid.x_generator)
#generate('rand-10k', {'N': 10000}, ds_rand.x_generator)
#generate('linspaced_10k', {'N1': 100, 'N2': 100}, ds_grid.x_generator)
#generate('linspaced_250k', {'N1': 500, 'N2': 500}, ds_grid.x_generator)
