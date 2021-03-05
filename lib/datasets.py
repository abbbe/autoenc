import os
import numpy as np

from lib.config import DATASETS_DIR

def load_dataset(env_name, dataset_name):
    fname = os.path.join(DATASETS_DIR, ("%s_%s.npz" % (env_name, dataset_name)))
    print("Loading %s ..." % fname)
    dataset = np.load(fname)
    assert(dataset['angles'].shape[0] == dataset['images'].shape[0])
    print("Loaded %d datapoints from %s" % (dataset['angles'].shape[0], fname))
    return dataset
    
def load_datasets(env_name):
    train = load_dataset(env_name, 'linspaced_250k')
    val = load_dataset(env_name, 'rand_1k')
    grid = load_dataset(env_name, 'grid_100_1000')
    
    return {'train': train, 'val': val, 'grid': grid}
