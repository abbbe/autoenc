import numpy as np

import os
os.environ['SDL_VIDEODRIVER']='dummy'
import pygame
pygame.display.set_mode((600,600))

import gym 

import skimage.transform

class ArmRobotEnv:
    def __init__(self, D=64):
        self.D = D
        self.env = gym.make('gym_robot_arm:robot-arm-v1')
        self.env.draw_target = lambda: None # sabotage target (red circle) drawing
    
    def reset(self):
        return self.env.reset()
    
    def step(self, a):
        return self.env.step(a)

    def render(self):
        return self.env.render()

    def close(self):
        _res = self.env.close()
        self.env = None
        return _res

    def get_image(self):
        """
        This function returns pygame display images, preprocessed as (D,D,C=1) matrix
        If pyGameDisplay is set, also returns unprocessed
        """
        return self.process_raw_image(self.get_raw_image())

    def get_raw_image(self):
        img_surface = pygame.display.get_surface()
        img_array = pygame.surfarray.array3d(img_surface)
        return img_array

    def process_raw_image(self, img_array):
        # scale to 0..1 (will rescale later again, but need it here for thresholding to work)
        img_array = np.array(img_array, dtype=np.float32) / 255.

        # crop
        img_array = img_array[70:-70,70:-70]

        # condition the blue plane, highlight the endpoint circle and the links
        blue_plane = img_array[...,2]
        thresh0 = 0.3
        blue_plane[blue_plane > thresh0] = 1.
        blue_plane[blue_plane <= thresh0] = 0.
        blue_plane = 1. - blue_plane

        # apply threhold to averaged image, to highlight the joint circles
        grey_plane = np.average(img_array, axis=2)
        thresh1 = 0.1
        grey_plane[grey_plane < thresh1] = 0.
        grey_plane[grey_plane >= thresh1] = 1.

        # combine the two
        img_array = grey_plane + blue_plane

        # resize
        img_array = skimage.transform.resize(img_array, (self.D, self.D))

        # apply second thresold
        #thresh2 = -1.8
        #img_array[img_array < thresh2] = 0.
        #img_array[img_array >= thresh2] = 1.

        # normalize to [0, 1]
        img_array = (img_array - np.min(img_array))/np.ptp(img_array)

        # add a dummy channel axis
        img_array = np.expand_dims(img_array, axis=-1)
        return img_array

    #def get_action_image(self, action):
    #    self.env.step(action)
    #    self.env.render()
    #    return self.get_image()

#    def get_demo_images(self):
#        demo_as = [[-1, -1], [0, 0], [1, 1]]
#        return list(map(self.get_action_image, demo_as))
