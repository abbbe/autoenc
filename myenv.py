import pygame
import math
import numpy as np
import skimage.transform

SCREEN_SIZE = (128,128)
JOINTS_LEN = [30, 30]
SCREEN_COLOR = (0, 0, 0)
LINK_COLOR = (0, 0, 0)
JOINT_COLOR = (255, 0, 0)
TIP_COLOR = (255, 0, 0)
TIP_RADIUS = 3
JOINT_RADIUS = 3
INIT_THETA = np.array([0, 0])

class MyEnv:
    def __init__(self, D=64):
        self.D = D
        self.set_window_size(SCREEN_SIZE)
        self.screen = pygame.display.set_mode(self.window_size)
        self.set_link_properties(JOINTS_LEN)
        
        self.reset()
    
    def reset(self):
        self.step(INIT_THETA)

    def step(self, actions):
        self.theta = actions
        return None, None, None, None
    
    def render(self):
        self.draw(self.theta)

    def set_window_size(self, window_size):
        self.window_size = window_size
        self.centre_window = [window_size[0]//2, window_size[1]//2]

        
    def get_image(self):
        return self.process_raw_image(self.get_raw_image())

    def get_raw_image(self):
        img_surface = pygame.display.get_surface()
        img_array = pygame.surfarray.array3d(img_surface)
        return img_array

    def process_raw_image(self, img_array):
        # condition the blue plane, highlight the endpoint circle and the links
        y = img_array[...,0]

        # resize
        y = skimage.transform.resize(y, (self.D, self.D))

        # normalize to [0, 1]
        y = (y - np.min(y))/np.ptp(y)

        # add a dummy channel axis
        y = np.expand_dims(y, axis=-1)

        return y
        
    def draw(self, theta):
        self.screen.fill(SCREEN_COLOR)

        theta = self.inverse_theta(theta)
        P = self.forward_kinematics(theta)
        origin = np.eye(4)
        origin_to_base = self.translate(self.centre_window[0],self.centre_window[1],0)
        base = origin.dot(origin_to_base)
        F_prev = base.copy()
        for i in range(1, len(P)):
            F_next = base.dot(P[i])
            pygame.draw.line(self.screen, LINK_COLOR, (int(F_prev[0,3]), int(F_prev[1,3])), (int(F_next[0,3]), int(F_next[1,3])), 5)
            pygame.draw.circle(self.screen, JOINT_COLOR, (int(F_prev[0,3]), int(F_prev[1,3])), JOINT_RADIUS)
            F_prev = F_next.copy()
        pygame.draw.circle(self.screen, TIP_COLOR, (int(F_next[0,3]), int(F_next[1,3])), TIP_RADIUS)

    def rotate_z(self, theta):
        rz = np.array([[np.cos(theta), - np.sin(theta), 0, 0],
                       [np.sin(theta), np.cos(theta), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        return rz

    def translate(self, dx, dy, dz):
        t = np.array([[1, 0, 0, dx],
                      [0, 1, 0, dy],
                      [0, 0, 1, dz],
                      [0, 0, 0, 1]])
        return t

    def forward_kinematics(self, theta):
        P = []
        P.append(np.eye(4))
        for i in range(0, self.n_links):
            R = self.rotate_z(theta[i])
            T = self.translate(self.links[i], 0, 0)
            P.append(P[-1].dot(R).dot(T))
        return P

    def inverse_theta(self, theta):
        new_theta = theta.copy()
        for i in range(theta.shape[0]):
            new_theta[i] = -1*theta[i]
        return new_theta

    def set_link_properties(self, links):
        self.links = links
        self.n_links = len(self.links)
        self.min_theta = math.radians(0)
        self.max_theta = math.radians(90)
        self.max_length = sum(self.links)

#    def generate_random_angle(self):
#        theta = np.zeros(self.n_links)
#        for i in range(len(JOINTS_LEN)):
#            theta[i] = np.random.uniform(self.min_theta, self.max_theta)
#        return theta
        