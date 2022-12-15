import cv2
import os
import numpy as np
from xml.dom import minidom 


class Moving_part :
    
    def __init__(self, attr_list) :
        self.top = attr_list[0]
        self.bottom = attr_list[1]
        self.left = attr_list[2]
        self.right = attr_list[3]
        self.center = attr_list[4]
        
    def __repr__(self) :
        value = '- top = ' + str(self.top)
        value += '\n\t - bottom = ' + str(self.bottom)
        value += '\n\t - left = ' + str(self.left)
        value += '\n\t - right = ' + str(self.right)
        value += '\n\t - center = ' + str(self.center) + '\n'
        return value
    
    def get_points(self) :
        ''' Fonction that returns the not None attributes '''
        result = []
        if self.top is not None :
            result.append(self.top)
        if self.bottom is not None :
            result.append(self.bottom)
        if self.left is not None :
            result.append(self.left)
        if self.right is not None :
            result.append(self.right)
        if self.center is not None :
            result.append(self.center)
            
        return result
