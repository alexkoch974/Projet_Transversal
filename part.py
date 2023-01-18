import cv2
import os
import numpy as np
from xml.dom import minidom 


class Moving_part :
    
    def __init__(self, attr_list) :
        self.bottom = attr_list[0]
        self.left = attr_list[1]
        self.top = attr_list[2]
        self.center = attr_list[3]
        self.right = attr_list[4]
        
        
    def __repr__(self) :
        value = '- bottom = ' + str(self.bottom)
        value += '\n\t - left = ' + str(self.left)
        value += '\n\t - top = ' + str(self.top)
        value += '\n\t - center = ' + str(self.center) 
        value += '\n\t - right = ' + str(self.right) + '\n'
        return value
    
    def get_points(self) :
        ''' Fonction that returns the not None attributes '''
        result = []
        if self.bottom is not None :
            result.append(self.bottom)
        if self.left is not None :
            result.append(self.left)
        if self.top is not None :
            result.append(self.top)
        if self.center is not None :
            result.append(self.center)
        if self.right is not None :
            result.append(self.right)
        
            
        return result
