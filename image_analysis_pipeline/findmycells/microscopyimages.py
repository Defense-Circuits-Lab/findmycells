from abc import ABC, abstractmethod
import os
import numpy as np
import czifile



class RGBZStack(ABC): #dependency inversion!
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.isrgb = True
        self.is3d = True
        self.read()
        self.total_planes = self.as_array.shape[0]
    
    @abstractmethod
    def read(self):
        """ create self.as_array with structure: [plane, rows, columns, rgb] """
        pass  
        
        
class CZIZStack(RGBZStack):
    
    def read(self):
        self.as_array = czifile.imread(self.filepath)[0, 0, 0]