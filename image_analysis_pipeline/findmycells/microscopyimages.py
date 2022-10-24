from abc import ABC, abstractmethod
import os
from pathlib import Path
import numpy as np
import pandas as pd
import czifile
from skimage.io import imread




class MicroscopyImageReader(ABC):

    @abstractmethod
    def read(self, filepath: Path) -> np.ndarray:
        """ 
        Return the image as np.ndarray with structure: [planes, rows, columns, rgbs or single channel] 
        For instance, the shape of the array of a RGB Zstack with 10 image planes and 1024x1024 pixels would look like:
        [10, 1024, 1024, 3]
        To improve re-usability of the same functions for all different kinds of input images, this structure will be used if even there is just a single plane. 
        For instance, the shape of the array of a grayscale 2D image with 1024 x 1024 pixels should look like this:
        [1, 1024, 1024, 1]
        """
        pass
    
class CZIReader(MicroscopyImageReader):
    
    def read(self, filepath: Path) -> np.ndarray:
        return czifile.imread(filepath.as_posix())[0, 0, 0]
    

class FromExcel(MicroscopyImageReader):
    
    # should actually again check which loaded is applicable! Could be any!
    def read(self, filepath: Path) -> np.ndarray:
        df_single_plane_filepaths = pd.read_excel(filepath)
        single_plane_images = []
        for row_index in range(df_single_plane_filepaths.shape[0]):
            single_plane_image_filepath = df_single_plane_filepaths['plane_filepath'].iloc[row_index]
            single_plane_images.append(imread(single_plane_image_filepath))
        return np.stack(single_plane_images)

    
class RegularImageFiletypeReader(MicroscopyImageReader):
    
    def read(self, filepath: Path) -> np.ndarray:
        single_plane_image = imread(filepath)
        return np.expand_dims(single_plane_image, axis=[0, -1])
    

class MicroscopyImageLoader:
    
    def __init__(self, filepath: Path, filetype: str):
        self.filepath = filepath
        self.reader = self.determine_reader(filetype = filetype)
    
    def determine_reader(self, filetype: str) -> MicroscopyImageReader:
        if filetype == '.czi':
            reader = CZIReader()
        elif filetype in ['.png', '.PNG']: #add more that are applicable!
            reader = RegularImageFiletypeReader()
        elif filetype == '.xlsx':
            reader = FromExcel()
        else:
            message_part1 = 'The microscopy image format you are trying to load is not implemented yet.'
            message_part2 = 'Please consider raising an issue in our GitHub repository!'
            full_message = message_part1 + message_part2
            raise ValueError(full_message)
        return reader
    
    def as_array(self):
        return self.reader.read(filepath = self.filepath)


# old code starts here:
"""
class RGBZStack(ABC): #dependency inversion!
    
    def __init__(self, filepath: Path):
        self.filepath = filepath #adapt to Path .as_osix() ?
        self.isrgb = True
        self.is3d = True
        self.as_array = self.read()
        self.total_planes = self.as_array.shape[0]
    
    @abstractmethod
    def read(self) -> np.ndarray:
        pass
        
        
        
class CZIZStack(RGBZStack):
    
    def read(self):
        return czifile.imread(self.filepath)[0, 0, 0]
"""