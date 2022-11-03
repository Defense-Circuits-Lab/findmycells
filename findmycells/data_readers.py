# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_microscopy_images-Copy1.ipynb.

# %% auto 0
__all__ = ['MicroscopyImageReader', 'CZIReader', 'RegularImageFiletypeReader', 'FromExcel', 'MicroscopyImageLoader']

# %% ../nbs/02_microscopy_images-Copy1.ipynb 4
from abc import ABC, abstractmethod
import os
from pathlib import Path
import numpy as np
import pandas as pd
import czifile
from skimage.io import imread

# %% ../nbs/02_microscopy_images-Copy1.ipynb 5
class MicroscopyImageReader(ABC):
    
    """
    Abstract base class that serves as interface to load different types of microscopy image filetypes.
    """

    @abstractmethod
    def read(self, 
             filepath: Path # filepath to the microscopy image file
            ) -> np.ndarray: # numpy array with the structure: [imaging-planes, rows, columns, imaging-channel], see docstring for examples
        """ 
        Abstract method that needs to be defined by the respective subclass
        Returns the microscopy image as np.ndarray with structure: [imaging-planes, rows, columns, imaging-channel] 
        For instance, the shape of the array of a RGB Zstack with 10 image planes and 1024x1024 pixels would look like:
        [10, 1024, 1024, 3]
        To improve re-usability of the same functions for all different kinds of input images, this structure will be used even if there is just a single plane. 
        For instance, the shape of the array of a grayscale 2D image with 1024 x 1024 pixels should look like this:
        [1, 1024, 1024, 1]
        """
        pass

# %% ../nbs/02_microscopy_images-Copy1.ipynb 8
class CZIReader(MicroscopyImageReader):
    
    """
    This reader enables loading of images acquired with the ZEN imaging software by Zeiss, using the czifile package.
    """
    def read(self,
             filepath: Path # filepath to the microscopy image file
            ) -> np.ndarray: # numpy array with the structure: [imaging-planes, rows, columns, imaging-channel]
        return czifile.imread(filepath.as_posix())[0, 0, 0]

# %% ../nbs/02_microscopy_images-Copy1.ipynb 10
class RegularImageFiletypeReader(MicroscopyImageReader):
    
    """
    This reader enables loading of all regular image filetypes, that scikit-image can read, using the scikit-image.io.imread function.
    """
    def read(self,
             filepath: Path # filepath to the microscopy image file
            ) -> np.ndarray: # numpy array with the structure: [imaging-planes, rows, columns, imaging-channel]
        single_plane_image = imread(filepath)
        return np.expand_dims(single_plane_image, axis=[0, -1])

# %% ../nbs/02_microscopy_images-Copy1.ipynb 12
class FromExcel(MicroscopyImageReader):
    
    """
    This reader is actually only a wrapper to the other MicroscopyImageReader subclasses. It can be used if you stored the filepaths
    to your individual plane images in an excel sheet, for instance if you were using our "prepare my data for findmycells" functions.
    Please be aware that the corresponding datatype has to be loadable with any of the corresponding MicroscopyImageReaders!
    """
    # should actually again check which loaded is applicable! Could be any!
    def read(self,
             filepath: Path # filepath to the excel sheet that contains the filepaths to the corresponding image files
            ) -> np.ndarray: # numpy array with the structure: [imaging-planes, rows, columns, imaging-channel]
        df_single_plane_filepaths = pd.read_excel(filepath)
        single_plane_images = []
        for row_index in range(df_single_plane_filepaths.shape[0]):
            single_plane_image_filepath = df_single_plane_filepaths['plane_filepath'].iloc[row_index]
            single_plane_images.append(imread(single_plane_image_filepath))
        return np.stack(single_plane_images)

# %% ../nbs/02_microscopy_images-Copy1.ipynb 15
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