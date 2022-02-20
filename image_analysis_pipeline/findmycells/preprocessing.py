# Probably has to become an entire module with several submodules, e.g. for cropping, channel splitting, max projections, ...

from abc import ABC, abstractmethod
import os
import numpy as np
from skimage.io import imsave
from shapely.geometry import Polygon
from typing import List, Tuple, Dict
from skimage.io import imsave

from .database import Database
from .microscopyimages import MicroscopyImageLoader
from .rois import ROILoader
from .utils import convert_12_to_8_bit_rgb_image


"""
Next steps:
 - load ROIs (or create one for with shape of the image)
 - make adaptations to database:
   - use Path objects to enable continous integration
   - create sorted list of all preprocessing steps (i.e. preprocessing strategies)
 - adapt main
 - function that saves preprocessed images

 - Minimal preprocessing steps are:
    - save the "unprocessed" microscopy images to the preprocessed_dir
    - load the unprocessed ROIs into the database (create ROI with shape of image if whole image is to be analyzed)
"""

class PreprocessingObject:
    
    def __init__(self, database: Database, file_id: str) -> None:
        self.database = database
        self.file_id = file_id
        self.file_info = self.database.get_file_infos(identifier = self.file_id)
        self.preprocessed_image = self.load_microscopy_image()
        self.total_planes = self.preprocessed_image.shape[0]
        if self.preprocessed_image.shape[3] == 3:
            self.is_rgb = True
        else:
            self.is_rgb = False
        self.preprocessed_rois = self.load_rois()

        
    def load_microscopy_image(self) -> np.ndarray:
        microscopy_image_loader = MicroscopyImageLoader(filepath = self.file_info['microscopy_filepath'],
                                                        filetype = self.file_info['microscopy_filetype'])
        return microscopy_image_loader.as_array()
    
    
    def load_rois(self) -> Dict:
        if self.file_info['rois_present'] ==  False:
            message_line0 = 'As of now, it is not supported to not provide a ROI file for each image.\n'
            message_line1 = 'If you would like to quantify the image features in the entire image, please provide a ROI that covers the entire image.\n'
            message_line2 = 'Warning: However, please be aware that this feature was not fully tested yet and will probably cause problems,\n'
            message_line3 = 'specifically if any cropping is used as preprocessing method.'
            error_message = message_line0 + message_line1 + message_line2 + message_line3
            raise NotImplementedError(error_message)
        elif self.file_info['rois_present']:
            roi_loader = ROILoader(filepath = self.file_info['rois_filepath'],
                                   filetype = self.file_info['rois_filetype'])
        return roi_loader.as_dict()
    
    
    def run_all_preprocessing_steps(self) -> None:
        for preprocessing_strategy in self.database.preprocessing_strategies:
            self = preprocessing_strategy().run(preprocessing_object = self, step = self.database.preprocessing_strategies.index(preprocessing_strategy))
    
    
    def save_preprocessed_images_on_disk(self) -> None:
        for plane_index in range(self.total_planes):
            image = self.preprocessed_image[plane_index].astype('uint8')
            filepath_out = self.database.preprocessed_images_dir.joinpath(f'{self.file_id}-{str(plane_index).zfill(3)}.png')
            imsave(filepath_out, image)

    
    def save_preprocessed_rois_in_database(self) -> None:
        self.database.import_rois_dict(file_id = self.file_id, rois_dict = self.preprocessed_rois)
        
    
    def update_database(self) -> None:
        updates = dict()
        # RGB and total_planes should be updated at this stage again to make sure that they represent the actual preprocessed image!
        updates['RGB'] = self.is_rgb
        updates['total_planes'] = self.total_planes
        updates['preprocessing_completed'] = True
        self.database.update_file_infos(file_id = self.file_id, updates = updates)

            

class PreprocessingStrategy(ABC):
    
    @abstractmethod
    def run(self, preprocessing_object: PreprocessingObject, step: int) -> PreprocessingObject:
        # do preprocessing
        preprocessing_object.database = self.update_database(database = preprocessing_object.database, file_id = preprocessing_object.file_id, step = step)
        return preprocessing_object
    
    @abstractmethod
    def update_database(self, database: Database, file_id: str, step: int) -> Database:
        updates = dict()
        updates[f'preprocessing_step_{str(step).zfill(2)}'] = 'PreprocessingStrategyName'
        # Add additional information if neccessary
        database.update_file_infos(file_id = file_id, updates = updates)
        return database



class CropStitchingArtefactsRGB(PreprocessingStrategy):
    
    def run(self, preprocessing_object: PreprocessingObject, step: int) -> PreprocessingObject:
        self.cropping_indices = self.determine_cropping_indices_for_entire_zstack(preprocessing_object = preprocessing_object)
        preprocessing_object.preprocessed_image = self.crop_rgb_zstack(zstack = preprocessing_object.preprocessed_image)
        preprocessing_object.preprocessed_rois = self.adjust_rois(rois_dict = preprocessing_object.preprocessed_rois)
        preprocessing_object.database = self.update_database(database = preprocessing_object.database, file_id = preprocessing_object.file_id, step = step)
        return preprocessing_object
        
    
    def get_cropping_indices(self, a, min_black_px_stretch: int=100) -> Tuple[int, int]:
        unique, counts = np.unique(a, return_counts=True)
        indices_with_black_pixels = unique[np.where(counts >= min_black_px_stretch)]
        if indices_with_black_pixels.shape[0] > 0: 
            if np.where(np.diff(indices_with_black_pixels) > 1)[0].shape[0] > 0:
                lower_cropping_index = indices_with_black_pixels[np.where(np.diff(indices_with_black_pixels) > 1)[0]][0] + 1
                upper_cropping_index = indices_with_black_pixels[np.where(np.diff(indices_with_black_pixels) > 1)[0] + 1][0]
            else:
                if indices_with_black_pixels[0] == 0:
                    lower_cropping_index = indices_with_black_pixels[-1]
                    upper_cropping_index = a.shape[0] - 1
                else:
                    lower_cropping_index = 0
                    upper_cropping_index = indices_with_black_pixels[0]
        else:
            lower_cropping_index = 0
            upper_cropping_index = a.shape[0] - 1
        return lower_cropping_index, upper_cropping_index 
                                                  
    
    def determine_cropping_indices_for_entire_zstack(self, preprocessing_object: PreprocessingObject) -> Dict:
        for plane_index in range(preprocessing_object.total_planes):
            rgb_image_plane = preprocessing_object.preprocessed_image[plane_index]
            rows_with_black_px, columns_with_black_px = np.where(np.all(rgb_image_plane == 0, axis = -1))
            lower_row_idx, upper_row_idx = self.get_cropping_indices(rows_with_black_px)
            lower_col_idx, upper_col_idx = self.get_cropping_indices(columns_with_black_px)  
            if plane_index == 0:
                min_lower_row_cropping_idx, max_upper_row_cropping_idx = lower_row_idx, upper_row_idx
                min_lower_col_cropping_idx, max_upper_col_cropping_idx = lower_col_idx, upper_col_idx
            else:
                if lower_row_idx > min_lower_row_cropping_idx:
                    min_lower_row_cropping_idx = lower_row_idx
                if upper_row_idx < max_upper_row_cropping_idx:
                    max_upper_row_cropping_idx = upper_row_idx
                if lower_col_idx > min_lower_col_cropping_idx:
                    min_lower_col_cropping_idx = lower_col_idx
                if upper_col_idx < max_upper_col_cropping_idx:
                    max_upper_col_cropping_idx = upper_col_idx  
        cropping_indices = {'lower_row_cropping_idx': min_lower_row_cropping_idx,
                            'upper_row_cropping_idx': max_upper_row_cropping_idx,
                            'lower_col_cropping_idx': min_lower_col_cropping_idx,
                            'upper_col_cropping_idx': max_upper_col_cropping_idx}
        return cropping_indices
    
    
    def crop_rgb_zstack(self, zstack: np.ndarray) -> np.ndarray:
        min_row_idx = self.cropping_indices['lower_row_cropping_idx']
        max_row_idx = self.cropping_indices['upper_row_cropping_idx']
        min_col_idx = self.cropping_indices['lower_col_cropping_idx']
        max_col_idx = self.cropping_indices['upper_col_cropping_idx']
        return zstack[:, min_row_idx:max_row_idx, min_col_idx:max_col_idx, :]
                                                           
    
    def adjust_rois(self, rois_dict: Dict) -> Dict:
        lower_row_idx = self.cropping_indices['lower_row_cropping_idx']
        lower_col_idx = self.cropping_indices['lower_col_cropping_idx']
        for plane_identifier in rois_dict.keys():
            for roi_id in rois_dict[plane_identifier].keys():
                adjusted_row_coords = [coordinates[0] - lower_row_idx for coordinates in rois_dict[plane_identifier][roi_id].boundary.coords[:]]
                adjusted_col_coords = [coordinates[1] - lower_col_idx for coordinates in rois_dict[plane_identifier][roi_id].boundary.coords[:]]
                rois_dict[plane_identifier][roi_id] = Polygon(np.asarray(list(zip(adjusted_row_coords, adjusted_col_coords))))
        return rois_dict
    
    def update_database(self, database: Database, file_id: str, step: int) -> Database:
        updates = dict()
        updates[f'preprocessing_step_{str(step).zfill(2)}'] = 'CropStitchingArtefactsRGB'
        updates['cropping_row_indices'] = (self.cropping_indices['lower_row_cropping_idx'], self.cropping_indices['upper_row_cropping_idx'])
        updates['cropping_column_indices'] = (self.cropping_indices['lower_col_cropping_idx'], self.cropping_indices['upper_col_cropping_idx'])       
        database.update_file_infos(file_id = file_id, updates = updates)
        return database
        
        
                                                           
                                                           
class ConvertFrom12To8BitRGB(PreprocessingStrategy):
    
    def run(self, preprocessing_object: PreprocessingObject, step: int) -> PreprocessingObject:
        for plane_index in range(preprocessing_object.total_planes):
            preprocessing_object.preprocessed_image[plane_index] = self.convert_rgb_image(rgb_image = preprocessing_object.preprocessed_image[plane_index])
        preprocessing_object.database = self.update_database(database = preprocessing_object.database, file_id = preprocessing_object.file_id, step = step)
        return preprocessing_object
    
    def convert_rgb_image(self, rgb_image: np.ndarray) -> np.ndarray:
        converted_image = (rgb_image / 4095 * 255).round(0).astype('uint8')
        return converted_image

    def update_database(self, database: Database, file_id: str, step: int) -> Database:
        updates = dict()
        updates[f'preprocessing_step_{str(step).zfill(2)}'] = 'ConvertFrom12To8BitRGB'    
        database.update_file_infos(file_id = file_id, updates = updates)
        return database
    

class MaximumIntensityProjection(PreprocessingStrategy):
    

    def run(self, preprocessing_object: PreprocessingObject, step: int) -> PreprocessingObject:
        preprocessing_object.preprocessed_image = self.run_maximum_projection_on_zstack(zstack = preprocessing_object.preprocessed_image)
        preprocessing_object.preprocessed_rois = self.adjust_rois(rois_dict = preprocessing_object.preprocessed_rois)
        preprocessing_object.database = self.update_database(database = preprocessing_object.database, file_id = preprocessing_object.file_id, step = step)
        return preprocessing_object
    
    
    def run_maximum_projection_on_zstack(self, zstack: np.ndarray) -> np.ndarray:
        # make sure that input shape matches expected shape
        return np.max(zstack, axis=0)
    
    
    def adjust_rois(self, rois_dict: Dict) -> Dict:
        # Structure of rois_dict nested dicts: 1st lvl = plane_id, 2nd lvl = roi_id
        if 'all_planes' not in rois_dict.keys():
            message_line_0 = 'For findmycells to be able to perform a MaximumIntensityProjection as preprocessing step,\n'
            message_line_1 = 'all ROIs that specify the areas for quantification must apply to all planes of the microscopy image stack.'
            message_line_2 = 'Suggested solution not yet specified - please contact segebarth_d@ukw.de for more information.'
            error_message = message_line_0 + message_line_1 + message_line_2
            raise ValueError(error_message)
        for key in rois_dict.keys():
            if key != 'all_planes':
                rois_dict.pop(key)
        return rois_dict
    
    
    def update_database(self, database: Database, file_id: str, step: int) -> Database:
        updates = dict()
        updates[f'preprocessing_step_{str(step).zfill(2)}'] = 'MaximumIntensityProjection'
        # Add additional information if neccessary
        database.update_file_infos(file_id = file_id, updates = updates)
        return database
    

    
class AdjustBrightnessAndContrast(PreprocessingStrategy):
    

    def run(self, preprocessing_object: PreprocessingObject, step: int) -> PreprocessingObject:
        preprocessing_object.preprocessed_image = self.adjust_brightness(database = preprocessing_object.database, image = preprocessing_object.preprocessed_image)
        preprocessing_object.preprocessed_image = self.adjust_contrast(database = preprocessing_object.database, image = preprocessing_object.preprocessed_image)
        preprocessing_object.database = self.update_database(database = preprocessing_object.database, file_id = preprocessing_object.file_id, step = step)
        return preprocessing_object
    
    
    def run_maximum_projection_on_zstack(self, zstack: np.ndarray) -> np.ndarray:
        # make sure that input shape matches expected shape
        return np.max(zstack, axis=0)
    
    
    def adjust_rois(self, rois_dict: Dict) -> Dict:
        # Structure of rois_dict nested dicts: 1st lvl = plane_id, 2nd lvl = roi_id
        if 'all_planes' not in rois_dict.keys():
            message_line_0 = 'For findmycells to be able to perform a MaximumIntensityProjection as preprocessing step,\n'
            message_line_1 = 'all ROIs that specify the areas for quantification must apply to all planes of the microscopy image stack.'
            message_line_2 = 'Suggested solution not yet specified - please contact segebarth_d@ukw.de for more information.'
            error_message = message_line_0 + message_line_1 + message_line_2
            raise ValueError(error_message)
        for key in rois_dict.keys():
            if key != 'all_planes':
                rois_dict.pop(key)
        return rois_dict
    
    
    def update_database(self, database: Database, file_id: str, step: int) -> Database:
        updates = dict()
        updates[f'preprocessing_step_{str(step).zfill(2)}'] = 'MaximumIntensityProjection'
        # Add additional information if neccessary
        database.update_file_infos(file_id = file_id, updates = updates)
        return database