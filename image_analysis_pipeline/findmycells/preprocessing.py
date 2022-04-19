# Probably has to become an entire module with several submodules, e.g. for cropping, channel splitting, max projections, ...

from abc import ABC, abstractmethod
import os
import numpy as np
from shapely.geometry import Polygon
from typing import List, Tuple, Dict
from skimage.io import imsave
from skimage.exposure import rescale_intensity

from .database import Database
from .microscopyimages import MicroscopyImageLoader
from .rois import ROILoader
from .utils import convert_12_to_8_bit_rgb_image
from .core import ProcessingObject, ProcessingStrategy


"""
 - Minimal preprocessing steps are:
    - save the "unprocessed" (8bit converted) microscopy images to the preprocessed_dir
    - load the unprocessed ROIs into the database (create ROI with shape of image if whole image is to be analyzed)
"""

class PreprocessingStrategy(ProcessingStrategy):
    
    @property
    def processing_type(self):
        return 'preprocessing'


class PreprocessingObject(ProcessingObject):
    
    def __init__(self, database: Database, file_ids: List[str], strategies: List[ProcessingStrategy]) -> None:
        super().__init__(database = database, file_ids = file_ids, strategies = strategies)
        self.file_id = file_ids[0]
        self.file_info = self.database.get_file_infos(identifier = self.file_id)
        self.preprocessed_image = self.load_microscopy_image()
        self.preprocessed_rois = self.load_rois()


    @property
    def processing_type(self):
        return 'preprocessing'


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


    def save_preprocessed_images_on_disk(self) -> None:
        for plane_index in range(self.preprocessed_image.shape[0]):
            image = self.preprocessed_image[plane_index].astype('uint8')
            filepath_out = self.database.preprocessed_images_dir.joinpath(f'{self.file_id}-{str(plane_index).zfill(3)}.png')
            imsave(filepath_out, image)


    def save_preprocessed_rois_in_database(self) -> None:
        self.database.import_rois_dict(file_id = self.file_id, rois_dict = self.preprocessed_rois)


    def add_processing_specific_infos_to_updates(self, updates: Dict) -> Dict:
        if self.preprocessed_image.shape[3] == 3:
            updates['RGB'] = True
        else:
            updates['RGB'] = False
        updates['total_planes'] = self.preprocessed_image.shape[0]
        return updates

            




class CropStitchingArtefactsRGB(PreprocessingStrategy):
    
    def run(self, processing_object: PreprocessingObject) -> PreprocessingObject:
        self.cropping_indices = self.determine_cropping_indices_for_entire_zstack(preprocessing_object = processing_object)
        processing_object.preprocessed_image = self.crop_rgb_zstack(zstack = processing_object.preprocessed_image)
        processing_object.preprocessed_rois = self.adjust_rois(rois_dict = processing_object.preprocessed_rois)
        return processing_object
        
    
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
        for plane_index in range(preprocessing_object.preprocessed_image.shape[0]):
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


    def add_strategy_specific_infos_to_updates(self, updates: Dict) -> Dict:
        updates['cropping_row_indices'] = (self.cropping_indices['lower_row_cropping_idx'], self.cropping_indices['upper_row_cropping_idx'])
        updates['cropping_column_indices'] = (self.cropping_indices['lower_col_cropping_idx'], self.cropping_indices['upper_col_cropping_idx']) 
        return updates
    
    
    

class CropToROIsBoundingBox(PreprocessingStrategy):
    
    def run(self, processing_object: PreprocessingObject) -> PreprocessingObject:
        self.cropping_indices = self.determine_bounding_box(preprocessing_object = processing_object, pad_size = 100)
        processing_object.preprocessed_image = self.crop_rgb_zstack(zstack = processing_object.preprocessed_image)
        processing_object.preprocessed_rois = self.adjust_rois(rois_dict = processing_object.preprocessed_rois)
        return processing_object
                                                  
    
    def determine_bounding_box(self, preprocessing_object: PreprocessingObject, pad_size: int=100) -> Dict:
        rois_dict = preprocessing_object.preprocessed_rois.copy()
        max_row_idx = preprocessing_object.preprocessed_image.shape[1]
        max_col_idx = preprocessing_object.preprocessed_image.shape[2]
        min_lower_row_cropping_idx, min_lower_col_cropping_idx, max_upper_row_cropping_idx, max_upper_col_cropping_idx = None, None, None, None
        for plane_id in rois_dict.keys():
            for roi_id in rois_dict[plane_id].keys():
                lower_row_idx, lower_col_idx, upper_row_idx, upper_col_idx =  rois_dict[plane_id][roi_id].bounds
                if min_lower_row_cropping_idx == None:
                    min_lower_row_cropping_idx, max_upper_row_cropping_idx = lower_row_idx, upper_row_idx
                    min_lower_col_cropping_idx, max_upper_col_cropping_idx = lower_col_idx, upper_col_idx
                else:
                    if lower_row_idx < min_lower_row_cropping_idx:
                        min_lower_row_cropping_idx = lower_row_idx
                    if upper_row_idx > max_upper_row_cropping_idx:
                        max_upper_row_cropping_idx = upper_row_idx
                    if lower_col_idx < min_lower_col_cropping_idx:
                        min_lower_col_cropping_idx = lower_col_idx
                    if upper_col_idx > max_upper_col_cropping_idx:
                        max_upper_col_cropping_idx = upper_col_idx
        if min_lower_row_cropping_idx - pad_size <= 0:
            min_lower_row_cropping_idx = 0
        else:
            min_lower_row_cropping_idx -= pad_size
        if min_lower_col_cropping_idx - pad_size <= 0:
            min_lower_col_cropping_idx = 0
        else:
            min_lower_col_cropping_idx -= pad_size
        
        if max_upper_row_cropping_idx + pad_size >= max_row_idx:
            max_upper_row_cropping_idx = max_row_idx
        else:
            max_upper_row_cropping_idx += pad_size
        if max_upper_col_cropping_idx + pad_size >= max_col_idx:
            max_upper_col_cropping_idx = max_col_idx
        else:
            max_upper_col_cropping_idx += pad_size        
    
        cropping_indices = {'lower_row_cropping_idx': int(min_lower_row_cropping_idx),
                            'upper_row_cropping_idx': int(max_upper_row_cropping_idx),
                            'lower_col_cropping_idx': int(min_lower_col_cropping_idx),
                            'upper_col_cropping_idx': int(max_upper_col_cropping_idx)}
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

    
    def add_strategy_specific_infos_to_updates(self, updates: Dict) -> Dict:
        updates['cropping_row_indices'] = (self.cropping_indices['lower_row_cropping_idx'], self.cropping_indices['upper_row_cropping_idx'])
        updates['cropping_column_indices'] = (self.cropping_indices['lower_col_cropping_idx'], self.cropping_indices['upper_col_cropping_idx'])
        return updates 



class ConvertTo8Bit(PreprocessingStrategy):
    
    def run(self, processing_object: PreprocessingObject) -> PreprocessingObject:
        processing_object.preprocessed_image = self.convert_to_8bit(zstack = processing_object.preprocessed_image)
        return processing_object
    
    def convert_to_8bit(self, zstack: np.ndarray) -> np.ndarray:
        max_value = zstack.max()
        if max_value <= 255:
            pass
        elif max_value <= 4095:
            for plane_index in range(zstack.shape[0]):
                zstack[plane_index] = (zstack[plane_index] / 4095 * 255).round(0)
        elif max_value <= 65535:
            for plane_index in range(zstack.shape[0]):
                zstack[plane_index] = (zstack[plane_index] / 4095 * 255).round(0)
        if zstack.dtype.name != 'uint8':
            zstack = zstack.astype('uint8')
        return zstack
    

    def add_strategy_specific_infos_to_updates(self, updates: Dict) -> Dict:
        return updates 
    


class MaximumIntensityProjection(PreprocessingStrategy):

    def run(self, processing_object: PreprocessingObject) -> PreprocessingObject:
        processing_object.preprocessed_image = self.run_maximum_projection_on_zstack(zstack = processing_object.preprocessed_image)
        processing_object.preprocessed_rois = self.adjust_rois(rois_dict = processing_object.preprocessed_rois)
        return processing_object
    
    
    def run_maximum_projection_on_zstack(self, zstack: np.ndarray) -> np.ndarray:
        max_projection = np.max(zstack, axis=0)
        return max_projection[np.newaxis, :]
    
    
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
    
    
    def add_strategy_specific_infos_to_updates(self, updates: Dict) -> Dict:
        return updates 
    


class MinimumIntensityProjection(PreprocessingStrategy):

    def run(self, processing_object: PreprocessingObject) -> PreprocessingObject:
        processing_object.preprocessed_image = self.run_minimum_projection_on_zstack(zstack = processing_object.preprocessed_image)
        processing_object.preprocessed_rois = self.adjust_rois(rois_dict = processing_object.preprocessed_rois)
        return processing_object
    
    
    def run_minimum_projection_on_zstack(self, zstack: np.ndarray) -> np.ndarray:
        min_projection = np.min(zstack, axis=0)
        return min_projection[np.newaxis, :]
    
    
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
    
    
    def add_strategy_specific_infos_to_updates(self, updates: Dict) -> Dict:
        return updates 
    


class AdjustBrightnessAndContrast(PreprocessingStrategy):

    def run(self, processing_object: PreprocessingObject) -> PreprocessingObject:
        self.percentage_saturated_pixels, self.channel_adjustment_method = self.get_method_specific_attributes(database= processing_object.database)
        processing_object.preprocessed_image = self.adjust_brightness_and_contrast(zstack = processing_object.preprocessed_image,
                                                                                      percentage_saturated_pixels = self.percentage_saturated_pixels, 
                                                                                      channel_adjustment_method = self.channel_adjustment_method)
        return processing_object


    def get_method_specific_attributes(self, database: Database) -> Tuple[float, str]:
        if hasattr(database, 'preprocessing_configs'):
            if 'AdjustBrightnessAndContrast' in database.preprocessing_configs.keys():
                percentage_saturated_pixels = database.preprocessing_configs['AdjustBrightnessAndContrast']['percentage_saturated_pixels']
                channel_adjustment_method = database.preprocessing_configs['AdjustBrightnessAndContrast']['channel_adjustment_method']
            else:
                percentage_saturated_pixels = 0.0
                channel_adjustment_method = 'globally'
        else:
            percentage_saturated_pixels = 0.0
            channel_adjustment_method = 'globally'            
        return percentage_saturated_pixels, channel_adjustment_method
    
    
    def adjust_brightness_and_contrast(self, zstack: np.ndarray, percentage_saturated_pixels: float, channel_adjustment_method: str) -> np.ndarray:
        """
        percentage_saturated_pixels: float, less than 50.0
        channel_adjustment_method: str, one of: 'individually', 'global'
        """
        adjusted_zstack = zstack.copy()
        if percentage_saturated_pixels >= 50:
            message_line0 = 'The percentage of saturated pixels cannot be set to values equal to or higher than 50.\n'
            message_line1 = 'Suggested default (also used by the ImageJ Auto Adjust method): 0.35'
            error_message = message_line0 + message_line1
            raise ValueError(error_message)
        if channel_adjustment_method == 'individually':
            self.min_max_ranges_per_plane_and_channel = list()
            for plane_index in range(adjusted_zstack.shape[0]):
                min_max_ranges = list()
                for channel_index in range(adjusted_zstack.shape[3]):
                    in_range_min = int(round(np.percentile(adjusted_zstack[plane_index, :, :, channel_index], percentage_saturated_pixels), 0))
                    in_range_max = int(round(np.percentile(adjusted_zstack[plane_index, :, :, channel_index], 100 - percentage_saturated_pixels), 0))
                    in_range = (in_range_min, in_range_max)
                    adjusted_zstack[plane_index, :, :, channel_index] = rescale_intensity(image = adjusted_zstack[plane_index, :, :, channel_index], in_range = in_range)
                    min_max_ranges.append(in_range)
                self.min_max_ranges_per_plane_and_channel.append(min_max_ranges)
        elif channel_adjustment_method == 'globally':
            self.min_max_ranges_per_plane_and_channel = list()
            for plane_index in range(adjusted_zstack.shape[0]):
                in_range_min = int(round(np.percentile(adjusted_zstack[plane_index], percentage_saturated_pixels), 0))
                in_range_max = int(round(np.percentile(adjusted_zstack[plane_index], 100 - percentage_saturated_pixels), 0))
                in_range = (in_range_min, in_range_max)
                adjusted_zstack[plane_index] = rescale_intensity(image = adjusted_zstack[plane_index], in_range = in_range)
                self.min_max_ranges_per_plane_and_channel.append(in_range)
        else:
            message_line0 = "The 'channel_adjustment_method' has to be one of: ['individually', 'globally'].\n"
            message_line1 = "-->'individually': the range of intensity values wil be calculated and scaled to the min and max values for each individual channel.\n"
            message_line2 = "-->'globally': the range of intensity values will be calculated from and scaled to the global min and max of all channels.\n" 
            message_line3 = "Either way, min and max values will be determined for each image plane individually."
            error_message = message_line0 + message_line1 + message_line2 + message_line3
            raise NotImplementedError(error_message)
        return adjusted_zstack.copy()


    def add_strategy_specific_infos_to_updates(self, updates: Dict) -> Dict:
        updates['percentage_saturated_pixels'] = self.percentage_saturated_pixels
        updates['channel_adjustment_method'] = self.channel_adjustment_method
        updates['min_max_ranges_per_plane_and_channel'] = self.min_max_ranges_per_plane_and_channel        
        return updates 