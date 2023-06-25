# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/api/05_preprocessing_01_strategies.ipynb.

# %% auto 0
__all__ = ['CropStitchingArtefactsRGBStrat', 'CropToROIsBoundingBoxStrat', 'ConvertTo8BitStrat',
           'MaximumIntensityProjectionStrat', 'MinimumIntensityProjectionStrat', 'AdjustBrightnessAndContrastStrat']

# %% ../../nbs/api/05_preprocessing_01_strategies.ipynb 2
from typing import List, Dict, Tuple
from shapely.geometry import Polygon
import numpy as np
from skimage import exposure


from .specs import PreprocessingObject, PreprocessingStrategy
from ..database import Database
from ..configs import DefaultConfigs, GUIConfigs

# %% ../../nbs/api/05_preprocessing_01_strategies.ipynb 4
class CropStitchingArtefactsRGBStrat(PreprocessingStrategy):
    
    #ToDo:
    # - Option to specify whether artefact pixel color is black or white
    # - if white: make sure to identify bit type of the image (whether white == 255, 4095, ..)
    # - check whether it also works if it´s only a single channel image
    
    """
    When you acquire microscopy images that are essentially several individual 
    images (= tiles) stitched together, you may end up with some artefacts on the
    borders of the image as a result from the stitching process. These pixels are
    usually either fully black or fully white and can therefore interfere with 
    other processing strategies that you might want to apply to your images (for 
    instance, if you´d like to adjust brightness and contrast). This strategy aims
    at identifying these pixels that were added to account for some offset between
    the individual tiles and eventually remove them. As these artefacts might 
    interfere with other processing steps, it is recommended to add this (or any other
    cropping strategy to get rid of these artefacts) prior to other preprocessing 
    strategies. 
    """
    
    @property
    def dropdown_option_value_for_gui(self):
        return 'Crop stitching artefacts (RGB image version)'
    
    @property
    def default_configs(self):
        default_values = {}
        valid_types = {}
        default_configs = DefaultConfigs(default_values = default_values, valid_types = valid_types)
        return default_configs
        
    @property
    def widget_names(self):
        return {}

    @property
    def descriptions(self):
        return {}
    
    @property
    def tooltips(self):
        return {}
    
    def run(self, processing_object: PreprocessingObject, strategy_configs: Dict) -> PreprocessingObject:
        self.cropping_indices = self._determine_cropping_indices_for_entire_zstack(preprocessing_object = processing_object)
        processing_object.preprocessed_image = processing_object.crop_rgb_zstack(zstack = processing_object.preprocessed_image,
                                                                                 cropping_indices = self.cropping_indices)
        processing_object.preprocessed_rois = processing_object.adjust_rois(rois_dict = processing_object.preprocessed_rois,
                                                                            lower_row_cropping_idx = self.cropping_indices['lower_row_cropping_idx'],
                                                                            lower_col_cropping_idx = self.cropping_indices['lower_col_cropping_idx'])
        return processing_object


    def _add_strategy_specific_infos_to_updates(self, updates: Dict) -> Dict:
        updates[f'cropping_row_indices'] = (self.cropping_indices['lower_row_cropping_idx'], 
                                            self.cropping_indices['upper_row_cropping_idx'])
        updates[f'cropping_column_indices'] = (self.cropping_indices['lower_col_cropping_idx'], 
                                               self.cropping_indices['upper_col_cropping_idx']) 
        return updates


    def _determine_cropping_indices_for_entire_zstack(self, preprocessing_object: PreprocessingObject) -> Dict:
        for plane_index in range(preprocessing_object.preprocessed_image.shape[0]):
            rgb_image_plane = preprocessing_object.preprocessed_image[plane_index]
            rows_with_black_px, columns_with_black_px = np.where(np.all(rgb_image_plane == 0, axis = -1))
            lower_row_idx, upper_row_idx = self._get_cropping_indices(rows_with_black_px)
            lower_col_idx, upper_col_idx = self._get_cropping_indices(columns_with_black_px)  
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
    
    
    def _get_cropping_indices(self, a, min_black_px_stretch: int=100) -> Tuple[int, int]:
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

# %% ../../nbs/api/05_preprocessing_01_strategies.ipynb 5
class CropToROIsBoundingBoxStrat(PreprocessingStrategy):
    
    """
    You might not be interested in analyzing the entire image, but only to quantify
    image features of interest in a certain region of your image (or actually also
    several regions). Now, chances are that it is possible to find a bounding box that
    contains all regions of the image that you are interested in, which is, however,
    smaller than the original image. Cropping your original image down to that smaller 
    size will then significantly reduce computation time, required memory space, and also
    required disk space. Therefore, it is highly recommended to add this strategy to your
    preprocessing. You can also combine it with additional cropping strategies, like the
    one that tries to remove stitching artefacts.
    """
    
    @property
    def dropdown_option_value_for_gui(self):
        return 'Crop image to bounding box enclosing all ROIs'
    
    @property
    def default_configs(self):
        default_values = {'pad_size': 100}
        valid_types = {'pad_size': [int]}
        valid_ranges = {'pad_size': (0, 500, 1)}
        default_configs = DefaultConfigs(default_values = default_values, valid_types = valid_types, valid_value_ranges = valid_ranges)
        return default_configs
        
    @property
    def widget_names(self):
        return {'pad_size': 'IntSlider'}

    @property
    def descriptions(self):
        return {'pad_size': 'Pad size [pixel]:'}
    
    @property
    def tooltips(self):
        return {}
    
    
    def run(self, processing_object: PreprocessingObject, strategy_configs: Dict) -> PreprocessingObject:
        self.cropping_indices = self._determine_bounding_box(preprocessing_object = processing_object,
                                                             pad_size = strategy_configs['pad_size'])
        processing_object.preprocessed_image = processing_object.crop_rgb_zstack(zstack = processing_object.preprocessed_image,
                                                                                 cropping_indices = self.cropping_indices)
        processing_object.preprocessed_rois = processing_object.adjust_rois(rois_dict = processing_object.preprocessed_rois,
                                                                            lower_row_cropping_idx = self.cropping_indices['lower_row_cropping_idx'],
                                                                            lower_col_cropping_idx = self.cropping_indices['lower_col_cropping_idx'])
        return processing_object
                                                  
    
    def _determine_bounding_box(self, preprocessing_object: PreprocessingObject, pad_size: int) -> Dict:
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

    
    def _add_strategy_specific_infos_to_updates(self, updates: Dict) -> Dict:
        updates[f'cropping_row_indices'] = (self.cropping_indices['lower_row_cropping_idx'], 
                                            self.cropping_indices['upper_row_cropping_idx'])
        updates[f'cropping_column_indices_step'] = (self.cropping_indices['lower_col_cropping_idx'], 
                                                    self.cropping_indices['upper_col_cropping_idx']) 
        return updates

# %% ../../nbs/api/05_preprocessing_01_strategies.ipynb 6
class ConvertTo8BitStrat(PreprocessingStrategy):
    
    """
    This strategy converts your image to an 8-bit format. Adding this strategy is
    at the moment mandatory, as all implemented segmentation tools (deepflash2 & cellpose)
    require 8-bit as input format. So you actually don´t really have a choice but adding it! :-)
    """
    
    @property
    def dropdown_option_value_for_gui(self):
        return 'Convert into 8-bit format'
    
    @property
    def default_configs(self):
        default_values = {}
        valid_types = {}
        default_configs = DefaultConfigs(default_values = default_values, valid_types = valid_types)
        return default_configs
        
    @property
    def widget_names(self):
        return {}

    @property
    def descriptions(self):
        return {}
    
    @property
    def tooltips(self):
        return {}
    
    
    def run(self, processing_object: PreprocessingObject, strategy_configs: Dict) -> PreprocessingObject:
        processing_object.preprocessed_image = self._convert_to_8bit(zstack = processing_object.preprocessed_image)
        return processing_object
    
    
    def _convert_to_8bit(self, zstack: np.ndarray) -> np.ndarray:
        max_value = zstack.max()
        if max_value <= 255:
            pass
        elif max_value <= 4095:
            for plane_index in range(zstack.shape[0]):
                zstack[plane_index] = (zstack[plane_index] / 4095 * 255).round(0)
        elif max_value <= 65535:
            for plane_index in range(zstack.shape[0]):
                zstack[plane_index] = (zstack[plane_index] / 65535 * 255).round(0)
        if zstack.dtype.name != 'uint8':
            zstack = zstack.astype('uint8')
        return zstack
    

    def _add_strategy_specific_infos_to_updates(self, updates: Dict) -> Dict:
        return updates 

# %% ../../nbs/api/05_preprocessing_01_strategies.ipynb 7
class MaximumIntensityProjectionStrat(PreprocessingStrategy):

    """
    If you acquired your microscopy images as z-stack, you can use this strategy to
    project it from a 3D image stack (commonly referred to as 2.5D) into a two
    dimensional single plane image. If you select this strategy, the brightest (= maximal)
    pixel value from the z-stack will be used in the final 2D projection. Alternatively,
    feel free to use the "Minimum intenstity projection" strategy, if you´d like to 
    keep only the darkest (= minimal) value of each pixel.
    """
    
    @property
    def dropdown_option_value_for_gui(self):
        return 'Maximum intensity projection'
    
    @property
    def default_configs(self):
        default_values = {}
        valid_types = {}
        default_configs = DefaultConfigs(default_values = default_values, valid_types = valid_types)
        return default_configs
        
    @property
    def widget_names(self):
        return {}

    @property
    def descriptions(self):
        return {}
    
    @property
    def tooltips(self):
        return {}
    
    
    def run(self, processing_object: PreprocessingObject, strategy_configs: Dict) -> PreprocessingObject:
        processing_object.preprocessed_image = self._run_maximum_projection_on_zstack(zstack = processing_object.preprocessed_image)
        processing_object.preprocessed_rois = self._remove_all_single_plane_rois(rois_dict = processing_object.preprocessed_rois)
        return processing_object
    
    
    def _run_maximum_projection_on_zstack(self, zstack: np.ndarray) -> np.ndarray:
        max_projection = np.max(zstack, axis=0)
        return max_projection[np.newaxis, :]
    
    
    def _remove_all_single_plane_rois(self, rois_dict: Dict[str, Dict[str, Polygon]]) -> Dict[str, Dict[str, Polygon]]:
        if 'all_planes' not in rois_dict.keys():
            raise ValueError('For findmycells to be able to perform a MaximumIntensityProjection as preprocessing step, '
                             'all ROIs that specify the areas for quantification must apply to all planes of the microscopy image stack.')
        for key in rois_dict.keys():
            if key != 'all_planes':
                rois_dict.pop(key)
        return rois_dict
    
    
    def _add_strategy_specific_infos_to_updates(self, updates: Dict) -> Dict:
        return updates 

# %% ../../nbs/api/05_preprocessing_01_strategies.ipynb 8
class MinimumIntensityProjectionStrat(PreprocessingStrategy):
    
    """
    If you acquired your microscopy images as z-stack, you can use this strategy to
    project it from a 3D image stack (commonly referred to as 2.5D) into a two
    dimensional single plane image. If you select this strategy, the darkest (= minimal)
    pixel value from the z-stack will be used in the final 2D projection. Alternatively,
    feel free to use the "Maximum intenstity projection" strategy, if you´d like to 
    keep only the brightest (= maximal) value of each pixel.
    """
    
    @property
    def dropdown_option_value_for_gui(self):
        return 'Minimum intensity projection'
    
    @property
    def default_configs(self):
        default_values = {}
        valid_types = {}
        default_configs = DefaultConfigs(default_values = default_values, valid_types = valid_types)
        return default_configs
        
    @property
    def widget_names(self):
        return {}

    @property
    def descriptions(self):
        return {}
    
    @property
    def tooltips(self):
        return {}
    

    def run(self, processing_object: PreprocessingObject, strategy_configs: Dict) -> PreprocessingObject:
        processing_object.preprocessed_image = self._run_minimum_projection_on_zstack(zstack = processing_object.preprocessed_image)
        processing_object.preprocessed_rois = self._remove_all_single_plane_rois(rois_dict = processing_object.preprocessed_rois)
        return processing_object
    
    
    def _run_minimum_projection_on_zstack(self, zstack: np.ndarray) -> np.ndarray:
        min_projection = np.min(zstack, axis=0)
        return min_projection[np.newaxis, :]
    
    
    def _remove_all_single_plane_rois(self, rois_dict: Dict[str, Dict[str, Polygon]]) -> Dict[str, Dict[str, Polygon]]:
        if 'all_planes' not in rois_dict.keys():
            raise ValueError('For findmycells to be able to perform a MaximumIntensityProjection as preprocessing step, '
                             'all ROIs that specify the areas for quantification must apply to all planes of the microscopy image stack.')
        for key in rois_dict.keys():
            if key != 'all_planes':
                rois_dict.pop(key)
        return rois_dict
    
    
    def _add_strategy_specific_infos_to_updates(self, updates: Dict) -> Dict:
        return updates 

# %% ../../nbs/api/05_preprocessing_01_strategies.ipynb 10
class AdjustBrightnessAndContrastStrat(PreprocessingStrategy):

    """
    This strategy allows you to automatically adjust brightness and contrast
    of your images. For this, please specify the percentage of pixels that
    you want to be saturated (default: 0.35 % - same as in ImageJ2). This 
    strategy will then ensure that this specified percentage of pixels will
    be fully saturated in all of your images. If you have z-stack images,
    you can furthermore also specify whether you´d like to run this operation
    on the full z-stack (chose "globally"), or on each individual plane of the
    z-stack (chose "individually"). I would rather recommend using "globally" 
    to keep a somewhat consistent meaning of pixel intensities. And, finally, 
    if you are anyhow dealing with 2D images (either from the get-go, or since
    you applied a maximum or minimum intensity projection strategy prior to
    this one - both "globally" and "individually" will lead to the same result.
    """
    
    @property
    def dropdown_option_value_for_gui(self):
        return 'Adjust brightness and contrast'
    
    @property
    def default_configs(self):
        default_values = {'percentage_saturated_pixels': 0.35,
                          'channel_adjustment_method': 'globally'}
        valid_types = {'percentage_saturated_pixels': [float],
                       'channel_adjustment_method': [str]}
        valid_ranges = {'percentage_saturated_pixels': (0.05, 49.95, 0.05)}
        valid_options = {'channel_adjustment_method': ('globally', 'individually')}
        default_configs = DefaultConfigs(default_values = default_values,
                                         valid_types = valid_types,
                                         valid_value_ranges = valid_ranges,
                                         valid_value_options = valid_options)
        return default_configs
        
    @property
    def widget_names(self):
        return {'percentage_saturated_pixels': 'FloatSlider',
                'channel_adjustment_method': 'Dropdown'}

    @property
    def descriptions(self):
        return {'percentage_saturated_pixels': 'Percentage of pixels that will be saturated:',
                'channel_adjustment_method': 'Adjust on whole zstack level (= globally) or for each plane (= individually):'}
    
    @property
    def tooltips(self):
        return {}

    def run(self, processing_object: PreprocessingObject, strategy_configs: Dict) -> PreprocessingObject:
        processing_object.preprocessed_image = self._adjust_brightness_and_contrast(zstack = processing_object.preprocessed_image,
                                                                                    percentage_saturated_pixels = strategy_configs['percentage_saturated_pixels'], 
                                                                                    channel_adjustment_method = strategy_configs['channel_adjustment_method'])
        return processing_object
    
    
    def _adjust_brightness_and_contrast(self, zstack: np.ndarray, percentage_saturated_pixels: float, channel_adjustment_method: str) -> np.ndarray:
        """
        percentage_saturated_pixels: float, less than 50.0
        channel_adjustment_method: str, one of: 'individually', 'globally'
        """
        adjusted_zstack = zstack.copy()
        if percentage_saturated_pixels >= 50:
            message_line0 = 'The percentage of saturated pixels cannot be set to values equal to or higher than 50.\n'
            message_line1 = 'Suggested default (also used by the ImageJ Auto Adjust method): 0.35'
            error_message = message_line0 + message_line1
            raise ValueError(error_message)
        if channel_adjustment_method == 'individually':
            self.min_max_ranges_per_plane_and_channel = []
            for plane_index in range(adjusted_zstack.shape[0]):
                min_max_ranges = []
                for channel_index in range(adjusted_zstack.shape[3]):
                    in_range_min = int(round(np.percentile(adjusted_zstack[plane_index, :, :, channel_index], percentage_saturated_pixels), 0))
                    in_range_max = int(round(np.percentile(adjusted_zstack[plane_index, :, :, channel_index], 100 - percentage_saturated_pixels), 0))
                    in_range = (in_range_min, in_range_max)
                    adjusted_zstack[plane_index, :, :, channel_index] = exposure.rescale_intensity(image = adjusted_zstack[plane_index, :, :, channel_index], in_range = in_range)
                    min_max_ranges.append(in_range)
                self.min_max_ranges_per_plane_and_channel.append(min_max_ranges)
        elif channel_adjustment_method == 'globally':
            self.min_max_ranges_per_plane_and_channel = []
            for plane_index in range(adjusted_zstack.shape[0]):
                in_range_min = int(round(np.percentile(adjusted_zstack[plane_index], percentage_saturated_pixels), 0))
                in_range_max = int(round(np.percentile(adjusted_zstack[plane_index], 100 - percentage_saturated_pixels), 0))
                in_range = (in_range_min, in_range_max)
                adjusted_zstack[plane_index] = exposure.rescale_intensity(image = adjusted_zstack[plane_index], in_range = in_range)
                self.min_max_ranges_per_plane_and_channel.append(in_range)
        else:
            raise NotImplementedError("The 'channel_adjustment_method' has to be one of: ['individually', 'globally'].\n",
                                      "-->'individually': the range of intensity values wil be calculated and scaled to the "
                                      "min and max values for each individual channel.\n"
                                      "-->'globally': the range of intensity values will be calculated from and scaled to the "
                                      "global min and max of all channels.\n"
                                      "Either way, min and max values will be determined for each image plane individually.")
        return adjusted_zstack.copy()


    def _add_strategy_specific_infos_to_updates(self, updates: Dict) -> Dict:
        updates['min_max_ranges_per_plane_and_channel'] = self.min_max_ranges_per_plane_and_channel        
        return updates 
