# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/api/09_inspection_00_methods.ipynb.

# %% auto 0
__all__ = ['InspectionMethod', 'InspectStackIn3D', 'InspectSinglePlane']

# %% ../../nbs/api/09_inspection_00_methods.ipynb 3
from typing import Tuple, List, Dict, Any, Optional
from traitlets.traitlets import MetaHasTraits as WidgetType
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
from skimage import io, color

from ..database import Database
from .. import utils
from ..configs import DefaultConfigs, GUIConfigs

from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt

# %% ../../nbs/api/09_inspection_00_methods.ipynb 5
class InspectionMethod(ABC):
    
    @abstractmethod
    def _initialize_default_configs(self) -> DefaultConfigs:
        pass
    

    @abstractmethod
    def _initialize_gui_configs(self) -> GUIConfigs:
        pass
    
    
    @abstractmethod
    def _method_specific_inspection(self, center_pixel_coords: Tuple[int, int], inspection_configs: Dict[str, Any]) -> None:
        pass
    
    @property
    @abstractmethod
    def dropdown_option_value_for_gui(self) -> str:
        pass
    
    
    def load_data(self, file_id: str, area_roi_id: str, database: Database, plane_idx: Optional[int]=None) -> None:
        self.file_id = file_id
        self.area_roi_id = area_roi_id
        self.database = database
        self.plane_idx = plane_idx
        self.preprocessed_image = self._load_preprocessed_image()
        self.postprocessed_segmentation_mask = self._load_postprocessed_segmentation_mask()
        self.rgb_color_coded_2d_overlay_of_image_and_mask = self._create_rgb_color_coded_2d_overlay_of_image_and_mask()
        self.area_roi_boundary_coords = self._load_area_roi_boundary_coords()
        self.default_configs = self._initialize_default_configs()
        self.gui_configs = self._initialize_gui_configs()

    
    def _load_preprocessed_image(self) -> np.ndarray:
        preprocessed_images_dir_path = self.database.project_configs.root_dir.joinpath(self.database.preprocessed_images_dir)
        preprocessed_image = utils.load_zstack_as_array_from_single_planes(path = preprocessed_images_dir_path, file_id = self.file_id)
        if type(self.plane_idx) == int:
            preprocessed_image = preprocessed_image[self.plane_idx]
        return preprocessed_image


    def _load_postprocessed_segmentation_mask(self) -> np.ndarray:
        postprocessed_masks_dir_path = self.database.project_configs.root_dir.joinpath(self.database.quantified_segmentations_dir, self.area_roi_id)
        postprocessed_mask = utils.load_zstack_as_array_from_single_planes(path = postprocessed_masks_dir_path, file_id = self.file_id)
        if type(self.plane_idx) == int:
            postprocessed_mask = postprocessed_mask[self.plane_idx]
        return postprocessed_mask
    
      
    def _create_rgb_color_coded_2d_overlay_of_image_and_mask(self) -> np.ndarray:
        if type(self.plane_idx) == int:
            image_converted_to_2d = self.preprocessed_image
            mask_converted_to_2d = self.postprocessed_segmentation_mask
        else: # means image & mask np arrays are of shape (planes, rows, cols, colors)
            if self.preprocessed_image.shape[0] > 1:
                image_converted_to_2d = np.max(self.preprocessed_image, axis=0)
                mask_converted_to_2d = np.max(self.postprocessed_segmentation_mask, axis=0)
            else:
                image_converted_to_2d = self.preprocessed_image[0]
                mask_converted_to_2d = self.postprocessed_segmentation_mask[0]
        return color.label2rgb(mask_converted_to_2d, image = image_converted_to_2d, bg_label = 0, bg_color = None, saturation = 1, alpha = 1)     
    

    def _load_area_roi_boundary_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        y_coords, x_coords = self.database.area_rois_for_quantification[self.file_id]['all_planes'][self.area_roi_id].boundary.coords.xy
        return np.asarray(x_coords, dtype='int'), np.asarray(y_coords, dtype='int')
    
    
    def get_available_label_ids(self) -> List[int]:
        available_label_ids = list(np.unique(self.postprocessed_segmentation_mask))
        if 0 in available_label_ids:
            available_label_ids.remove(0)
        return available_label_ids
    
    
    def get_available_multi_match_idxs(self) -> List[int]:
        multi_match_idxs = []
        if hasattr(self.database, 'multi_matches_traceback') == True:
            if self.file_id in self.database.multi_matches_traceback.keys():
                multi_match_idxs = self.database.multi_matches_traceback[self.file_id]['final_label_id']
        return multi_match_idxs
    
    
    def get_center_coords_from_label_id(self, label_id: int) -> Tuple[int, int]:
        if self.plane_idx == None: # means shape looks like: (planes, rows, cols, colors)
            if self.postprocessed_segmentation_mask.shape[0] > 1:
                mask_cleared_of_all_other_label_ids = self.postprocessed_segmentation_mask.copy()
                mask_cleared_of_all_other_label_ids[np.where(mask_cleared_of_all_other_label_ids != label_id)] = 0
                mask_as_single_plane = np.max(mask_cleared_of_all_other_label_ids, axis=0)
            else:
                mask_as_single_plane = self.postprocessed_segmentation_mask[0]
        else:
            mask_as_single_plane = self.postprocessed_segmentation_mask
        feature_roi = utils.get_polygon_from_instance_segmentation(single_plane = mask_as_single_plane, label_id = label_id)
        return (feature_roi.centroid.y, feature_roi.centroid.x)


    def get_center_coords_from_multi_match_idx(self, multi_match_idx: int) -> Tuple[int, int]:
        label_id = self.database.multi_matches_traceback[self.file_id]['final_label_id'][multi_match_idx]
        return self.get_center_coords_from_label_id(label_id = label_id)

    
    def get_center_coords_from_mouse_click_position(self, target_output_widget: Optional[WidgetType]=None) -> Tuple[int, int]:
        if target_output_widget != None:
            self.target_output_widget = target_output_widget
        self._check_for_matplotlib_setup()
        fig = plt.figure(figsize=(10, 10), facecolor = 'white')
        plt.connect('button_press_event', self._matplotlib_figure_clicked)
        plt.imshow(self.rgb_color_coded_2d_overlay_of_image_and_mask)
        plt.plot(self.area_roi_boundary_coords[0], self.area_roi_boundary_coords[1], c = 'cyan')
        plt.show()
        
        
    def _check_for_matplotlib_setup(self) -> None:
        if hasattr(self, 'matplotlib_all_set_up') == False:
            from IPython import get_ipython
            ipy = get_ipython()
            if ipy is not None:
                ipy.run_line_magic('matplotlib', 'tk')
            self.matplotlib_all_set_up = True
        else:
            pass

        
    def _matplotlib_figure_clicked(self, event):
        if event.button is MouseButton.RIGHT:
            plt.close()
            if hasattr(self, 'target_output_widget') == True:
                with self.target_output_widget:
                    self.target_output_widget.clear_output()
                    print(f'x: {event.x}, and y: {event.y}')
            else:
                print(f'x: {event.x}, and y: {event.y}')    
                
        
    def build_widget_for_remaining_conifgs(self) -> None:
        info_text = 'Additional configs for your inspection plot:'
        self.gui_configs.construct_widget(info_text = info_text, default_configs = self.default_configs)
        self.widget = self.gui_configs.strategy_widget
   
    
    def run_inspection(self, center_pixel_coords: Tuple[int, int], inspection_configs: Optional[Dict[str, Any]]=None) -> None:
        self._check_for_matplotlib_setup()
        inspection_configs = self._validate_and_update_inspection_configs(inspection_configs = inspection_configs)
        self._method_specific_inspection(center_pixel_coords = center_pixel_coords, inspection_configs = inspection_configs)
        
        
    def _validate_and_update_inspection_configs(self, inspection_configs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if type(inspection_configs) != dict:
            inspection_configs = self.default_configs.values
        else:
            self.default_configs.assert_user_input(user_input = inspection_configs)
            inspection_configs = self.default_configs.fill_user_input_with_defaults_where_needed(user_input = inspection_configs)
        return inspection_configs


    def _calculate_cropping_boundaries(self, center_coords: Tuple[int, int], box_height: int, box_width: int) -> Dict:
        center_col, center_row = center_coords
        max_row, max_col = self.rgb_color_coded_2d_overlay_of_image_and_mask.shape[:2]
        lower_row_cropping_idx, upper_row_cropping_idx = self._determine_cropping_indices(center_px_idx = center_row, 
                                                                                          half_box_size = box_height // 2, 
                                                                                          max_idx = max_row)
        lower_col_cropping_idx, upper_col_cropping_idx = self._determine_cropping_indices(center_px_idx = center_col,
                                                                                          half_box_size = box_width // 2,
                                                                                          max_idx = max_col)
        cropping_boundaries = {'lower_row': lower_row_cropping_idx,
                               'upper_row': upper_row_cropping_idx,
                               'lower_col': lower_col_cropping_idx,
                               'upper_col': upper_col_cropping_idx}
        return cropping_boundaries
    
    
    def _determine_cropping_indices(self, center_px_idx: int, half_box_size: int, max_idx: int) -> Tuple[int, int]:
        if (center_px_idx - half_box_size >= 0) & (center_px_idx + half_box_size <= max_idx):
            lower_cropping_index, upper_cropping_index = center_px_idx - half_box_size,  center_px_idx + half_box_size
        elif 2*half_box_size <= max_idx:
            if center_px_idx - half_box_size < 0:
                lower_cropping_index, upper_cropping_index = 0,  0 + 2*half_box_size
            else: # means: center_px_idx + half_box_size > max_index
                lower_cropping_index, upper_cropping_index = max_idx - 2*half_box_size, max_idx
        else:
            raise ValueError((f'The desired box size {2*half_box_size}) is larger than one of the '
                              f'image axes ({max_idx}). Please select a smaller "box_size"!'))
        return lower_cropping_index, upper_cropping_index

# %% ../../nbs/api/09_inspection_00_methods.ipynb 6
class InspectStackIn3D(InspectionMethod):
    
    @property
    def dropdown_option_value_for_gui(self) -> str:
        return 'Inspect using interactive 3D plots'

    
    def _initialize_default_configs(self) -> DefaultConfigs:
        max_box_size = min(self.rgb_color_coded_2d_overlay_of_image_and_mask.shape[0:2])
        box_size, binning_factor = self._get_default_box_size_and_binning_factor(max_box_size = max_box_size)
        default_values = {'box_size': box_size,
                          'binning_factor': binning_factor,
                          'show': True,
                          'save': False}
        valid_types = {'box_size': [int],
                       'binning_factor': [int],
                       'show': [bool], 
                       'save': [bool]}
        valid_ranges = {'box_size': (50, max_box_size, 50),
                        'binning_factor': (0, 100, 1)}
        default_configs = DefaultConfigs(default_values = default_values,
                                         valid_types = valid_types,
                                         valid_value_ranges = valid_ranges)
        return default_configs
    
    
    def _initialize_gui_configs(self) -> GUIConfigs:
        descriptions = {'box_size': 'Specify the size of the area that you would like to inspect [px]:',
                        'binning_factor': 'We recommend to select a binning factor such that: box size / binning factor ~ 50 (no remainder allowed!):',
                        'show': 'show the plot',
                        'save': 'save the resulting plot'}
        widget_names = {'box_size': 'IntSlider',
                        'binning_factor': 'IntSlider',
                        'show': 'Checkbox',
                        'save': 'Checkbox'}
        return GUIConfigs(widget_names = widget_names, descriptions = descriptions)
    
    
    def _get_default_box_size_and_binning_factor(self, max_box_size: int) -> Tuple[int, int]:
        size_factor = max_box_size // 500
        default_box_size = size_factor * 100
        if default_box_size == 0:
            default_box_size = min(max_box_size, 100)
        if default_box_size < 100:
            default_binning_factor = 0
        else:
            default_binning_factor = default_box_size // 50
        return default_box_size, default_binning_factor
    
    
    
    def _method_specific_inspection(self,
                                    center_pixel_coords: Tuple[int, int],
                                    inspection_configs: Dict[str, Any]
                                   ) -> None:
        cropping_boundaries = self._calculate_cropping_boundaries(center_coords = center_pixel_coords,
                                                                  box_height = inspection_configs['box_size'],
                                                                  box_width = inspection_configs['box_size'])
        cropped_and_binned_mask_zstack = self._crop_and_bin_zstack(cropping_boundaries = cropping_boundaries, inspection_configs = inspection_configs)
        voxel_color_code = color.label2rgb(cropped_and_binned_mask_zstack)
        self._create_3d_plot(box_boundaries = cropping_boundaries, 
                             voxels = cropped_and_binned_mask_zstack, 
                             color_code = voxel_color_code, 
                             inspection_configs = inspection_configs)
    

    def _crop_and_bin_zstack(self, cropping_boundaries: Dict, inspection_configs: Dict[str, Any]) -> np.ndarray:
        zstack = self.postprocessed_segmentation_mask
        cropped_zstack = zstack[:,
                                cropping_boundaries['lower_row']:cropping_boundaries['upper_row'],
                                cropping_boundaries['lower_col']:cropping_boundaries['upper_col']].copy()
        if inspection_configs['binning_factor'] > 0:
            cropped_and_binned_zstack = self._bin_zstack(zstack_to_bin = cropped_zstack, binning_factor = inspection_configs['binning_factor'])
        else:
            cropped_and_binned_zstack = cropped_zstack
        return cropped_and_binned_zstack
    
    
    def _bin_zstack(self, zstack_to_bin: np.ndarray, binning_factor: int) -> np.ndarray:
        zstack, new_shape = self._adjust_zstack_for_binning_to_new_shape(input_zstack = zstack_to_bin, binning_factor = binning_factor)
        binned_single_planes = []
        for plane_index in range(zstack.shape[0]):
            binned_single_planes.append(self._bin_2d_image(single_plane = zstack[plane_index], new_shape = new_shape))
        return np.asarray(binned_single_planes)
    
    
    def _adjust_zstack_for_binning_to_new_shape(self, input_zstack: np.ndarray, binning_factor: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        zstack = input_zstack.copy()
        rows, cols = zstack[0].shape[:2]
        if rows % binning_factor != 0:
            excess_pixels = rows % binning_factor
            cropping_lower_end = int(excess_pixels/2)
            cropping_upper_end = rows - (excess_pixels - cropping_lower_end)
            zstack = zstack[:, cropping_lower_end:cropping_upper_end, :]
        if cols % binning_factor != 0:
            excess_pixels = cols % binning_factor
            cropping_lower_end = int(excess_pixels/2)
            cropping_upper_end = cols - (excess_pixels - cropping_lower_end)
            zstack = zstack[:, :, cropping_lower_end:cropping_upper_end]
        adjusted_rows, adjusted_cols = zstack[0].shape[:2]
        if (adjusted_rows % binning_factor != 0) or (adjusted_cols % binning_factor != 0):
            raise ValueError(('Sorry! Something went wrong during the binning process. To avoid this issue, '
                              'consider changing the "box_size" or the "binning_factor", such that '
                              'division of the box size by the binning factor does not yield any remainder!'))
        else:
            new_shape = (int(adjusted_rows / binning_factor), int(adjusted_cols / binning_factor))
        return zstack, new_shape
    
    
    def _bin_2d_image(self, single_plane: np.ndarray, new_shape: Tuple[int, int]):
        shape = (new_shape[0], single_plane.shape[0] // new_shape[0],
                 new_shape[1], single_plane.shape[1] // new_shape[1])
        return single_plane.reshape(shape).max(-1).max(1)
    

    def _create_3d_plot(self, box_boundaries: Dict, voxels: np.ndarray, color_code: np.ndarray, inspection_configs: Dict[str, Any]) -> None:
        
        box_size = box_boundaries['upper_row'] - box_boundaries['lower_row']
        center_row_index = box_boundaries['lower_row'] + 0.5*inspection_configs['box_size']
        center_col_index = box_boundaries['lower_col'] + 0.5*inspection_configs['box_size']
        box_row_coords = [box_boundaries['lower_row'], box_boundaries['lower_row'],
                          box_boundaries['upper_row'], box_boundaries['upper_row'], 
                          box_boundaries['lower_row']]
        box_col_coords = [box_boundaries['lower_col'], box_boundaries['upper_col'], 
                          box_boundaries['upper_col'], box_boundaries['lower_col'], 
                          box_boundaries['lower_col']]

        fig = plt.figure(figsize=(12,8), facecolor='white')
        gs = fig.add_gridspec(1,3)

        fig.add_subplot(gs[0,0])
        plt.imshow(self.rgb_color_coded_2d_overlay_of_image_and_mask)
        plt.plot(self.area_roi_boundary_coords[0], self.area_roi_boundary_coords[1])
        plt.plot(box_col_coords, box_row_coords, c='magenta', lw='3', linestyle='dashed')

        ax1 = fig.add_subplot(gs[0,1], projection='3d')
        ax1.voxels(voxels, facecolors=color_code)
        ax1.view_init(elev=30, azim=33)

        ax2 = fig.add_subplot(gs[0,2], projection='3d')
        ax2.voxels(voxels, facecolors=color_code)
        ax2.view_init(elev=30, azim=213)

        plt.suptitle((f'Area_id {self.area_roi_id} of file_id {self.file_id}, '
                      f'centered at ({center_row_index}, {center_col_index}) '
                      f'with a binning of factor {inspection_configs["binning_factor"]}'), y=0.9)
        if inspection_configs['save'] == True:
            filename = f'{self.file_id}_3D_inspection_of_{self.area_roi_id}.png'
            filepath = self.database.project_configs.root_dir.joinpath(self.database.inspection_dir, filename)
            plt.savefig(filepath, dpi=300)
        if inspection_configs['show'] == True:
            plt.show()
        else:
            plt.close()   

# %% ../../nbs/api/09_inspection_00_methods.ipynb 7
class InspectSinglePlane(InspectionMethod):
    
    @property
    def dropdown_option_value_for_gui(self) -> str:
        return 'Inspect single plane(s) using interactive 2D plots'

    
    def _initialize_default_configs(self) -> DefaultConfigs:
        max_box_shape = self.rgb_color_coded_2d_overlay_of_image_and_mask.shape[:2]
        default_values = {'box_height': max_box_shape[0] // 20 * 2,
                          'box_width': max_box_shape[1] // 20 * 2,
                          'show': True,
                          'save': False}
        valid_types = {'box_height': [int],
                       'box_width': [int],
                       'show': [bool], 
                       'save': [bool]}
        valid_ranges = {'box_height': (10, max_box_shape[0], 2),
                        'box_width': (10, max_box_shape[1], 2)}
        default_configs = DefaultConfigs(default_values = default_values,
                                         valid_types = valid_types,
                                         valid_value_ranges = valid_ranges)
        return default_configs
    
    
    def _initialize_gui_configs(self) -> GUIConfigs:
        descriptions = {'box_height': 'Specify the height of the area that you would like to inspect [px]:',
                        'box_width': 'Specify the width of the area that you would like to inspect [px]:',
                        'show': 'show the plot',
                        'save': 'save the resulting plot'}
        widget_names = {'box_height': 'IntSlider',
                        'box_width': 'IntSlider',
                        'show': 'Checkbox',
                        'save': 'Checkbox'}
        return GUIConfigs(widget_names = widget_names, descriptions = descriptions)

    
    def _method_specific_inspection(self,
                                    center_pixel_coords: Tuple[int, int],
                                    inspection_configs: Dict[str, Any]
                                   ) -> None:
        cropping_boundaries = self._calculate_cropping_boundaries(center_coords = center_pixel_coords,
                                                                  box_height = inspection_configs['box_height'],
                                                                  box_width = inspection_configs['box_width'])
        if self.plane_idx == None: # means image & mask shape looks like: (planes, rows, cols, colors)
            self._convert_image_and_mask_to_correct_2d_format()
        cropped_image = self._crop_single_plane_array(cropping_boundaries = cropping_boundaries,
                                                      single_plane = self.preprocessed_image)
        color_coded_mask = color.label2rgb(self.postprocessed_segmentation_mask, bg_label = 0, bg_color = None, saturation = 1, alpha = 1)
        cropped_color_coded_mask = self._crop_single_plane_array(cropping_boundaries = cropping_boundaries, single_plane = color_coded_mask)
        self._create_2d_plot(box_boundaries = cropping_boundaries, 
                             cropped_image = cropped_image,
                             cropped_color_coded_mask = cropped_color_coded_mask, 
                             inspection_configs = inspection_configs)
        
        
    def _convert_image_and_mask_to_correct_2d_format(self) -> None:
        if self.preprocessed_image.shape[0] > 1:
            self.preprocessed_image = np.max(self.preprocessed_image, axis=0)
            self.postprocessed_segmentation_mask = np.max(self.postprocessed_segmentation_mask, axis=0)
            self.max_projection_was_done = True
        else: # means shape looks like: (1, rows, cols, colors)
            self.preprocessed_image = self.preprocessed_image[0]
            self.postprocessed_segmentation_mask = self.postprocessed_segmentation_mask[0]
            self.max_projection_was_done = False
    
    
    def _crop_single_plane_array(self, cropping_boundaries: Dict, single_plane: np.ndarray) -> np.ndarray:
        cropped_array = single_plane[cropping_boundaries['lower_row']:cropping_boundaries['upper_row'],
                                     cropping_boundaries['lower_col']:cropping_boundaries['upper_col']]
        return cropped_array
    

    def _create_2d_plot(self, box_boundaries: Dict,
                        cropped_image: np.ndarray,
                        cropped_color_coded_mask: np.ndarray,
                        inspection_configs: Dict[str, Any]) -> None:        
        center_row_index = box_boundaries['lower_row'] + 0.5*inspection_configs['box_height']
        center_col_index = box_boundaries['lower_col'] + 0.5*inspection_configs['box_width']
        box_row_coords = [box_boundaries['lower_row'], box_boundaries['lower_row'],
                          box_boundaries['upper_row'], box_boundaries['upper_row'], 
                          box_boundaries['lower_row']]
        box_col_coords = [box_boundaries['lower_col'], box_boundaries['upper_col'], 
                          box_boundaries['upper_col'], box_boundaries['lower_col'], 
                          box_boundaries['lower_col']]

        fig = plt.figure(figsize = (12, 8), facecolor='white')
        gs = fig.add_gridspec(1,3)

        ax0 = fig.add_subplot(gs[0,0])
        ax0 = plt.imshow(self.rgb_color_coded_2d_overlay_of_image_and_mask)
        ax0 = plt.plot(self.area_roi_boundary_coords[0], self.area_roi_boundary_coords[1])
        ax0 = plt.plot(box_col_coords, box_row_coords, c='magenta', lw='3', linestyle='dashed')

        ax1 = fig.add_subplot(gs[0,1])
        ax1 = plt.imshow(cropped_image)

        ax2 = fig.add_subplot(gs[0,2])
        ax2 = plt.imshow(cropped_color_coded_mask)
        
        plane_title_str, plane_filename_str = self._get_plane_string_for_plot_title_and_filename()
        plt.suptitle(f'{plane_title_str}segmentation mask in area ID {self.area_roi_id} of file ID {self.file_id}', y=0.9)
        if inspection_configs['save'] == True:
            filename = f'{self.file_id}_2D_inspection_of_{self.area_roi_id}{plane_filename_str}.png'
            filepath = self.database.project_configs.root_dir.joinpath(self.database.inspection_dir, filename)
            plt.savefig(filepath, dpi=300)
        if inspection_configs['show'] == True:
            plt.show()
        else:
            plt.close()
        
        
    def _get_plane_string_for_plot_title_and_filename(self) -> Tuple[str, str]:
        if self.plane_idx == None:
            if self.max_projection_was_done == True:
                plane_title_string = 'max. projection of '
                plane_filename_string = '_using_max_projection'
            else:
                plane_title_string = ''
                plane_filename_string = ''
        else:
            plane_title_string = f'plane with index {self.plane_idx} of '
            plane_filename_string = f'_of_plane_idx_{self.plane_idx}'
        return plane_title_string, plane_filename_string
