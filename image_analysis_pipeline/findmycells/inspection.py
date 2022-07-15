import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb

from typing import Tuple, Dict

from .main import Project
from .utils import load_zstack_as_array_from_single_planes, listdir_nohidden


def inspect_in_3d(fmc_project: Project, file_id: str, area_id: str, center_pixel_coords: Tuple[int, int], half_box_size: int=100, binning_factor: int=0, save: bool=False, show: bool=True) -> None:
    image_zstack, mask_zstack, area_roi_coordinates = _load_data(fmc_project = fmc_project, file_id = file_id, area_id = area_id)
    rgb_labeled_mask_image_2d_overlay = _create_rgb_labeled_2d_overlay_of_mask_and_image(mask_zstack = mask_zstack, image_zstack = image_zstack)
    cropping_boundaries = _calculate_cropping_boundaries(center_pixel_coords = center_pixel_coords, half_box_size = half_box_size, image_shape = mask_zstack[0].shape[:2])
    cropped_and_binned_mask_zstack = _crop_and_bin_zstack(zstack = mask_zstack, cropping_boundaries = cropping_boundaries, binning_factor = binning_factor)
    voxel_color_code = label2rgb(cropped_and_binned_mask_zstack)
    _create_3d_plot(fmc_project = fmc_project,
                    file_id = file_id,
                    area_id = area_id,
                    binning_factor = binning_factor,
                    overview_image = rgb_labeled_mask_image_2d_overlay,
                    area_roi_coordinates = area_roi_coordinates,
                    box_boundaries = cropping_boundaries,
                    voxels = cropped_and_binned_mask_zstack,
                    color_code = voxel_color_code,
                    save = save,
                    show = show)


def _load_data(fmc_project: Project, file_id: str, area_id: str) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    image_zstack = load_zstack_as_array_from_single_planes(path = fmc_project.database.preprocessed_images_dir, file_id = file_id)
    mask_zstack = load_zstack_as_array_from_single_planes(path = fmc_project.database.quantified_segmentations_dir.joinpath(area_id), file_id = file_id)
    area_roi_coordinates = fmc_project.database.area_rois_for_quantification[file_id]['all_planes'][area_id].boundary.coords.xy
    return image_zstack, mask_zstack, area_roi_coordinates


def _create_rgb_labeled_2d_overlay_of_mask_and_image(mask_zstack: np.ndarray, image_zstack: np.ndarray) -> np.ndarray:
    max_projection_mask = np.max(mask_zstack, axis=0)
    max_projection_image = np.max(image_zstack, axis=0)
    return label2rgb(max_projection_mask, image=max_projection_image, bg_label = 0, bg_color = None, saturation=1, alpha=1)
    

def _calculate_cropping_boundaries(center_pixel_coords: Tuple[int, int], half_box_size: int, image_shape: Tuple[int, int]) -> Dict:
    center_pixel_row, center_pixel_col = center_pixel_coords
    max_row, max_col = image_shape
    lower_row_cropping_index, upper_row_cropping_index = _determine_cropping_indices(center_pixel_index = center_pixel_row, half_box_size = half_box_size, max_index = max_row)
    lower_col_cropping_index, upper_col_cropping_index = _determine_cropping_indices(center_pixel_index = center_pixel_col, half_box_size = half_box_size, max_index = max_col)
    cropping_boundaries = {'lower_row': lower_row_cropping_index,
                           'upper_row': upper_row_cropping_index,
                           'lower_col': lower_col_cropping_index,
                           'upper_col': upper_col_cropping_index}
    return cropping_boundaries
    
    
def _determine_cropping_indices(center_pixel_index: int, half_box_size: int, max_index: int) -> Tuple[int, int]:
    if (center_pixel_index - half_box_size >= 0) & (center_pixel_index + half_box_size <= max_index):
        lower_cropping_index, upper_cropping_index = center_pixel_index - half_box_size,  center_pixel_index + half_box_size
    elif 2*half_box_size <= max_index:
        if center_pixel_index - half_box_size < 0:
            lower_cropping_index, upper_cropping_index = 0,  0 + 2*half_box_size
        else: # means: center_pixel_index + half_box_size > max_index
            lower_cropping_index, upper_cropping_index = max_index - 2*half_box_size, max_index
    else:
        raise ValueError(f'The desired box size (2 * "half_box_size" = {2*half_box_size}) is larger than one of the image axes ({max_index}). Please select a smaller "half_box_size"!')
    return lower_cropping_index, upper_cropping_index

    
def _crop_and_bin_zstack(zstack: np.ndarray, cropping_boundaries: Dict, binning_factor: int=0) -> np.ndarray:
    cropped_zstack = zstack[:, cropping_boundaries['lower_row']:cropping_boundaries['upper_row'], cropping_boundaries['lower_col']:cropping_boundaries['upper_col']].copy()
    if binning_factor > 0:
        cropped_and_binned_zstack = _bin_zstack(zstack_to_bin = cropped_zstack, binning_factor = binning_factor)
    else:
        cropped_and_binned_zstack = cropped_zstack
    return cropped_and_binned_zstack


def _bin_zstack(zstack_to_bin: np.ndarray, binning_factor: int) -> np.ndarray:
    zstack, new_shape = _adjust_zstack_for_binning_to_new_shape(input_zstack = zstack_to_bin, binning_factor = binning_factor)
    binned_single_planes = []
    for plane_index in range(zstack.shape[0]):
        binned_single_planes.append(_bin_2d_image(single_plane = zstack[plane_index], new_shape = new_shape))
    return np.asarray(binned_single_planes)
        

def _adjust_zstack_for_binning_to_new_shape(input_zstack: np.ndarray, binning_factor: int) -> Tuple[np.ndarray, Tuple[int, int]]:
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
        error_message_line_1 = 'Sorry! Something went wrong during the binning process.\n'
        error_message_line_2 = 'To avoid this issue, consider changing the "half_box_size" or the "binning_factor", such that\n'
        error_message_line_3 = 'division of the entire box size (= 2*"half_box_size) by the binning factor does not yield any remainder, e.g.:\n'
        error_message_line_4 = 'half_box_size = 100, binning_factor = 4 --> entire box size = 2 * half_box_size = 200 --> 200 / 4 = 50 (no remainder!)'
        error_message = error_message_line_1 + error_message_line_2 + error_message_line_3 + error_message_line_4
        raise ValueError(error_message)
    else:
        new_shape = (int(adjusted_rows / binning_factor), int(adjusted_cols / binning_factor))
    return zstack, new_shape


def _bin_2d_image(single_plane: np.ndarray, new_shape: Tuple[int, int]):
    shape = (new_shape[0], single_plane.shape[0] // new_shape[0],
             new_shape[1], single_plane.shape[1] // new_shape[1])
    return single_plane.reshape(shape).max(-1).max(1)
    

def _create_3d_plot(fmc_project: Project, file_id: str, area_id: str, binning_factor: int, overview_image: np.ndarray, 
                    area_roi_coordinates: Tuple[np.ndarray, np.ndarray], box_boundaries: Dict, voxels: np.ndarray, 
                    color_code: np.ndarray, save: bool, show: bool) -> None:
    
    box_size = box_boundaries['upper_row'] - box_boundaries['lower_row']
    center_row_index = box_boundaries['lower_row'] + 0.5*box_size
    center_col_index = box_boundaries['lower_col'] + 0.5*box_size
    box_row_coords = [box_boundaries['lower_row'], box_boundaries['lower_row'], box_boundaries['upper_row'], box_boundaries['upper_row'], box_boundaries['lower_row']]
    box_col_coords = [box_boundaries['lower_col'], box_boundaries['upper_col'], box_boundaries['upper_col'], box_boundaries['lower_col'], box_boundaries['lower_col']]

    fig = plt.figure(figsize=(24,8), facecolor='white')
    gs = fig.add_gridspec(1,3)

    fig.add_subplot(gs[0,0])
    plt.imshow(overview_image)
    plt.plot(area_roi_coordinates[1], area_roi_coordinates[0])
    plt.plot(box_col_coords, box_row_coords, c='magenta', lw='3', linestyle='dashed')

    ax1 = fig.add_subplot(gs[0,1], projection='3d')
    ax1.voxels(voxels, facecolors=color_code)
    ax1.view_init(elev=30, azim=33)
    
    ax2 = fig.add_subplot(gs[0,2], projection='3d')
    ax2.voxels(voxels, facecolors=color_code)
    ax2.view_init(elev=30, azim=213)

    plt.suptitle(f'Area_id {area_id} of file_id {file_id}, centered at ({center_row_index}, {center_col_index}) with a binning of factor {binning_factor}', y=0.9)
    if save:
        plt.savefig(fmc_project.database.inspected_area_plots_dir.joinpath(f'{file_id}_3D_inspection_centered_at_{center_row_index}-{center_col_index}_and_{binning_factor}xbinning.png'), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()



# Code of previous implementation:
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
from skimage.io import imread, imsave
from shapely.geometry import Polygon
import cc3d

from typing import Dict, List, Tuple, Optional, Union

from .database import Database
from .utils import load_zstack_as_array_from_single_planes, get_polygon_from_instance_segmentation, get_cropping_box_arround_centroid
from .utils import get_color_code, get_rgb_color_code_for_3D, listdir_nohidden

class InspectionObject:
    
    def __init__(self, database: Database, file_id: str, area_roi_id: str, label_index: int, show: bool, save: bool) -> None:
        self.database = database
        self.file_id = file_id
        self.area_roi_id = area_roi_id
        self.show = show
        self.save = save
        self.zstack = self.load_postprocessed_segmentation()
        self.label_id = self.get_label_id(index = label_index)
        self.plane_id = self.get_plane_id()
        
    def load_postprocessed_segmentation(self) -> np.ndarray:
        path = self.database.quantified_segmentations_dir.joinpath(self.area_roi_id)
        return load_zstack_as_array_from_single_planes(path = path, file_id = self.file_id)
    
    
    def get_label_id(self, index: int) -> int:
        label_ids = list(np.unique(self.zstack))
        for background_id in [0, 0.0]:
            if background_id in label_ids:
                label_ids.remove(background_id)
        return label_ids[index]
    
    
    def get_plane_id(self) -> int:
        all_planes_with_label_id = list(np.where(self.zstack == self.label_id)[0])
        if len(all_planes_with_label_id) > 1:
            plane_id = all_planes_with_label_id[int(len(all_planes_with_label_id) / 2)]
        else:
            plane_id = all_planes_with_label_id[0]
        return plane_id
        

    def run_all_inspection_steps(self) -> None:
        if hasattr(self.database, 'inspection_strategies'):
            for inspection_strategy in self.database.inspection_strategies:
                inspection_strategy().run(inspection_object = self) 


class InspectionStrategy(ABC):
    
    @abstractmethod
    def run(self, inspection_object: InspectionObject):
        # create the inspection plot
        return 
    

                
class InspectReconstructedCells2D(InspectionStrategy):
    
    def run(self, inspection_object: InspectionObject):
        cminx, cmaxx, cminy, cmaxy = self.get_cropping_indices(inspection_object = inspection_object)
        cropped_zstack = inspection_object.zstack.copy()
        cropped_zstack = cropped_zstack[:, cminx:cmaxx, cminy:cmaxy]
        plotting_info = self.get_plotting_info(zstack = cropped_zstack)
        cropped_preprocessed_zstack = load_zstack_as_array_from_single_planes(path = inspection_object.database.preprocessed_images_dir, 
                                                                              file_id = inspection_object.file_id, 
                                                                              minx = cminx, 
                                                                              maxx = cmaxx, 
                                                                              miny = cminy, 
                                                                              maxy = cmaxy)
        cropped_instance_seg_zstack = load_zstack_as_array_from_single_planes(path = inspection_object.database.instance_segmentations_dir, 
                                                                              file_id = inspection_object.file_id, 
                                                                              minx = cminx, 
                                                                              maxx = cmaxx, 
                                                                              miny = cminy, 
                                                                              maxy = cmaxy)
        filepath = inspection_object.database.inspected_area_plots_dir.joinpath(f'{inspection_object.file_id}_{inspection_object.area_roi_id}_{inspection_object.label_id}_2D.png')
        if inspection_object.show:
            print(f'Plot to inspect segmentation of label #{inspection_object.label_id} in area roi id {inspection_object.area_roi_id} of file id #{inspection_object.file_id}:')
        self.plot_reconstructed_cells(preprocessed_zstack = cropped_preprocessed_zstack, 
                                      instance_seg_zstack = cropped_instance_seg_zstack, 
                                      final_labels_zstack = cropped_zstack, 
                                      plotting_info = plotting_info, 
                                      plane_id_of_interest = inspection_object.plane_id,
                                      filepath = filepath,
                                      save = inspection_object.save,
                                      show = inspection_object.show)
        
        
    def get_cropping_indices(self, inspection_object: InspectionObject) -> Tuple[int, int, int, int]:
        half_window_size = 200
        roi = get_polygon_from_instance_segmentation(single_plane = inspection_object.zstack[inspection_object.plane_id], label_id = inspection_object.label_id)
        centroid_x, centroid_y = round(roi.centroid.x), round(roi.centroid.y)
        max_x, max_y = inspection_object.zstack[inspection_object.plane_id].shape[0], inspection_object.zstack[inspection_object.plane_id].shape[1]
        cminx, cmaxx = self.adjust_cropping_box_to_image_borders(centroid_coord = centroid_x, max_value = max_x, half_window_size = half_window_size)
        cminy, cmaxy = self.adjust_cropping_box_to_image_borders(centroid_coord = centroid_y, max_value = max_y, half_window_size = half_window_size)
        return cminx, cmaxx, cminy, cmaxy
        
        
    def adjust_cropping_box_to_image_borders(self, centroid_coord: int, max_value: int, half_window_size: int) -> Tuple[int, int]:
        if (centroid_coord - half_window_size >= 0) & (centroid_coord + half_window_size <= max_value):
            lower_index = centroid_coord - half_window_size
            upper_index = centroid_coord + half_window_size
        elif (centroid_coord - half_window_size < 0) & (2*half_window_size <= max_value):
            lower_index = 0
            upper_index = 2*half_window_size
        elif (centroid_coord - 2*half_window_size >= 0) & (centroid_coord + half_window_size > max_value):
            lower_index = max_value - 2*half_window_size
            upper_index = max_value
        else:
            lower_index = 0
            upper_index = max_value
        return lower_index, upper_index
        
        
    def get_plotting_info(self, zstack: np.ndarray) -> Dict:
        label_ids = list(np.unique(zstack))
        if 0 in label_ids:
            label_ids.remove(0)
        if 0.0 in label_ids:
            label_ids.remove(0)
        color_code = get_color_code(label_ids)
        z_dim, x_dim, y_dim = zstack.shape
        plotting_info = dict()
        for plane_index in range(z_dim):
            plotting_info[plane_index] = dict()
        for label_id in label_ids:
            for plane_index in range(z_dim):
                if label_id in np.unique(zstack[plane_index]):
                    roi = get_polygon_from_instance_segmentation(zstack[plane_index], label_id) 
                    boundary_x_coords, boundary_y_coords = np.asarray(roi.boundary.xy[0]), np.asarray(roi.boundary.xy[1])
                    plotting_info[plane_index][label_id] = {'color': color_code[label_id],
                                                            'boundary_x_coords': boundary_x_coords,
                                                            'boundary_y_coords': boundary_y_coords} 
        return plotting_info            

        
    def plot_reconstructed_cells(self, preprocessed_zstack: np.ndarray, instance_seg_zstack: np.ndarray, 
                                 final_labels_zstack: np.ndarray, plotting_info: Dict, plane_id_of_interest: int, 
                                 filepath: Path, save: bool, show: bool) -> None:
        z_dim = final_labels_zstack.shape[0]
        fig = plt.figure(figsize=(15, 5*z_dim), facecolor='white')
        gs = fig.add_gridspec(z_dim, 3)

        for plane_index in range(z_dim):
            fig.add_subplot(gs[plane_index, 0])
            plt.imshow(preprocessed_zstack[plane_index])
            plt.ylabel(f'plane_{plane_index}', fontsize=14)
            if plane_index == 0:
                plt.title('input image', fontsize=14, pad=15)

        for plane_index in range(z_dim):
            fig.add_subplot(gs[plane_index, 1])
            plt.imshow(instance_seg_zstack[plane_index])
            if plane_index == 0:
                plt.title('instance segmentation', fontsize=14, pad=15)

        for plane_index in range(z_dim):
            fig.add_subplot(gs[plane_index, 2])
            plt.imshow(final_labels_zstack[plane_index], cmap = 'Greys_r')
            for label_id in plotting_info[plane_index].keys():
                plt.plot(plotting_info[plane_index][label_id]['boundary_y_coords'], 
                         plotting_info[plane_index][label_id]['boundary_x_coords'], 
                         c=plotting_info[plane_index][label_id]['color'], 
                         lw=3)
            if plane_index == plane_id_of_interest:
                plt.plot([185, 215], [200, 200], c='red', lw='3')
                plt.plot([200, 200], [185, 215], c='red', lw='3')
            if plane_index == 0:
                plt.title('connected components (color-coded)', fontsize=14, pad=15)

        if save:
            plt.savefig(filepath, dpi=300)
            print(f'The resulting plot was successfully saved to: {filepath}')
        if show:
            plt.show()
        else:
            plt.close()        

# even older:

class InspectionStrategy(ABC):
    
    @abstractmethod
    def run(self, database: Database, file_id: str) -> Union[plt.Axes, None]:
        # do something that might save a plot and/or return it for display
        pass



class InspectReconstructedCells2D(InspectionStrategy):
    
    def __init__(self, plane_id_of_interest: int, label_id_of_interest: int, zstack_with_label_id_of_interest: np.ndarray, save=False, show=True):
        self.plane_id_of_interest = plane_id_of_interest
        self.label_id_of_interest = label_id_of_interest
        self.zstack_with_label_id_of_interest = zstack_with_label_id_of_interest
        self.save = save
        self.show = show

    
    def get_plotting_info(self, zstack):
        label_ids = list(np.unique(zstack))
        if 0 in label_ids:
            label_ids.remove(0)
        color_code = get_color_code(label_ids)
        
        z_dim, x_dim, y_dim = zstack.shape
        plotting_info = dict()
        for plane_index in range(z_dim):
            plotting_info[plane_index] = dict()

        for label_id in label_ids:
            for plane_index in range(z_dim):
                if label_id in np.unique(zstack[plane_index]):
                    roi = get_polygon_from_instance_segmentation(zstack[plane_index], label_id) 
                    boundary_x_coords, boundary_y_coords = np.asarray(roi.boundary.xy[0]), np.asarray(roi.boundary.xy[1])
                    plotting_info[plane_index][label_id] = {'color': color_code[label_id],
                                                            'boundary_x_coords': boundary_x_coords,
                                                            'boundary_y_coords': boundary_y_coords} 
        return plotting_info
    

    def plot_reconstructed_cells(self, preprocessed_zstack, instance_seg_zstack, final_labels_zstack, plotting_info, plane_id_of_interest, save=False, show=True):
        z_dim = final_labels_zstack.shape[0]
        fig = plt.figure(figsize=(15, 5*z_dim), facecolor='white')
        gs = fig.add_gridspec(z_dim, 3)

        for plane_index in range(z_dim):
            print(plane_index)
            fig.add_subplot(gs[plane_index, 0])
            plt.imshow(preprocessed_zstack[plane_index])
            plt.ylabel(f'plane_{plane_index}', fontsize=14)
            if plane_index == 0:
                plt.title('input image', fontsize=14, pad=15)

        for plane_index in range(z_dim):
            fig.add_subplot(gs[plane_index, 1])
            plt.imshow(instance_seg_zstack[plane_index])
            if plane_index == 0:
                plt.title('instance segmentation', fontsize=14, pad=15)

        for plane_index in range(z_dim):
            fig.add_subplot(gs[plane_index, 2])
            plt.imshow(final_labels_zstack[plane_index], cmap = 'Greys_r')
            for label_id in plotting_info[plane_index].keys():
                plt.plot(plotting_info[plane_index][label_id]['boundary_y_coords'], 
                         plotting_info[plane_index][label_id]['boundary_x_coords'], 
                         c=plotting_info[plane_index][label_id]['color'], 
                         lw=3)
            if plane_index == plane_id_of_interest:
                plt.plot([185, 215], [200, 200], c='red', lw='3')
                plt.plot([200, 200], [185, 215], c='red', lw='3')
            if plane_index == 0:
                plt.title('connected components (color-coded)', fontsize=14, pad=15)

        if save:
            filepath = f'{self.database.inspected_area_plots_dir}{self.file_id}_{self.plane_id_of_interest}_{self.label_id_of_interest}_2D.png'
            plt.savefig(filepath, dpi=300)
            print(f'The resulting plot was successfully saved to: {self.database.inspected_area_plots_dir}')
        if show:
            plt.show()
        else:
            plt.close()

    
    def run(self, database: Database, file_id: str) -> Union[plt.Axes, None]:
        self.database = database
        self.file_id = file_id
        
        roi = get_polygon_from_instance_segmentation(self.zstack_with_label_id_of_interest[self.plane_id_of_interest], self.label_id_of_interest)
        cminx, cmaxx, cminy, cmaxy = get_cropping_box_arround_centroid(roi, 200)

        cropped_new_zstack = self.zstack_with_label_id_of_interest.copy()
        cropped_new_zstack = cropped_new_zstack[:, cminx:cmaxx, cminy:cmaxy]

        plotting_info = self.get_plotting_info(cropped_new_zstack)
        print(cminx, cmaxx, cminy, cmaxy)
        cropped_preprocessed_zstack = load_zstack_as_array_from_single_planes(path = database.preprocessed_images_dir, 
                                                                              file_id = file_id, 
                                                                              minx = cminx, 
                                                                              maxx = cmaxx, 
                                                                              miny = cminy, 
                                                                              maxy = cmaxy)

        cropped_instance_seg_zstack = load_zstack_as_array_from_single_planes(path = database.instance_segmentations_dir, 
                                                                              file_id = file_id, 
                                                                              minx = cminx, 
                                                                              maxx = cmaxx, 
                                                                              miny = cminy, 
                                                                              maxy = cmaxy)

        self.plot_reconstructed_cells(preprocessed_zstack = cropped_preprocessed_zstack, 
                                 instance_seg_zstack = cropped_instance_seg_zstack, 
                                 final_labels_zstack = cropped_new_zstack, 
                                 plotting_info = plotting_info, 
                                 plane_id_of_interest = self.plane_id_of_interest,
                                 save = self.save,
                                 show = self.show)



class InspectReconstructedCells3D(InspectionStrategy):
    
    def __init__(self, plane_id_of_interest: int, label_id_of_interest: int, zstack_with_label_id_of_interest: np.ndarray, save: bool=False, show: bool=True):
        self.plane_id_of_interest = plane_id_of_interest
        self.label_id_of_interest = label_id_of_interest
        self.zstack_with_label_id_of_interest = zstack_with_label_id_of_interest
        self.save = save
        self.show = show


    def plot_reconstructed_cells_in_3D(self, final_labels_zstack: np.ndarray, color_code: Dict, save: bool=False, show: bool=True):
        fig = plt.figure(figsize=(15, 15), facecolor='white')
        ax = fig.add_subplot(projection='3d')
        ax.voxels(final_labels_zstack, facecolors=color_code)
        ax.set(xlabel='single planes of z-stack', ylabel='x-dimension', zlabel='y-dimension')
        if save:
            filepath = f'{self.database.inspected_area_plots_dir}{self.file_id}_{self.label_id_of_interest}_3D.png'
            plt.savefig(filepath, dpi=300)
            print(f'The resulting plot was successfully saved to: {self.database.inspected_area_plots_dir}')
        if show:
            plt.show()
        else:
            plt.close()


    def run(self, database: Database, file_id: str):
        self.database = database
        self.file_id = file_id

        roi = get_polygon_from_instance_segmentation(self.zstack_with_label_id_of_interest[self.plane_id_of_interest], self.label_id_of_interest)
        cminx, cmaxx, cminy, cmaxy = get_cropping_box_arround_centroid(roi, 100)

        cropped_new_zstack = self.zstack_with_label_id_of_interest.copy()
        cropped_new_zstack = cropped_new_zstack[:, cminx:cmaxx, cminy:cmaxy]

        rgb_color_code = get_rgb_color_code_for_3D(zstack = cropped_new_zstack)

        self.plot_reconstructed_cells_in_3D(final_labels_zstack = cropped_new_zstack, 
                                            color_code = rgb_color_code, 
                                            save = self.save,
                                            show = self.show)



class InspectUsingMultiMatchIDX(InspectionStrategy):
    
    def __init__(self, multi_match_index: int, reconstruction_strategy: str='2D', save: bool=False, show: bool=True):
        self.multi_match_index = multi_match_index
        self.save = save
        self.show = show
        self.reconstruction_strategy = reconstruction_strategy
    
    
    def run(self, database: Database, file_id: str):
        
        zstack_with_final_label_ids = load_zstack_as_array_from_single_planes(path = database.inspection_final_label_planes_dir, file_id = file_id)
        multi_matches_traceback = database.multi_matches_traceback[file_id]
        if self.reconstruction_strategy == '2D':
            reconstruction_obj = InspectReconstructedCells2D(plane_id_of_interest = multi_matches_traceback['plane_index'][self.multi_match_index], 
                                                             label_id_of_interest = multi_matches_traceback['final_label_id'][self.multi_match_index], 
                                                             zstack_with_label_id_of_interest = zstack_with_final_label_ids,
                                                             save = self.save, 
                                                             show = self.show)
        elif self.reconstruction_strategy == '3D':
            reconstruction_obj = InspectReconstructedCells3D(plane_id_of_interest = multi_matches_traceback['plane_index'][self.multi_match_index], 
                                                             label_id_of_interest = multi_matches_traceback['final_label_id'][self.multi_match_index], 
                                                             zstack_with_label_id_of_interest = zstack_with_final_label_ids,
                                                             save = self.save, 
                                                             show = self.show)
        else:
            raise InputError("reconstruction_strategy has be one of the following strings: ['2D', '3D']")
        reconstruction_obj.run(database, file_id)
"""