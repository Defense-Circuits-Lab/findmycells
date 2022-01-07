from abc import ABC, abstractmethod
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
from skimage.io import imread, imsave
from shapely.geometry import Polygon
import cc3d

from typing import Dict, List, Tuple, Optional, Union

from .database import Database
from .utils import load_zstack_as_array_from_single_planes, get_polygon_from_instance_segmentation


class InspectionStrategy(ABC):
    
    @abstractmethod
    def run(self, database: Database, file_id: str) -> Union[plt.Axes, None]:
        # do something that might save a plot and/or return it for display
        pass
       
    
class InspectReconstructedCells(InspectionStrategy):

# the 'results' and the 'file_ids' dictionaries have to become attributes of the object
# also the corresponding z-stacks (both unpadded again): 
#   - the 'zstack_original_label_ids' with the original instance label ids (the one we are looking for)
#   - the 'zstack_with_final_label_ids' with the final label ids
# likewise, also the directory paths have to be accessible via the database (just as the current file_id)
    
    def __init__(self, plane_id_of_interest: int, label_id_of_interest: int, zstack_with_label_id_of_interest: np.ndarray, save=False, show=True):
        self.plane_id_of_interest = plane_id_of_interest
        self.label_id_of_interest = label_id_of_interest
        self.zstack_with_label_id_of_interest = zstack_with_label_id_of_interest
        self.save = save
        self.show = show
    
    
    def get_cropping_box_arround_centroid(self, roi: Polygon, half_window_size: int) -> Tuple[int, int, int, int]:
        centroid_x, centroid_y = round(roi.centroid.x), round(roi.centroid.y)
        cminx, cmaxx = centroid_x - half_window_size, centroid_x + half_window_size
        cminy, cmaxy = centroid_y - half_window_size, centroid_y + half_window_size
        return cminx, cmaxx, cminy, cmaxy
    
    
    def get_color_code(self, label_ids):
        n_label_ids = len(label_ids)
        colormixer = plt.cm.rainbow(np.linspace(0, 1, n_label_ids))

        color_code = dict()
        for idx in range(n_label_ids):
            color_code[label_ids[idx]] = colormixer[idx]

        return color_code   
    
    
    
    def get_plotting_info(self, zstack):
        label_ids = list(np.unique(zstack))
        if 0 in label_ids:
            label_ids.remove(0)
        color_code = self.get_color_code(label_ids)

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
            filepath = f'{self.database.inspection_dir}{self.file_id}_label-{self.label_id_of_interest}_plane-{self.plane_id_of_interest}_inspected_area.png'
            plt.savefig(filepath, dpi=300)
            print(f'The resulting plot was successfully saved to: {self.database.inspection_dir}')
        if show:
            plt.show()
        else:
            plt.close()

    
    
    def run(self, database: Database, file_id: str) -> Union[plt.Axes, None]:
        self.database = database
        self.file_id = file_id
        
        roi = get_polygon_from_instance_segmentation(self.zstack_with_label_id_of_interest[self.plane_id_of_interest], self.label_id_of_interest)
        cminx, cmaxx, cminy, cmaxy = self.get_cropping_box_arround_centroid(roi, 200)

        cropped_new_zstack = self.zstack_with_label_id_of_interest.copy()
        cropped_new_zstack = cropped_new_zstack[:, cminx:cmaxx, cminy:cmaxy]

        plotting_info = self.get_plotting_info(cropped_new_zstack)

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
        
        
class InspectReconstructedCellsBasedOnMultiMatchIDX(InspectionStrategy):
    
    def __init__(self, multi_match_index: int, save: bool=False, show: bool=True):
        self.multi_match_index = multi_match_index
        self.save = save
        self.show = show
    
    
    def run(self, database: Database, file_id: str) -> Union[plt.Axes, None]:
        
        zstack_with_final_label_ids = load_zstack_as_array_from_single_planes(path = database.inspection_dir, file_id = file_id)
        
        multi_matches_traceback = database.multi_matches_traceback[file_id]
        
        helper_obj = InspectReconstructedCells(plane_id_of_interest = multi_matches_traceback['plane_index'][self.multi_match_index], 
                                                label_id_of_interest = multi_matches_traceback['final_label_id'][self.multi_match_index], 
                                                zstack_with_label_id_of_interest = zstack_with_final_label_ids,
                                                save = self.save, 
                                                show = self.show)
        helper_obj.run(database, file_id)