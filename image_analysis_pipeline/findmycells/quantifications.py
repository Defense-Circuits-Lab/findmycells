from abc import ABC, abstractmethod
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
from skimage import measure
from shapely.geometry import Polygon
import cc3d

from typing import Dict, List, Tuple, Optional


"""

What we need here:

    - Processing of the instance labels to reconstructed cells with final ids
    - Inspection of the results (on demand)
    - Based on these reconstructed cells --> quantification in image region of interest (here SN mask)
        - Either in entire structure
        - Or estimation in accordance to grid method
        - Either way: implementation of specific rules for quantification, like exclude / include depending on which borders are touched 

"""

# Functions of general use for quantification classes:
def load_zstack_as_array_from_single_planes(path, file_id, minx=None, maxx=None, miny=None, maxy=None):
    types = list(set([type(minx), type(maxx), type(miny), type(maxy)]))    
    if any([minx, maxx, miny, maxy]):
        if (len(types) == 1) & (types[0] == int):
            cropping = True
        else:
            raise TypeError("'minx', 'maxx', 'miny', and 'maxy' all have to be integers - or None if no cropping has to be done")
    else:
        cropping = False
    filenames = [filename for filename in os.listdir(path) if filename.startswith(file_id)]
    cropped_zstack = list()
    for single_plane_filename in filenames:
        tmp_image = imread(path + single_plane_filename)
        if cropping:
            tmp_image = tmp_image[minx:maxx, miny:maxy]
        cropped_zstack.append(tmp_image.copy())
        del tmp_image
    return np.asarray(cropped_zstack) 


def unpad_x_y_dims_in_2d_array(padded_2d_array, pad_width):
    return padded_2d_array[pad_width:padded_2d_array.shape[0]-pad_width, pad_width:padded_2d_array.shape[1]-pad_width]
    
    
def unpad_x_y_dims_in_3d_array(padded_3d_array, pad_width):
    return padded_3d_array[:, pad_width:padded_3d_array.shape[1]-pad_width, pad_width:padded_3d_array.shape[2]-pad_width]


def get_polygon_from_instance_segmentation(single_plane: np.ndarray, label_id: int) -> Polygon:
    x_dim, y_dim = single_plane.shape
    tmp_array = np.zeros((x_dim, y_dim), dtype='uint8')
    tmp_array[np.where(single_plane == label_id)] = 1
    tmp_contours = measure.find_contours(tmp_array, level = 0)[0]
    return Polygon(tmp_contours)




class QuantificationObject:
    # 2D alternative still to come..
    
    def __inif__(self, file_id: int, database: Database):
        self.file_id = file_id
        self.database = database
        if self.database.quantification_configs['segmentations_to_use'] == 'binary':
            path = self.database.binary_segmentations_dir
        elif self.database.quantification_configs['segmentations_to_use'] == 'instance':
            path = self.database.instance_segmentations_dir
        self.segmentation_mask_path = path
        self.zstack_original_label_ids = load_zstack_as_array_from_single_planes(path = self.segmentation_mask_path, file_id = self.file_id)
        self.rois_to_analyze = self.database.rois_as_shapely_polygons[self.file_id]




class QuantificationPreprocessingStrategy(ABC):
    
    @abstractmethod
    def preprocess(self, quantification_obj: QuantificationObject, database: Database) -> Tuple[QuantificationObject, Database]:
        # do something
        return quantification_obj, database
    

class SomeCropping(QuantificationPreprocessingStrategy):
    
    def preprocess(self, quantification_obj: QuantificationObject, database: Database) -> Tuple[QuantificationObject, Database]:
        # crop the images to decrease computing time
        return quantification_obj, database
    
    
class RemoveUneccessaryLabels(QuantificationPreprocessingStrategy):
    
    def preprocess(self, quantification_obj: QuantificationObject, database: Database) -> Tuple[QuantificationObject, Database]:
        # "delete" all labels that are not needed -> replace with 0s
        return quantification_obj, database

    
class ReconstructCellsIn3DFrom2DInstanceLabels(QuantificationPreprocessingStrategy):
    pad_width = 1
    lowest_final_label_id = 2000

    def preprocess(self, quantification_obj: QuantificationObject, database: Database) -> Tuple[QuantificationObject, Database]:
        
        self.quantification_obj = quantification_obj
        self.database = database
        self.zstack_original_label_ids = np.pad(self.quantification_obj.zstack_original_label_ids, 
                                                pad_width = self.pad_width, 
                                                mode = 'constant', 
                                                constant_values = 0)
        self.zstack_original_label_ids = self.zstack_original_label_ids[self.pad_width : self.zstack_original_label_ids.shape[0] - self.pad_width]
        self.results = self.get_plane_to_plane_roi_matching_results(zstack = self.zstack_original_label_ids)
        for plane_id in range(self.zstack_original_label_ids.shape[0]):
            for label_id in self.results[plane_id].keys():
                self.results[plane_id][label_id] = self.find_best_matches(self.results[plane_id][label_id])
        self.final_ids, self.results = self.get_final_id_assignments(results = self.results, lowest_final_label_id = self.lowest_final_label_id)
        self.zstack_original_label_ids = unpad_x_y_dims_in_3d_array(self.zstack_original_label_ids, self.pad_width)
        self.zstack_with_final_label_ids = self.set_new_label_ids(zstack_with_old_label_ids = self.zstack_original_label_ids, 
                                                                  new_ids_assignment = self.final_ids)
        setattr(self.quantification_obj, 'zstack_with_final_label_ids', self.zstack_with_final_label_ids)
        
        multi_matches_traceback = self.get_rois_with_multiple_matches(results = self.results)
        if hasattr(self.database, 'multi_matches_traceback') == False:
            setattr(self.database, 'multi_matches_traceback', dict())
        self.database.multi_matches_traceback[self.quantification_obj.file_id] = multi_matches_traceback
        
        return self.quantification_obj, self.database

    
    def roi_matching(self, original_roi: Polygon, roi_to_compare: Polygon, label_id_roi_to_compare: int, results: Dict, plane_indicator: str) -> Dict:

            iou = original_roi.intersection(roi_to_compare).area / original_roi.union(roi_to_compare).area
            proportion = original_roi.intersection(roi_to_compare).area / original_roi.area

            if original_roi.within(roi_to_compare) or roi_to_compare.within(original_roi): within = True
            else: within = False

            results[f'matching_ids_{plane_indicator}_plane'].append(label_id_roi_to_compare)
            results[f'full_overlap_{plane_indicator}_plane'].append(within)
            results[f'overlapping_area_{plane_indicator}_plane'].append(proportion)
            results[f'IoUs_{plane_indicator}_plane'].append(iou)

            return results
    
    
    def get_plane_to_plane_roi_matching_results(self, zstack, verbose=True) -> Dict:
        z_dim, x_dim, y_dim = zstack.shape
        results = dict()

        for plane_idx in range(z_dim):
            if verbose:
                print(f'starting with plane {plane_idx}')
            results[plane_idx] = dict()

            if plane_idx == 0:
                previous_plane_info = (None, 'previous')
                next_plane_info = (plane_idx + 1, 'next')
            elif plane_idx == z_dim - 1:
                previous_plane_info = (plane_idx - 1, 'previous')
                next_plane_info = (None, 'next')
            else:
                previous_plane_info = (plane_idx - 1, 'previous')
                next_plane_info = (plane_idx + 1, 'next')

            plane = zstack[plane_idx]
            unique_label_ids = list(np.unique(plane))
            if 0 in unique_label_ids:
                unique_label_ids.remove(0)
            elif 0.0 in unique_label_ids:
                unique_label_ids.remove(0.0)

            for label_id in unique_label_ids:
                roi = get_polygon_from_instance_segmentation(single_plane = zstack[plane_idx], label_id = label_id)
                roi_area = roi.area
            
                results[plane_idx][label_id] = {'final_label_id_assigned': False,
                                                'final_label_id': None,
                                                'area': roi_area,
                                                'matching_ids_previous_plane': list(),
                                                'full_overlap_previous_plane': list(),
                                                'overlapping_area_previous_plane': list(),
                                                'IoUs_previous_plane': list(),
                                                'matching_ids_next_plane': list(),
                                                'full_overlap_next_plane': list(),
                                                'overlapping_area_next_plane': list(),
                                                'IoUs_next_plane': list(),
                                                'best_match_previous_plane': None,
                                                'overlapping_area_best_match_previous_plane': None,
                                                'IoU_best_match_previous_plane': None,
                                                'best_match_next_plane': None,
                                                'overlapping_area_best_match_next_plane': None,
                                                'IoU_best_match_next_plane': None}
            
                for plane_to_compare_info in [previous_plane_info, next_plane_info]:
                    if plane_to_compare_info[0] != None:
                        labels_of_pixels_in_plane_to_compare = None # Reset results - still required??
                        plane_to_compare_idx, plane_indicator = plane_to_compare_info[0], plane_to_compare_info[1]
                        labels_of_pixels_in_plane_to_compare = zstack[plane_to_compare_idx][np.where(plane == label_id)]
                        labels_of_pixels_in_plane_to_compare = list(np.unique(labels_of_pixels_in_plane_to_compare))
                        if 0 in labels_of_pixels_in_plane_to_compare:
                            labels_of_pixels_in_plane_to_compare.remove(0)
                        elif 0.0 in labels_of_pixels_in_plane_to_compare:
                            labels_of_pixels_in_plane_to_compare.remove(0.0)

                        for label_id_in_plane_to_compare in labels_of_pixels_in_plane_to_compare:
                            roi_to_compare = get_polygon_from_instance_segmentation(single_plane = zstack[plane_to_compare_idx], label_id = label_id_in_plane_to_compare)
                            results[plane_idx][label_id] = self.roi_matching(original_roi = roi, 
                                                                            roi_to_compare = roi_to_compare, 
                                                                            label_id_roi_to_compare =  label_id_in_plane_to_compare, 
                                                                            results = results[plane_idx][label_id], 
                                                                            plane_indicator = plane_indicator)
        return results    


    def find_best_matches(self, results: Dict) -> Dict:

        for plane_indicator in ['previous', 'next']:
            if len(results[f'matching_ids_{plane_indicator}_plane']) > 0:
                max_iou = max(results[f'IoUs_{plane_indicator}_plane'])
                if max_iou >= 0.5: # Does this really make sense here? IoU could also be < 0.5 and within == False, but only because of some pixel? Max reciprocal overlap??
                    index = results[f'IoUs_{plane_indicator}_plane'].index(max_iou)
                elif any(results[f'full_overlap_{plane_indicator}_plane']):
                    index = results[f'full_overlap_{plane_indicator}_plane'].index(True)
                else:
                    index = None

                if type(index) == int:
                    best_matching_id = results[f'matching_ids_{plane_indicator}_plane'][index]
                    iou = max_iou
                    overlap = results[f'overlapping_area_{plane_indicator}_plane'][index]
                else: 
                    best_matching_id, iou, overlap = None, None, None

                results[f'best_match_{plane_indicator}_plane'] = best_matching_id
                results[f'overlapping_area_best_match_{plane_indicator}_plane'] = overlap
                results[f'IoU_best_match_{plane_indicator}_plane'] = iou

        return results
    

    def trace_matches(self, matching_results: Dict, final_ids_assignment: Dict, current_final_id: int) -> Tuple[Dict, Dict, bool]:
        current_plane_idx = final_ids_assignment[current_final_id]['plane_index'][-1]
        current_plane_label_id = final_ids_assignment[current_final_id]['original_label_id'][-1]
        best_match_next_plane = matching_results[current_plane_idx][current_plane_label_id]['best_match_next_plane']
        next_plane_idx = current_plane_idx + 1

        if matching_results[next_plane_idx][best_match_next_plane]['final_label_id_assigned']:
            raise ValueError(f'ROI with ID {best_match_next_plane} in plane {next_plane_idx} was already assigned! :o')
        else:
            if matching_results[next_plane_idx][best_match_next_plane]['best_match_previous_plane'] != current_plane_label_id:
                raise ValueError(f'ROI with ID {best_match_next_plane} in plane {next_plane_idx} does not share best matching with previous plane!')
            else:
                matching_results[next_plane_idx][best_match_next_plane]['final_label_id_assigned'] = True
                matching_results[next_plane_idx][best_match_next_plane]['final_label_id'] = current_final_id
                final_ids_assignment[current_final_id]['plane_index'].append(next_plane_idx)
                final_ids_assignment[current_final_id]['original_label_id'].append(best_match_next_plane)

                if matching_results[next_plane_idx][best_match_next_plane]['best_match_next_plane'] != None:
                    keep_tracing = True

                else:
                    keep_tracing = False

        return matching_results, final_ids_assignment, keep_tracing

    
    def get_final_id_assignments(self, results: Dict, lowest_final_label_id = 2000) -> Tuple[Dict, Dict]:
        final_ids = dict()
        keep_going = True
        final_label_id = lowest_final_label_id

        for plane_idx in results.keys():
            for label_id in results[plane_idx].keys():
                if results[plane_idx][label_id]['final_label_id_assigned']:
                    continue
                else:
                    final_ids[final_label_id] = {'plane_index': list(),
                                                 'original_label_id': list()}

                    results[plane_idx][label_id]['final_label_id_assigned'] = True
                    results[plane_idx][label_id]['final_label_id'] = final_label_id
                    final_ids[final_label_id]['plane_index'].append(plane_idx)
                    final_ids[final_label_id]['original_label_id'].append(label_id)

                    # Now start tracing:
                    if results[plane_idx][label_id]['best_match_next_plane'] != None:
                        keep_tracing = True
                        while keep_tracing:
                            results, final_ids, keep_tracing = self.trace_matches(matching_results = results, 
                                                                                 final_ids_assignment = final_ids, 
                                                                                 current_final_id = final_label_id)
                    final_label_id += 1
        return final_ids, results
    
    
    def set_new_label_ids(self, zstack_with_old_label_ids: np.ndarray, new_ids_assignment: Dict) -> np.ndarray:
        zstack_with_new_label_ids = zstack_with_old_label_ids.copy()
        for new_label_id in new_ids_assignment.keys():
            for idx in range(len(new_ids_assignment[new_label_id]['plane_index'])):
                plane_index = new_ids_assignment[new_label_id]['plane_index'][idx]
                old_label_id = new_ids_assignment[new_label_id]['original_label_id'][idx]
                zstack_with_new_label_ids[plane_index][np.where(zstack_with_new_label_ids[plane_index] == old_label_id)] = new_label_id
        return zstack_with_new_label_ids
    
    
    def get_rois_with_multiple_matches(self, results: Dict) -> List:

        multi_matches_traceback = list()

        for plane_idx in results.keys():
            for label_id in results[plane_idx].keys():
                condition_a = len(results[plane_idx][label_id]['matching_ids_next_plane']) > 1
                condition_b = len(results[plane_idx][label_id]['matching_ids_previous_plane']) > 1
                if condition_a or condition_b:
                    multi_matches_traceback.append((plane_idx, label_id))

        return multi_matches_traceback
    


class QuantificationStrategy(ABC):
    
    @abstractmethod
    def quantify(self, quantification_obj: QuantificationObject, database: Database) -> Database:
        # do something
        return database
    
    
class CountCellsInWholeStructure(QuantificationStrategy):
    
    def quantify(self, quantification_obj: QuantificationObject, database: Database) -> Database:
        # quantify cells in the entire image region of interest (could also be multiple)
        return database  


class EstimateFromGridMethod(QuantificationStrategy):
    
    def quantify(self, quantification_obj: QuantificationObject, database: Database) -> Database:
        # Use the grid method and quantify cells only in some subregions of the image region of interest and estimate count in whole structure from there
        return database  





class Quantifier:
    
    def __init__(self, database: Database, file_ids):
        self.database = database
        self.file_ids = file_ids
        self.quantification_preprocessing_strategies = self.database.quantification_configs['quantification_preprocessing_strategies']
        self.quantification_strategy = self.database.quantification_configs['quantification_strategy']
        
        
        # Assign something like quantification_configs?
        
    def run_all(self) -> Database:
        
        for file_id in self.file_ids:
            quantification_obj = QuantificationObject(file_id, self.database)
            
            for quant_prepro_strat in self.quantification_preprocessing_strategies:
                quantification_obj, self.database = quant_prepro_strat.preprocesses(quantification_obj, self.database)
                        
            self.database = self.quantification_strategy.quantify(quantification_obj, self.database)
            
            del quantification_obj
            
        return self.database
        
        
            
            

