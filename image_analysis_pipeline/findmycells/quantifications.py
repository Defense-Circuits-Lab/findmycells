from abc import ABC, abstractmethod
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
from skimage.io import imsave, imread
from shapely.geometry import Polygon
import cc3d

from typing import Dict, List, Tuple, Optional

from .database import Database
from .utils import load_zstack_as_array_from_single_planes, unpad_x_y_dims_in_2d_array, unpad_x_y_dims_in_3d_array, get_polygon_from_instance_segmentation
from .utils import get_rgb_color_code_for_3D

"""

What we need here:

    - Processing of the instance labels to reconstructed cells with final ids
    - Inspection of the results (on demand)
    - Based on these reconstructed cells --> quantification in image region of interest (here SN mask)
        - Either in entire structure
        - Or estimation in accordance to grid method
        - Either way: implementation of specific rules for quantification, like exclude / include depending on which borders are touched 

"""


class QuantificationObject:
    # 2D alternative still to come..
    
    def __init__(self, file_id: int, database: Database):
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
        
        zstack_original_label_ids = np.pad(quantification_obj.zstack_original_label_ids, 
                                            pad_width = self.pad_width, 
                                            mode = 'constant', 
                                            constant_values = 0)
        zstack_original_label_ids = zstack_original_label_ids[self.pad_width : zstack_original_label_ids.shape[0] - self.pad_width]
        results = self.get_plane_to_plane_roi_matching_results(zstack = zstack_original_label_ids)
        for plane_id in range(zstack_original_label_ids.shape[0]):
            for label_id in results[plane_id].keys():
                results[plane_id][label_id] = self.find_best_matches(all_results = results, original_plane_id = plane_id, original_label_id = label_id)
        final_ids, results = self.get_final_id_assignments(results = results, lowest_final_label_id = self.lowest_final_label_id)
        zstack_original_label_ids = unpad_x_y_dims_in_3d_array(zstack_original_label_ids, self.pad_width)
        zstack_with_final_label_ids = self.set_new_label_ids(zstack_with_old_label_ids = zstack_original_label_ids, 
                                                                  new_ids_assignment = final_ids)
        setattr(quantification_obj, 'zstack_with_final_label_ids', zstack_with_final_label_ids)
        setattr(quantification_obj, 'final_ids', final_ids)
        
        for plane_index in range(zstack_with_final_label_ids.shape[0]):
            filepath = f'{database.inspection_final_label_planes_dir}{quantification_obj.file_id}_{str(plane_index).zfill(3)}_final_label_ids.png'
            imsave(filepath, zstack_with_final_label_ids[plane_index], check_contrast=False)        
        
        multi_matches_traceback = self.get_rois_with_multiple_matches(results)
        if hasattr(database, 'multi_matches_traceback') == False:
            setattr(database, 'multi_matches_traceback', dict())
        database.multi_matches_traceback[quantification_obj.file_id] = multi_matches_traceback
                
        return quantification_obj, database

    
    def roi_matching(self, original_roi: Polygon, roi_to_compare: Polygon, label_id_roi_to_compare: int, results: Dict, plane_indicator: str) -> Dict:

        iou = original_roi.intersection(roi_to_compare).area / original_roi.union(roi_to_compare).area
        proportion = original_roi.intersection(roi_to_compare).area / original_roi.area

        #if original_roi.within(roi_to_compare) or roi_to_compare.within(original_roi): within = True
        if original_roi.within(roi_to_compare): within = True
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
                print(f'--matching ROIs across planes ({plane_idx + 1}/{z_dim})')
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


    def find_best_matches(self, all_results: Dict, original_plane_id: int, original_label_id: int) -> Dict:
        
        results = all_results[original_plane_id][original_label_id].copy()
        for plane_indicator in ['previous', 'next']:
            if len(results[f'matching_ids_{plane_indicator}_plane']) > 0:
                if plane_indicator == 'previous':
                    reciprocal_plane_indicator = 'next'
                    reciprocal_plane_id = original_plane_id - 1
                else:
                    reciprocal_plane_indicator = 'previous'
                    reciprocal_plane_id = original_plane_id + 1
                
                
                within = any(results[f'full_overlap_{plane_indicator}_plane'])
                # First exit: our ROI is fully within another ROI, obviously making it our best match:
                if within:
                    index_of_reciprocal_label_id = results[f'full_overlap_{plane_indicator}_plane'].index(True)
                    reciprocal_label_id = results[f'matching_ids_{plane_indicator}_plane'][index_of_reciprocal_label_id]
                    reciprocal_results = all_results[reciprocal_plane_id][reciprocal_label_id].copy()
                    other_label_ids_within_reciprocal_label_id = list()
                    for label_id_matching_with_reciprocal_label_id in reciprocal_results[f'matching_ids_{reciprocal_plane_indicator}_plane']:
                        tmp_results = all_results[original_plane_id][label_id_matching_with_reciprocal_label_id].copy()
                        index_reciprocal_label_id = tmp_results[f'matching_ids_{plane_indicator}_plane'].index(reciprocal_label_id)
                        if tmp_results[f'full_overlap_{plane_indicator}_plane'][index_reciprocal_label_id]:
                            iou_other_label_id_to_reciprocal_label_id = tmp_results[f'IoUs_{plane_indicator}_plane'][index_reciprocal_label_id]
                            other_label_ids_within_reciprocal_label_id.append((label_id_matching_with_reciprocal_label_id, iou_other_label_id_to_reciprocal_label_id))
                    index_reciprocal_label_id = results[f'matching_ids_{plane_indicator}_plane'].index(reciprocal_label_id)
                    max_iou = results[f'IoUs_{plane_indicator}_plane'][index_reciprocal_label_id]
                    best_matching_label_id = original_label_id
                    if len(other_label_ids_within_reciprocal_label_id) > 0:
                        for other_label_id, iou_other_label_id in other_label_ids_within_reciprocal_label_id:
                            if iou_other_label_id > max_iou:
                                best_matching_label_id = other_label_id
                            elif iou_other_label_id == max_iou:
                                best_matching_label_id = min([best_matching_label_id, other_label_id])
                    if best_matching_label_id == original_label_id:
                        best_match_index = results[f'full_overlap_{plane_indicator}_plane'].index(True)
                    else:
                        best_match_index = None
                else:
                    reciprocal_within_label_ids = list()
                    reciprocal_label_ids_to_be_excluded = list()
                    for reciprocal_label_id in results[f'matching_ids_{plane_indicator}_plane']:
                        reciprocal_results = all_results[reciprocal_plane_id][reciprocal_label_id].copy()
                        index_original_label_id = reciprocal_results[f'matching_ids_{reciprocal_plane_indicator}_plane'].index(original_label_id)
                        # Is the reciprocal roi within our original roi?
                        if reciprocal_results[f'full_overlap_{reciprocal_plane_indicator}_plane'][index_original_label_id]:
                            reciprocal_within_label_ids.append(reciprocal_label_id)
                        # Is the reciprocal roi within another roi [not our original roi]?
                        elif any(reciprocal_results[f'full_overlap_{reciprocal_plane_indicator}_plane']):
                            reciprocal_label_ids_to_be_excluded.append(reciprocal_label_id)
                        # Is there a third roi (a roi that overlaps with the reciprocal roi) - that is fully within the reciprocal roi?
                        else:
                            other_label_ids_within_reciprocal_label_id = list()
                            for label_id_matching_with_reciprocal_label_id in reciprocal_results[f'matching_ids_{reciprocal_plane_indicator}_plane']:
                                tmp_results = all_results[original_plane_id][label_id_matching_with_reciprocal_label_id].copy()
                                index_reciprocal_label_id = tmp_results[f'matching_ids_{plane_indicator}_plane'].index(reciprocal_label_id)
                                # right now within´s are always prioritized over overlapping potential best matches. 
                                # So if there is any ROI that has a full overlapt (=within) our potential best matching reciprocal ROI - this reciprocal ROI will be excluded
                                if tmp_results[f'full_overlap_{plane_indicator}_plane'][index_reciprocal_label_id]:
                                    reciprocal_label_ids_to_be_excluded.append(reciprocal_label_id)
                            
                    # Second exit: (at least) one matching ROI is fully within our original ROI and, therefore, has to be considered as best match:
                    if len(reciprocal_within_label_ids) > 0:
                        max_iou = -1
                        best_reciprocal_within_label_id = None
                        for reciprocal_label_id in reciprocal_within_label_ids:
                            index_reciprocal_label_id = results[f'matching_ids_{plane_indicator}_plane'].index(reciprocal_label_id)
                            tmp_matching_iou = results[f'IoUs_{plane_indicator}_plane'][index_reciprocal_label_id]
                            if tmp_matching_iou > max_iou:
                                best_reciprocal_within_label_id = reciprocal_label_id
                                max_iou = tmp_matching_iou
                            elif tmp_matching_iou == max_iou:
                                best_reciprocal_within_label_id = min([best_reciprocal_within_label_id, reciprocal_label_id])
                        best_match_index = results[f'matching_ids_{plane_indicator}_plane'].index(best_reciprocal_within_label_id) 
                    else: # neither an original within, nor a reciprocal within - time to look for the highest IOU then:
                        max_iou = max(results[f'IoUs_{plane_indicator}_plane'])
                        max_iou_index = results[f'IoUs_{plane_indicator}_plane'].index(max_iou)
                        reciprocal_label_id = results[f'matching_ids_{plane_indicator}_plane'][max_iou_index]
                        # Third exit: The best matching ROI of our original ROI is fully within another ROI within the same plane as our original ROI --> no match!
                        if reciprocal_label_id in reciprocal_label_ids_to_be_excluded:
                            best_match_index = None
                        else:
                            reciprocal_results = all_results[reciprocal_plane_id][reciprocal_label_id].copy()
                            reciprocal_max_iou = max(reciprocal_results[f'IoUs_{reciprocal_plane_indicator}_plane'])
                            reciprocal_max_iou_index = reciprocal_results[f'IoUs_{reciprocal_plane_indicator}_plane'].index(reciprocal_max_iou)
                            # Fourth exit: Our original ROI is also the best matching ROI for its´ own best matching ROI: 
                            if original_label_id == reciprocal_results[f'matching_ids_{reciprocal_plane_indicator}_plane'][reciprocal_max_iou_index]:
                                best_match_index = max_iou_index
                            # Fifth and final exit: Our original ROI is not the best matching ROI of its´ own best match --> no match!
                            else:
                                best_match_index = None                    
                if type(best_match_index) == int:
                    best_match_label_id = results[f'matching_ids_{plane_indicator}_plane'][best_match_index]
                    best_match_iou = results[f'IoUs_{plane_indicator}_plane'][best_match_index]
                    best_match_overlap = results[f'overlapping_area_{plane_indicator}_plane'][best_match_index]
                else: 
                    best_match_label_id, best_match_iou, best_match_overlap = None, None, None

                results[f'best_match_{plane_indicator}_plane'] = best_match_label_id
                results[f'overlapping_area_best_match_{plane_indicator}_plane'] = best_match_overlap
                results[f'IoU_best_match_{plane_indicator}_plane'] = best_match_iou
        
        return results
    

    def trace_matches(self, matching_results: Dict, final_ids_assignment: Dict, current_final_id: int) -> Tuple[Dict, Dict, bool]:
        current_plane_idx = final_ids_assignment[current_final_id]['plane_index'][-1]
        current_plane_label_id = final_ids_assignment[current_final_id]['original_label_id'][-1]
        best_match_next_plane = matching_results[current_plane_idx][current_plane_label_id]['best_match_next_plane']
        next_plane_idx = current_plane_idx + 1

        if matching_results[next_plane_idx][best_match_next_plane]['final_label_id_assigned']:
            print(f'Was trying to trace down the matches of original_label_id: {current_plane_label_id} in plane {current_plane_idx}')
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
    
    
    def get_rois_with_multiple_matches(self, results: Dict) -> Dict:

        multi_matches_traceback = {'final_label_id': list(),
                                   'original_instance_label_id': list(),
                                   'plane_index': list()}

        for plane_index in results.keys():
            for label_id in results[plane_index].keys():
                condition_a = len(results[plane_index][label_id]['matching_ids_next_plane']) > 1
                condition_b = len(results[plane_index][label_id]['matching_ids_previous_plane']) > 1
                if condition_a or condition_b:
                    multi_matches_traceback['final_label_id'].append(results[plane_index][label_id]['final_label_id'])
                    multi_matches_traceback['original_instance_label_id'].append(label_id)
                    multi_matches_traceback['plane_index'].append(plane_index)

        return multi_matches_traceback
    


class QuantificationStrategy(ABC):
    
    @abstractmethod
    def quantify(self, quantification_obj: QuantificationObject, database: Database) -> Database:
        # do something
        return database
    
    
class CountCellsInWholeStructure(QuantificationStrategy):
    
    def quantify(self, quantification_obj: QuantificationObject, database: Database) -> Database:
        print('--quantifying reconstructed cells in 3D within region of interest')
        self.database = database
        quantification_obj.final_ids = self.add_relative_positions_to_final_assignment(final_ids = quantification_obj.final_ids,
                                                                                       rois_of_areas_to_quantify = self.database.rois_as_shapely_polygons[quantification_obj.file_id],
                                                                                       zstack = quantification_obj.zstack_with_final_label_ids)
        zstacks_for_quantification = self.get_zstacks_for_quantification(final_ids = quantification_obj.final_ids, 
                                                                         zstack_without_exclusions = quantification_obj.zstack_with_final_label_ids)
        
        for roi_id in zstacks_for_quantification.keys():
            for plane_id in range(zstacks_for_quantification[roi_id]['zstack'].shape[0]):
                filepath = f'{self.database.inspection_planes_for_quantification}{quantification_obj.file_id}_roi-{roi_id}_quantified_plane_{str(plane_id).zfill(4)}.tif'
                imsave(filepath, zstacks_for_quantification[roi_id]['zstack'][plane_id], check_contrast = False)        
        
        zstacks_for_quantification = self.get_n_connected_components_in_zstacks(zstacks_for_quantification = zstacks_for_quantification)
        
        for roi_id in zstacks_for_quantification.keys():
            zstacks_for_quantification[roi_id].pop('zstack')
            
        if hasattr(self.database, 'quantification_results') == False:
            setattr(self.database, 'quantification_results', dict())
            
        self.database.quantification_results[quantification_obj.file_id] = zstacks_for_quantification
        
        return self.database  
    
    
    def get_relative_position(self, roi_to_check: Polygon, reference: Polygon) -> str:
        if roi_to_check.within(reference):
            rel_position = 'within'
        elif roi_to_check.intersects(reference):
            rel_position = 'intersects'
        elif roi_to_check.touches(reference):
            rel_position = 'touches'
        else:
            rel_position = 'no_overlap'
        return rel_position    
    
    
    def add_relative_positions_to_final_assignment(self, final_ids: Dict, rois_of_areas_to_quantify: Dict, zstack: np.ndarray) -> Dict:

        roi_ids = rois_of_areas_to_quantify.keys()

        for roi_id in roi_ids:
            roi_area_to_quantify = rois_of_areas_to_quantify[roi_id]

            for plane_index in range(zstack.shape[0]):
                label_ids = np.unique(zstack[plane_index])
                label_ids = np.delete(label_ids, [0])

                for label_id in label_ids:
                    roi = get_polygon_from_instance_segmentation(single_plane = zstack[plane_index], label_id = label_id)
                    relative_position = self.get_relative_position(roi, roi_area_to_quantify)

                    if 'relative_positions_per_roi_id' not in final_ids[label_id].keys():
                        final_ids[label_id]['relative_positions_per_roi_id'] = dict()
                        for roi_id_to_add in roi_ids:
                            final_ids[label_id]['relative_positions_per_roi_id'][roi_id_to_add] = {'plane_index': list(),
                                                                                                   'relative_position': list()}

                    final_ids[label_id]['relative_positions_per_roi_id'][roi_id]['plane_index'].append(plane_index)
                    final_ids[label_id]['relative_positions_per_roi_id'][roi_id]['relative_position'].append(relative_position)

        return final_ids

    
    def get_zstacks_for_quantification(self, final_ids: Dict, zstack_without_exclusions: np.ndarray) -> Dict:

        roi_ids = final_ids[list(final_ids.keys())[0]]['relative_positions_per_roi_id'].keys()
        zstacks_for_quantification = dict()

        for roi_id in roi_ids:
            zstacks_for_quantification[roi_id] = {'zstack': None,
                                                  'inclusion_criteria_position': ['within', 'intersects'],
                                                  'inclusion_criteria_min_z_extension': 2,
                                                  'number_connected_components': None}

            zstacks_for_quantification[roi_id]['zstack'] = zstack_without_exclusions.copy()
            for label_id in final_ids.keys():
                relative_positions = final_ids[label_id]['relative_positions_per_roi_id'][roi_id]['relative_position']
                inclusion_per_position = all([elem in zstacks_for_quantification[roi_id]['inclusion_criteria_position'] for elem in relative_positions])
                inclusion_per_min_z_extension = len(relative_positions) >= zstacks_for_quantification[roi_id]['inclusion_criteria_min_z_extension']
                if inclusion_per_position == False or inclusion_per_min_z_extension == False:
                    zstacks_for_quantification[roi_id]['zstack'][np.where(zstacks_for_quantification[roi_id]['zstack'] == label_id)] = 0

        return zstacks_for_quantification

    
    def get_n_connected_components_in_zstacks(self, zstacks_for_quantification: Dict) -> Dict:
        for roi_id in zstacks_for_quantification.keys():
            _, zstacks_for_quantification[roi_id]['number_connected_components'] = cc3d.connected_components(zstacks_for_quantification[roi_id]['zstack'], return_N=True)
        return zstacks_for_quantification

    
class EstimateFromGridMethod(QuantificationStrategy):
    
    def quantify(self, quantification_obj: QuantificationObject, database: Database) -> Database:
        # Use the grid method and quantify cells only in some subregions of the image region of interest and estimate count in whole structure from there
        return database  





class Quantifier:
    
    def __init__(self, database: Database, file_ids: List[str]):
        self.database = database
        self.file_ids = file_ids
        self.file_info_dicts = self.get_file_info_dicts()
        self.quantification_preprocessing_strategies = self.database.quantification_configs['quantification_preprocessing_strategies']
        self.quantification_strategy = self.database.quantification_configs['quantification_strategy']
    
    
    def get_file_info_dicts(self) -> Dict:
        file_info_dicts = dict() 
        for file_id in self.file_ids:
            file_info_dicts[file_id] = self.database.get_file_infos(file_id)
        
        return file_info_dicts
        
        
    def run_all(self) -> Database:
        
        for file_id in self.file_ids:
            if self.file_info_dicts[file_id]['quantification_completed'] != True:
                print(f'Quantification of file ID: {file_id} ({self.file_ids.index(file_id) + 1}/{len(self.file_ids)})')  
                quantification_obj = QuantificationObject(file_id, self.database)

                for quant_prepro_strat in self.quantification_preprocessing_strategies:
                    quantification_obj, self.database = quant_prepro_strat.preprocess(quantification_obj, self.database)

                self.database = self.quantification_strategy.quantify(quantification_obj, self.database)
                self.database.update_file_infos(file_id, 'quantification_completed', True)
                del quantification_obj
            
        return self.database
        
        
            
            

