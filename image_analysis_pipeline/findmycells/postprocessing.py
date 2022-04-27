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
from .core import ProcessingObject, ProcessingStrategy
from .utils import load_zstack_as_array_from_single_planes, unpad_x_y_dims_in_2d_array, unpad_x_y_dims_in_3d_array, get_polygon_from_instance_segmentation
from .utils import get_rgb_color_code_for_3D



class PostprocessingStrategy(ProcessingStrategy):
    
    @property
    def processing_type(self):
        return 'postprocessing' 



class PostprocessingObject(ProcessingObject):
    
    def __init__(self, database: Database, file_ids: List[str], strategies: List[PostprocessingStrategy], segmentations_to_use: str) -> None:
        super().__init__(database = database, file_ids = file_ids, strategies = strategies)
        self.file_id = file_ids[0]
        self.file_info = self.database.get_file_infos(identifier = self.file_id)
        if segmentations_to_use == 'semantic':
            path = self.database.semantic_segmentations_dir
        elif segmentations_to_use == 'instance':
            path = self.database.instance_segmentations_dir
        else:
            raise ValueError("'segmentations_to_use' has to be either 'semantic' or 'instance'.")
        self.postprocessed_segmentations = load_zstack_as_array_from_single_planes(path = path, file_id = self.file_id)
        self.rois_dict = self.database.area_rois_for_quantification[self.file_id]
        self.segmentations_per_area_roi_id = dict()


    @property
    def processing_type(self):
        return 'postprocessing'
            
    
    def save_postprocessed_segmentations(self) -> None:
        for area_roi_id in self.segmentations_per_area_roi_id.keys():
            for plane_index in range(self.segmentations_per_area_roi_id[area_roi_id].shape[0]):
                image = self.segmentations_per_area_roi_id[area_roi_id][plane_index]
                filepath = self.database.quantified_segmentations_dir.joinpath(area_roi_id)
                if filepath.is_dir() == False:
                    filepath.mkdir()
                filename_path = filepath.joinpath(f'{self.file_id}-{str(plane_index).zfill(3)}_postprocessed_segmentations.png')
                imsave(filename_path, image, check_contrast=False)


    def add_processing_specific_infos_to_updates(self, updates: Dict) -> Dict:
        return updates

                 
            
class ReconstructCellsIn3DFrom2DInstanceLabels(PostprocessingStrategy):
    
    def run(self, processing_object: PostprocessingObject) -> PostprocessingObject:
        print('-Initializing 3D reconstruction from 2D instance segmentations')
        processing_object.postprocessed_segmentations, roi_matching_results = self.run_3d_instance_reconstruction(zstack = processing_object.postprocessed_segmentations)
        processing_object.database = self.save_multimatches_traceback_to_database(database = processing_object.database,
                                                                                      file_id = processing_object.file_id,
                                                                                      results = roi_matching_results)
        return processing_object
    
    
    def run_3d_instance_reconstruction(self, zstack: np.ndarray) -> Tuple[np.ndarray, Dict]:
        pad_width, lowest_final_label_id = 1, 2000
        zstack = np.pad(zstack, pad_width = pad_width, mode = 'constant', constant_values = 0)
        zstack = zstack[pad_width : zstack.shape[0] - pad_width]
        roi_matching_results = self.get_plane_to_plane_roi_matching_results(zstack = zstack)
        for plane_id in range(zstack.shape[0]):
            for label_id in roi_matching_results[plane_id].keys():
                roi_matching_results[plane_id][label_id] = self.find_best_matches(all_results = roi_matching_results, 
                                                                                       original_plane_id = plane_id,
                                                                                       original_label_id = label_id)
        final_ids, roi_matching_results = self.get_final_id_assignments(results = roi_matching_results, lowest_final_label_id = lowest_final_label_id)
        zstack = unpad_x_y_dims_in_3d_array(padded_3d_array = zstack, pad_width = pad_width)
        postprocessed_zstack = self.set_new_label_ids(zstack_with_old_label_ids = zstack, new_ids_assignment = final_ids)
        return postprocessed_zstack, roi_matching_results


    def roi_matching(self, original_roi: Polygon, roi_to_compare: Polygon, label_id_roi_to_compare: int, results: Dict, plane_indicator: str) -> Dict:
        iou = original_roi.intersection(roi_to_compare).area / original_roi.union(roi_to_compare).area
        proportion = original_roi.intersection(roi_to_compare).area / original_roi.area
        if original_roi.within(roi_to_compare): 
            within = True
        else: 
            within = False
        results[f'matching_ids_{plane_indicator}_plane'].append(label_id_roi_to_compare)
        results[f'full_overlap_{plane_indicator}_plane'].append(within)
        results[f'overlapping_area_{plane_indicator}_plane'].append(proportion)
        results[f'IoUs_{plane_indicator}_plane'].append(iou)
        return results


    def get_plane_to_plane_roi_matching_results(self, zstack: np.ndarray, verbose: Optional[bool]=True) -> Dict:
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

    
    def get_final_id_assignments(self, results: Dict, lowest_final_label_id: int=2000) -> Tuple[Dict, Dict]:
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
    

    def save_multimatches_traceback_to_database(self, database: Database, file_id: str, results: Dict) -> Database:
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
        if hasattr(database, 'multi_matches_traceback') == False:
            setattr(database, 'multi_matches_traceback', dict())
        database.multi_matches_traceback[file_id] = multi_matches_traceback        
        return database
                                                                                      
    
    def add_strategy_specific_infos_to_updates(self, updates: Dict) -> Dict:
        return updates          
  
   
            
class ApplyExclusionCriteria(PostprocessingStrategy):
    
    def run(self, processing_object: PostprocessingObject) -> PostprocessingObject:
        print('-applying exclusion criteria')
        instance_label_info = self.get_instance_label_info(postprocessing_object = processing_object)
        self.exclusion_criteria = self.get_exclusion_criteria(postprocessing_object = processing_object)
        all_area_roi_ids = self.get_all_unique_area_roi_ids(rois_dict = processing_object.rois_dict)
        segmentations_per_area_roi_id = dict()
        for area_roi_id in all_area_roi_ids:
            segmentations_per_area_roi_id[area_roi_id] = self.apply_exclusion_criteria(zstack_prior_to_exclusion = processing_object.postprocessed_segmentations,
                                                                                       area_roi_id = area_roi_id,
                                                                                       info = instance_label_info)
        processing_object.segmentations_per_area_roi_id = segmentations_per_area_roi_id
        return processing_object

    
    def get_instance_label_info(self, postprocessing_object: PostprocessingObject) -> Dict:
        instance_label_ids = list(np.unique(postprocessing_object.postprocessed_segmentations))
        for background_label in [0, 0.0]:
            if background_label in instance_label_ids:
                instance_label_ids.remove(background_label)
        instance_label_info = dict()
        for label_id in instance_label_ids:
            instance_label_info[label_id] = dict()
            plane_indices_with_label_id = list(set(np.where(postprocessing_object.postprocessed_segmentations == label_id)[0]))
            instance_label_info[label_id]['plane_indices_with_label_id'] = plane_indices_with_label_id
        instance_label_info = self.extend_info_with_relative_positions(info = instance_label_info, 
                                                                       rois_dict = postprocessing_object.rois_dict,
                                                                       zstack = postprocessing_object.postprocessed_segmentations)
        return instance_label_info
    

    def extend_info_with_relative_positions(self, info: Dict, rois_dict: Dict, zstack: np.ndarray) -> Dict:
        for label_id in info.keys():
            info[label_id]['area_roi_ids_with_matching_plane_index_and_id'] = list()
            info[label_id]['relative_positions_per_area_roi_id'] = dict()
            for plane_index in info[label_id]['plane_indices_with_label_id']:
                if plane_index in rois_dict.keys():
                    for area_roi_id in rois_dict[plane_index]:
                        if (area_roi_id, plane_index, plane_index) not in info[label_id]['area_roi_ids_with_matching_plane_index_and_id']:
                            info[label_id]['area_roi_ids_with_matching_plane_index_and_id'].append((area_roi_id, plane_index, plane_index))
                if 'all_planes' in rois_dict.keys(): # no elif, since there might be some ROIs assigned to single planes and others for the entire stack
                    for area_roi_id in rois_dict['all_planes']:
                        if (area_roi_id, plane_index, 'all_planes') not in info[label_id]['area_roi_ids_with_matching_plane_index_and_id']:
                            info[label_id]['area_roi_ids_with_matching_plane_index_and_id'].append((area_roi_id, plane_index, 'all_planes'))
            for area_roi_id, plane_index, plane_id in info[label_id]['area_roi_ids_with_matching_plane_index_and_id']:
                roi = get_polygon_from_instance_segmentation(single_plane = zstack[plane_index], label_id = label_id)
                area_roi = rois_dict[plane_id][area_roi_id]
                relative_position = self.get_relative_position(roi_to_check = roi, reference = area_roi)
                if area_roi_id not in info[label_id]['relative_positions_per_area_roi_id'].keys():
                    info[label_id]['relative_positions_per_area_roi_id'][area_roi_id] = {'relative_positions': list(),
                                                                                         'plane_indices': list()}
                info[label_id]['relative_positions_per_area_roi_id'][area_roi_id]['relative_positions'].append(relative_position)
                info[label_id]['relative_positions_per_area_roi_id'][area_roi_id]['plane_indices'].append(plane_index)
            for area_roi_id in info[label_id]['relative_positions_per_area_roi_id'].keys():
                relative_positions = list(set(info[label_id]['relative_positions_per_area_roi_id'][area_roi_id]['relative_positions']))
                if 'within' in relative_positions:
                    final_relative_position_for_quantifications = 'within'
                elif 'intersects' in relative_positions:
                    final_relative_position_for_quantifications = 'intersects'
                elif 'touches' in relative_positions:
                    final_relative_position_for_quantifications = 'touches'
                else:
                    final_relative_position_for_quantifications = 'no_overlap'
                info[label_id]['relative_positions_per_area_roi_id'][area_roi_id]['final_relative_position_for_quantifications'] = final_relative_position_for_quantifications
        return info
    

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
    
    
    def get_exclusion_criteria(self, postprocessing_object: PostprocessingObject) -> Dict:
        database = postprocessing_object.database
        total_planes = postprocessing_object.postprocessed_segmentations.shape[0]
        exclusion_criteria = dict()
        if hasattr(database, 'quantification_configs') == False:
            database.quantification_configs = dict()
        if 'exclusion_criteria' in database.quantification_configs.keys():
            if 'allowed_relative_positions' in database.quantification_configs['exclusion_criteria'].keys():
                exclusion_criteria['allowed_relative_positions'] = database.quantification_configs['exclusion_criteria']['allowed_relative_positions']
            if 'minimum_planes_covered' in database.quantification_configs['exclusion_criteria'].keys():
                exclusion_criteria['minimum_planes_covered'] = database.quantification_configs['exclusion_criteria']['minimum_planes_covered']
            if len(list(exclusion_criteria.keys())) < 1:
                warning_line0 = "Warning: You added 'exclusion_criteria' as attribute to the database, but it does not contain valid entries.\n"
                warning_line1 = "         The correct keys are: 'allowed_relative_positions' and 'minimum_planes_covered'."
                warning = warning_line0 + warning_line1
                print(warning)
        if 'allowed_relative_positions' not in exclusion_criteria.keys():
            exclusion_criteria['allowed_relative_positions'] = ['within', 'intersects']
        if 'minimum_planes_covered' not in exclusion_criteria.keys():
            if total_planes > 1:
                minimum_planes_covered = 2
            else:
                minimum_planes_covered = 1
            exclusion_criteria['minimum_planes_covered'] = minimum_planes_covered
        return exclusion_criteria
    
    
    def get_all_unique_area_roi_ids(self, rois_dict: Dict) -> List:
        unique_area_roi_ids = list()
        for plane_id in rois_dict.keys():
            for area_roi_id in rois_dict[plane_id]:
                if area_roi_id not in unique_area_roi_ids:
                    unique_area_roi_ids.append(area_roi_id)
        return unique_area_roi_ids


    def apply_exclusion_criteria(self, zstack_prior_to_exclusion: np.ndarray, area_roi_id: str, info: Dict) -> np.ndarray:
        zstack = zstack_prior_to_exclusion.copy()
        for label_id in info.keys():
            relative_position = info[label_id]['relative_positions_per_area_roi_id'][area_roi_id]['final_relative_position_for_quantifications']
            max_z_expansion = self.get_max_z_expansion(planes = info[label_id]['plane_indices_with_label_id'])
            if relative_position not in self.exclusion_criteria['allowed_relative_positions']:
                zstack[np.where(zstack == label_id)] = 0
            if max_z_expansion < self.exclusion_criteria['minimum_planes_covered']:
                zstack[np.where(zstack == label_id)] = 0
        return zstack      
        
        
    def get_max_z_expansion(self, planes: List) -> int:
        z_dim_expansions = list()
        for index in range(len(planes)):
            z_dim_expansions.append(self.count_continous_plane_ids(start_index = index, plane_indices = planes))
        return max(z_dim_expansions)


    def count_continous_plane_ids(self, start_index: int, plane_indices: List) -> int:
        index = start_index
        keep_going = True
        while keep_going:
            if index < len(plane_indices) - 1:
                if plane_indices[index + 1] == plane_indices[index] + 1:
                    if index < len(plane_indices) - 1:
                        index += 1
                    else:
                        keep_going = False
                else:
                    keep_going = False
            else:
                keep_going = False
        return index - start_index + 1
        

    def add_strategy_specific_infos_to_updates(self, updates: Dict) -> Dict:
        updates['included_relative_positions'] = self.exclusion_criteria['allowed_relative_positions']
        updates['minimum_number_of_planes_required_to_cover'] = self.exclusion_criteria['minimum_planes_covered']
        return updates  