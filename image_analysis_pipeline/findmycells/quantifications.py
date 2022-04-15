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


"""
Fixed integration of ApplyExclusionCriteria between postprocessing and quantifications
"""
   
class QuantificationStrategy(ProcessingStrategy):
    
    @property
    def processing_type(self):
        return 'quantification'
    
    @abstractmethod
    def add_quantification_results_to_database(quantification_object: ProcessingObject, results: Dict) -> ProcessingObject:
        # This method can actually just be copy-pasted into each strategy that inherits from QuantificationStrategy
        # However, it is implemented here like this to remind us to add the following line to the run() method:
        # quantification_object = self.add_quantification_results_to_database(quantification_object = quantification_object, results = quantification_results)
        if hasattr(quantification_object.database, 'quantification_results') == False:
            setattr(quantification_object.database, 'quantification_results', dict())
        if self.__class__.__name__ not in quantification_object.database.quantification_results.keys():
            quantification_object.database.quantification_results[self.__class__.__name__] = dict()
        quantification_object.database.quantification_results[self.__class__.__name__][quantification_object.file_id] = results
        return quantification_object



class QuantificationObject(ProcessingObject):
    
    def __init__(self, database: Database, file_ids: List[str], strategies: List[QuantificationStrategy]) -> None:
        super().__init__(database = database, file_ids = file_ids, strategies = strategies)
        self.file_id = file_ids[0]
        self.file_info = self.database.get_file_infos(identifier = self.file_id)
        # proper file loading needed! segmentations will be saved in individual subdirectories!
        self.segmentations_per_area_roi_id = self.reconstruct_segmentations_per_area_roi_id()
        path = self.database.quantified_segmentations_dir
        self.postprocessed_segmentations = load_zstack_as_array_from_single_planes(path = path, file_id = self.file_id)
        self.rois_dict = self.database.area_rois_for_quantification[self.file_id]
        self.segmentations_per_area_roi_id = dict()
            
    def reconstruct_segmentations_per_area_roi_id(self) -> Dict:
        # to be implemented
        return segmentations_per_area_roi_id
        
  
    


class CountFeaturesInWholeAreaROIs(QuantificationStrategy):
    
    
    def run(self, quantification_object: QuantificationObject) -> QuantificationObject:
        print('-counting the number of image features per region of interest')
        quantification_results = dict()
        for area_roi_id in quantification_object.segmentations_per_area_roi_id.keys():
            _, feature_count = cc3d.connected_components(quantification_object.segmentations_per_area_roi_id[area_roi_id], return_N=True)
            quantification_results[area_roi_id] = feature_count
        quantification_object = self.add_quantification_results_to_database(quantification_object = quantification_object, results = quantification_results)
        return quantification_object


    def add_quantification_results_to_database(quantification_object: QuantificationObject, results: Dict) -> QuantificationObject:
        if hasattr(quantification_object.database, 'quantification_results') == False:
            setattr(quantification_object.database, 'quantification_results', dict())
        if self.__class__.__name__ not in quantification_object.database.quantification_results.keys():
            quantification_object.database.quantification_results[self.__class__.__name__] = dict()
        quantification_object.database.quantification_results[self.__class__.__name__][quantification_object.file_id] = results
        return quantification_object


    def add_strategy_specific_infos_to_updates(self, updates: Dict) -> Dict:
        return updates