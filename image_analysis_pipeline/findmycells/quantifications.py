from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import cc3d

from .database import Database
from .core import ProcessingObject, ProcessingStrategy
from .utils import load_zstack_as_array_from_single_planes



   
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
        self.segmentations_per_area_roi_id = self.load_postprocessed_segmentations()


    @property
    def processing_type(self):
        return 'quantification'


    def load_postprocessed_segmentations(self) -> Dict:
        segmentations_per_area_roi_id = dict()
        for elem in self.database.quantified_segmentations_dir.iterdir():
            if elem.is_dir():
                area_roi_id = elem.name
                segmentations_per_area_roi_id[area_roi_id] = load_zstack_as_array_from_single_planes(path = elem, file_id = self.file_id)
        return segmentations_per_area_roi_id
            

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
        
    


class CountFeaturesInWholeAreaROIs(QuantificationStrategy):
    
    
    def run(self, processing_object: QuantificationObject) -> QuantificationObject:
        print('-counting the number of image features per region of interest')
        quantification_results = dict()
        for area_roi_id in processing_object.segmentations_per_area_roi_id.keys():
            _, feature_count = cc3d.connected_components(processing_object.segmentations_per_area_roi_id[area_roi_id], return_N=True)
            quantification_results[area_roi_id] = feature_count
        processing_object = self.add_quantification_results_to_database(quantification_object = processing_object, results = quantification_results)
        return processing_object


    def add_quantification_results_to_database(self, quantification_object: QuantificationObject, results: Dict) -> QuantificationObject:
        if hasattr(quantification_object.database, 'quantification_results') == False:
            setattr(quantification_object.database, 'quantification_results', dict())
        if self.__class__.__name__ not in quantification_object.database.quantification_results.keys():
            quantification_object.database.quantification_results[self.__class__.__name__] = dict()
        quantification_object.database.quantification_results[self.__class__.__name__][quantification_object.file_id] = results
        return quantification_object


    def add_strategy_specific_infos_to_updates(self, updates: Dict) -> Dict:
        return updates