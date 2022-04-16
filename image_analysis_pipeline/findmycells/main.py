from typing import List, Dict, Tuple, Optional
from collections.abc import Callable

from pathlib import Path

from .database import Database
from .preprocessing import PreprocessingStrategy, PreprocessingObject
from .segmentation import SegmentationStrategy, SegmentationObject
from .postprocessing import PostprocessingStrategy, PostprocessingObject
from .quantifications import QuantificationStrategy, QuantificationObject




class Project:
    def __init__(self, user_input: Dict):
        self.project_root_dir = user_input['project_root_dir']
        self.database = Database(user_input)
        
    def save_status(self) -> None:
        self.database.save_all()
    
    def load_status(self) -> None:
        self.database.load_all()
    
    def preprocess(self, strategies: List[PreprocessingStrategy], file_ids: Optional[List]=None, overwrite: bool=False) -> None:
        file_ids = self.database.get_file_ids_to_process(input_file_ids = file_ids, process_tracker_key = 'preprocessing_completed', overwrite = overwrite)
        for file_id in file_ids:
            preprocessing_object = PreprocessingObject(database = self.database, file_ids = [file_id], strategies = strategies)
            preprocessing_object.run_all_strategies()
            preprocessing_object.save_preprocessed_images_on_disk()
            preprocessing_object.save_preprocessed_rois_in_database()
            preprocessing_object.update_database()
            del preprocessing_object


    # process_in_batches(self, batch_size: int=3, findmycells_processing_step: Callable[..., None]) -> None:
        # create batches
        # call function
        # eg, taken from former segment function:
        # file_ids_per_batch = self.database.get_batches_of_file_ids(input_file_ids = file_ids, batch_size = batch_size)
        # for batch_file_ids in file_ids_per_batch:
            # segmentation_object = SegmentationObject(database = self.database, file_ids = batch_file_ids)     
            
        # this function will then also be responsible to add the batch information & id to the database!!
        

    def segment(self, strategies: List[SegmentationStrategy], file_ids: Optional[List]=None,
                run_strategies_individually: bool=True, overwrite: bool=False) -> None:
        if run_strategies_individually:
            for segmentation_strategy in strategies:
                tracker = f'{segmentation_strategy().segmentation_type}_segmentations_done'
                tmp_file_ids = self.database.get_file_ids_to_process(input_file_ids = file_ids, process_tracker_key = tracker, overwrite = overwrite)
                segmentation_object = SegmentationObject(database = self.database, file_ids = tmp_file_ids, strategies = [segmentation_strategy])
                segmentation_object.run_all_strategies()
                del segmentation_object
            file_ids = self.database.get_file_ids_to_process(input_file_ids = file_ids, process_tracker_key = 'segmentation_completed', overwrite = overwrite)
            segmentation_object = SegmentationObject(database = self.database, file_ids = file_ids, strategies = strategies)
            segmentation_object.update_database()
            del segmentation_object
        elif not run_strategies_individually:
            file_ids = self.database.get_file_ids_to_process(input_file_ids = file_ids, process_tracker_key = 'segmentation_completed', overwrite = overwrite)
            segmentation_object = SegmentationObject(database = self.database, file_ids = file_ids, strategies = strategies)
            segmentation_object.run_all_strategies()
            segmentation_object.update_database()
            del segmentation_object


    def postprocess(self, strategies: List[PostprocessingStrategy], segmentations_to_use: str, file_ids: Optional[List]=None, overwrite: bool=False) -> None:
        if segmentations_to_use not in ['semantic', 'instance']:
            raise ValueError('"segmentations_to_use" must be either "semantic" or "instance"')
        else:
            segmentations_to_use_dir = getattr(self.database, f'{segmentations_to_use}_segmentations_dir')
            segmentations_present = False
            for elem in segmentations_to_use_dir.iterdir():
                if elem.is_file():
                    segmentations_present = True
                    break
            if not segmentations_present:
                if segmentations_to_use == 'semantic':
                    error_message_line0 = f'It seems like there are no {segmentations_to_use} segmentations present in the corresponding directory.\n'
                    error_message_line1 = 'You need to run segmentations first, before you can postprocess them.'
                    error_message = error_message_line0 + error_message_line1
                    raise ValueError(error_message)
                else: # has to be instance then
                    error_message_line0 = f'It seems like there are no {segmentations_to_use} segmentations present in the corresponding directory.\n'
                    error_message_line1 = 'Did you mean to use "semantic" instead? Otherwise, please run the respective instance segmentations first.'
                    error_message = error_message_line0 + error_message_line1
                    raise ValueError(error_message)

            file_ids = self.database.get_file_ids_to_process(input_file_ids = file_ids, process_tracker_key = 'postprocessing_completed', overwrite = overwrite)
            for file_id in file_ids:
                print(f'Postprocessing segmentations of file ID: {file_id} ({file_ids.index(file_id) + 1}/{len(file_ids)})')
                postprocessing_object = PostprocessingObject(database = self.database, 
                                                             file_ids = [file_id], 
                                                             strategies = strategies, 
                                                             segmentations_to_use = segmentations_to_use)
                postprocessing_object.run_all_strategies()
                postprocessing_object.save_postprocessed_segmentations()
                postprocessing_object.update_database()
                del postprocessing_object


    def quantify(self, strategies: List[QuantificationStrategy], file_ids: Optional[List]=None, overwrite: bool=False) -> None:
        file_ids = self.database.get_file_ids_to_process(input_file_ids = file_ids, process_tracker_key = 'quantification_completed', overwrite = overwrite)
        for file_id in file_ids:
            print(f'Quantification of file ID: {file_id} ({file_ids.index(file_id) + 1}/{len(file_ids)})')
            quantification_object = QuantificationObject(database = self.database, file_ids = [file_id], strategies = strategies)
            quantification_object.run_all_strategies()
            quantification_object.update_database()
            del quantification_object
            
    """        
    def inspect(self, quantification_strategy_index: int=0, file_ids: Optional[List]=None, 
                area_roi_ids: Optional[List]=None, label_indices: Optional[List]=None, show: bool=True, save: bool=False) -> None:
        from .inspection import InspectionObject
        quantification_strategy_str = list(self.database.quantification_results.keys())[quantification_strategy_index]
        file_ids_not_quantified = self.database.get_file_ids_to_process(input_file_ids = file_ids, process_tracker_key = 'quantification_completed', overwrite = False)
        if file_ids == None:
            file_ids = self.database.file_infos['file_id']
        file_ids_quantified = [elem for elem in file_ids if elem not in file_ids_not_quantified]
        for file_id in file_ids_quantified:
            valid_area_roi_ids = self.database.area_rois_for_quantification[file_id]['all_planes'].keys()
            if area_roi_ids == None:
                tmp_area_roi_ids = valid_area_roi_ids
            else:
                tmp_area_roi_ids = [elem for elem in area_roi_ids if elem in valid_area_roi_ids]
            for area_roi_id in tmp_area_roi_ids:
                total_labels = self.database.quantification_results[quantification_strategy_str][file_id][area_roi_id]
                if label_indices == None:
                    tmp_label_indices = [elem for elem in range(total_labels)]
                else:
                    tmp_label_indices = [elem for elem in label_indices if elem < total_labels]
                for label_index in tmp_label_indices:
                    inspection_object = InspectionObject(database = self.database, file_id = file_id, area_roi_id = area_roi_id, label_index = label_index, show = show, save = save)
                    inspection_object.run_all_inspection_steps()

                    
    def run_inspection(self, file_id: str, inspection_strategy):
        from .inspection import InspectionStrategy
        inspection_strategy.run(self.database, file_id)
    """
        
    
    def remove_file_id_from_project(self, file_id: str):
        self.database.remove_file_id_from_project(file_id = file_id)

        
        
class Preparations:
    
    def create_excel_files_for_image_loader(self, root: Path, destination: Path, batch_processing: bool=True) -> None:
        from .prepare_my_data import CreateExcelFilesForImageLoader
        preparation = CreateExcelFilesForImageLoader(root = root, destination = destination, batch_processing = batch_processing)
        

        
        
        
        
        
        
        


            
            
            
