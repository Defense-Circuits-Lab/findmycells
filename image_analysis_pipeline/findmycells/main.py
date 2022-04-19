from typing import List, Dict, Tuple, Optional
from collections.abc import Callable

from pathlib import Path
import random

from .database import Database
from .core import ProcessingStrategy
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
            

    def segment(self, strategies: List[SegmentationStrategy], file_ids: Optional[List]=None, batch_size: Optional[int]=None,
                run_strategies_individually: bool=True, overwrite: bool=False, autosave: bool=True, clear_tmp_data: bool=False) -> None:
        # check if there is a new strategy - if yes: reset "segmentation_completed" for all files to "None"
        self.database.file_infos['segmentation_completed'] = self.reset_file_infos_if_new_strategy(strategies = strategies)
        
        if type(batch_size) == int:
            file_ids_per_batch = self.create_batches(batch_size = batch_size, file_ids = file_ids, process_tracker_key = 'segmentation_completed', overwrite = overwrite)
            if file_ids_per_batch == None:
                return None
        else:
            file_ids_per_batch = [file_ids]
        
        if run_strategies_individually:
            for segmentation_strategy in strategies:
                for batch_file_ids in file_ids_per_batch:
                    tracker = f'{segmentation_strategy().segmentation_type}_segmentations_done'
                    tmp_file_ids = self.database.get_file_ids_to_process(input_file_ids = batch_file_ids, process_tracker_key = tracker, overwrite = overwrite)
                    if len(tmp_file_ids) > 0:
                        segmentation_object = SegmentationObject(database = self.database, file_ids = tmp_file_ids, strategies = [segmentation_strategy])
                        segmentation_object.run_all_strategies()
                        del segmentation_object
                        if autosave:
                            self.database.save_all()
            all_file_ids = []
            for batch_file_ids in file_ids_per_batch:
                all_file_ids += batch_file_ids
            all_file_ids = self.database.get_file_ids_to_process(input_file_ids = all_file_ids, process_tracker_key = 'segmentation_completed', overwrite = overwrite)
            segmentation_object = SegmentationObject(database = self.database, file_ids = all_file_ids, strategies = strategies)
            segmentation_object.update_database()
            del segmentation_object
            if autosave:
                self.database.save_all()
        else:
            for batch_file_ids in file_ids_per_batch:
                batch_file_ids = self.database.get_file_ids_to_process(input_file_ids = batch_file_ids, process_tracker_key = 'segmentation_completed', overwrite = overwrite)
                segmentation_object = SegmentationObject(database = self.database, file_ids = batch_file_ids, strategies = strategies)
                segmentation_object.run_all_strategies()
                segmentation_object.update_database()
                del segmentation_object
                if autosave:
                    self.database.save_all()
        
        if clear_tmp_data:
            file_ids = self.database.get_file_ids_to_process(input_file_ids = None, process_tracker_key = 'segmentation_completed', overwrite = True)
            segmentation_object = SegmentationObject(database = self.database, file_ids = file_ids, strategies = strategies)
            segmentation_object.clear_all_tmp_data()


    def reset_file_infos_if_new_strategy(self, strategies: List[ProcessingStrategy]) -> List:
        new_strategy = False
        for strategy in strategies:
            processing_type = strategy().processing_type
            strategy_name = strategy().strategy_name
            matching_index = [key for key, column in self.database.file_infos.items() if f'{processing_type}_step' in key and strategy_name in column]
            if len(matching_index) == 0:
                new_strategy = True
                break
        if f'{processing_type}_completed' not in self.database.file_infos.keys():
            column = [None] * len(self.database.file_infos['file_ids'])
        elif new_strategy:
            column = [None] * len(self.database.file_infos['file_ids'])
        else:
            column = self.database.file_infos[f'{processing_type}_completed']
        return column


    def create_batches(self, batch_size: int, file_ids: List[str], process_tracker_key: str, overwrite: bool) -> Optional[List]:
            if batch_size <= 0:
                raise ValueError('"batch_size" must be greater than 0!')
            all_file_ids = self.database.get_file_ids_to_process(input_file_ids = file_ids, process_tracker_key = process_tracker_key, overwrite = overwrite)
            if len(all_file_ids) == 0:
                file_ids_per_batch = None
            else:
                file_ids_per_batch = []
                while len(all_file_ids) > 0:
                    if len(all_file_ids) >= batch_size:
                        subsample = random.sample(all_file_ids, batch_size)
                        for elem in subsample:
                            all_file_ids.remove(elem)
                        file_ids_per_batch.append(subsample)
                    else:
                        file_ids_per_batch.append(all_file_ids)
                        all_file_ids = []
            return file_ids_per_batch
    

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
        

        
        
        
        
        
        
        


            
            
            
