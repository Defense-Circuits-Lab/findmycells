from abc import ABC, abstractmethod
import os
from pathlib import Path
import numpy as np
import pickle
from datetime import datetime
from .utils import listdir_nohidden
import pandas as pd
import shutil
import random
from typing import List, Dict, Optional, Union

# Global variables required for the Database:
MAIN_SUBDIR_ATTRIBUTES = {'preprocessed_images_dir': {'foldername': '02_preprocessed_images', 'key_substring': 'preprocessed'},
                          'segmentation_tool_dir': {'foldername': '03a_segmentation_tool', 'key_substring': 'tool'},
                          'semantic_segmentations_dir': {'foldername': '03b_semantic_segmentations', 'key_substring': 'semantic'},
                          'instance_segmentations_dir': {'foldername': '03c_instance_segmentations', 'key_substring': 'instance'},
                          'segmentations_used_for_quantifications_dir': {'foldername': '04_segmentations_used_for_quantification', 'key_substring': 'quantification'},
                          'results_dir': {'foldername': '05_results', 'key_substring': 'results'},
                          'inspection_dir': {'foldername': '06_inspection', 'key_substring': 'inspection'}}

SEGMENTATION_TOOL_SUBDIR_ATTRIBUTES = {'trained_models_dir': {'foldername': 'trained_models', 'key_substring': 'models'},
                                       'segmentation_tool_temp_dir': {'foldername': 'temp', 'key_substring': 'temp'}}
                                
INSPECTION_SUBDIR_ATTRIBUTES = {'inspected_area_plots_dir': {'foldername': 'inspected_area_plots', 'key_substring': 'inspected_area'},
                                'inspection_final_label_planes_dir': {'foldername': 'planes_with_final_label_ids', 'key_substring': 'final_label_ids'},
                                'inspection_planes_for_quantification': {'foldername': 'planes_for_quantification', 'key_substring': 'for_quantification'}}

RENAMING_DICT = {'file_id': 'File ID', 
                 'original_file_id': 'Original File ID',               
                 'group_id': 'Group ID', 
                 'subject_id': 'Subject ID', 
                 'microscopy_filepath': 'Microscopy Filepath', 
                 'microscopy_filetype': 'Microscopy Filetype',
                 'rois_present': 'Rois Present', 
                 'rois_filepath': 'Rois Filepath',
                 'rois_filetype': 'Rois Filetype',
                 'preprocessing_completed': 'Preprocessing Completed',
                 'RGB': 'RGB',
                 'total_image_planes': 'Total Image Planes',
                 'cropping_method': 'Cropping Method',
                 'cropping_row_indices': 'Cropping Row Indices',
                 'cropping_column_indices': 'Cropping Column Indices',
                 'quantification_completed': 'Quantification Completed'}


class Database():
    '''
    The database object is intended to collect and hold all information about
    the image analysis project at hand. Depending on the type of analysis that 
    shall be performed, the Database needs to be flexible and adopt to the 
    respective needs. For instance, there might be more than just two groups 
    that are investigated, or just a single group but with multiple images per 
    subject, and so on.
    '''
    
    def __init__(self, user_input_via_gui: dict) -> None:
        self.extract_user_input(user_input_via_gui)
        self.construct_main_subdirectories()
        if hasattr(self, 'only_duplication') == False:
            self.create_file_infos()
        elif self.only_duplication == False:
            self.create_file_infos()
        else:
            self.load_all()
        
    
    def extract_user_input(self, user_input: dict) -> None:
        for key, value in user_input.items():
            if hasattr(self, key) == False:
                setattr(self, key, value)


    def construct_main_subdirectories(self) -> None:
        subdirectories = listdir_nohidden(self.project_root_dir)
        # Mandatory: directories with microscopy images and the rois of the areas that shall be quantified has to be created by the user
        self.microscopy_image_dir = self.project_root_dir.joinpath([elem for elem in subdirectories if 'microscopy' in elem][0])
        self.rois_to_analyze_dir = self.project_root_dir.joinpath([elem for elem in subdirectories if 'rois' in elem][0])
        # Remaining directories that are currently not required to exist when the database object is created:
        self.check_and_create_remaining_directories(root_dir = self.project_root_dir, subdirectory_attributes = MAIN_SUBDIR_ATTRIBUTES)   
        self.check_and_create_remaining_directories(root_dir = self.segmentation_tool_dir, subdirectory_attributes = SEGMENTATION_TOOL_SUBDIR_ATTRIBUTES)
        self.check_and_create_remaining_directories(root_dir = self.inspection_dir, subdirectory_attributes = INSPECTION_SUBDIR_ATTRIBUTES)
    
    
    def check_and_create_remaining_directories(self, root_dir: Path, subdirectory_attributes: Dict) -> None:
        existing_subdirectories = listdir_nohidden(root_dir)
        for attribute_key in subdirectory_attributes.keys():
            elements_matching_key_substring = [elem for elem in existing_subdirectories if subdirectory_attributes[attribute_key]['key_substring'] in elem]
            if len(elements_matching_key_substring) > 0:
                for matching_element in elements_matching_key_substring:
                    if (root_dir.joinpath(matching_element).is_dir()) & (hasattr(self, attribute_key) == False):
                        setattr(self, attribute_key, root_dir.joinpath(matching_element))
            if hasattr(self, attribute_key) == False:
                subdirectory_path = root_dir.joinpath(subdirectory_attributes[attribute_key]['foldername'])
                subdirectory_path.mkdir()
                setattr(self, attribute_key, subdirectory_path)                       
    
    
    def create_file_infos(self) -> None:
        # Initial information will be retrieved from the microscopy_image_dir
        self.file_infos = {'file_id': list(),
                           'original_file_id': list(),
                           'group_id': list(),
                           'subject_id': list(),
                           'microscopy_filepath': list(),
                           'microscopy_filetype': list(),
                           'rois_present': list(),
                           'rois_filepath': list(),
                           'rois_filetype': list()}
        file_id = 0
        for group in listdir_nohidden(self.microscopy_image_dir):
            for subject in listdir_nohidden(self.microscopy_image_dir.joinpath(group)):
                for filename in listdir_nohidden(self.microscopy_image_dir.joinpath(group, subject)):
                    self.file_infos['file_id'].append(str(file_id).zfill(4))
                    original_file_id = filename[:filename.find('.')]
                    self.file_infos['original_file_id'].append(original_file_id)
                    self.file_infos['group_id'].append(group)
                    self.file_infos['subject_id'].append(subject)
                    self.file_infos['microscopy_filepath'].append(self.microscopy_image_dir.joinpath(group, subject, filename))
                    self.file_infos['microscopy_filetype'].append(filename[filename.find('.'):])
                    
                    matching_roi_filenames = [elem for elem in listdir_nohidden(self.rois_to_analyze_dir.joinpath(group,subject)) if elem.startswith(original_file_id)]
                    if len(matching_roi_filenames) == 0:
                        self.file_infos['rois_present'].append(False)
                        self.file_infos['rois_filepath'].append('not_available')
                        self.file_infos['rois_filetype'].append('not_available')                      
                    elif len(matching_roi_filenames) == 1:
                        roi_filename = matching_roi_filenames[0]
                        self.file_infos['rois_present'].append(True)
                        self.file_infos['rois_filepath'].append(self.rois_to_analyze_dir.joinpath(group, subject, roi_filename))
                        self.file_infos['rois_filetype'].append(roi_filename[roi_filename.find('.'):])
                    else:
                        message_line_0 = 'It seems like you provided more than a single ROI file in:\n'
                        message_line_1 = f'{self.rois_to_analyze_dir.joinpath(group,subject)}\n'
                        message_line_2 = 'If you want to quantify image features within multiple ROIs per image, please use RoiSets created with ImageJ as described here:\n'
                        message_line_3 = 'Documentation not live yet - please contact: segebarth_d@ukw.de for more information.'
                        error_message = message_line_0 + message_line_1 + message_line_2 + message_line_3
                        raise ValueError(error_message)
                    
                    file_id += 1
                
                    
    def get_file_infos(self, identifier: str) -> Dict:
        # supports use of either original_file_id, file_id, or microscopy_filepath as input parameter identifier       
        if identifier in self.file_infos['file_id']:
            index = self.file_infos['file_id'].index(identifier)
        elif identifier in self.file_infos['original_file_id']:
            index = self.file_infos['original_file_id'].index(identifier)
        elif identifier in self.file_infos['microscopy_filepath']:
            index = self.file_infos['microscopy_filepath'].index(identifier)
        else:
            raise NameError(f'{identifier} is not a valid input!')
        
        file_infos = dict()    
        for key in self.file_infos.keys():
            if len(self.file_infos[key]) > 0:
                file_infos[key] = self.file_infos[key][index]
         
        return file_infos
    
    
    def add_new_key_to_file_infos(self, key: str, values: Optional[List]=None, preferred_empty_value: Union[bool, str, None]=None) -> None:
        """
        Allows us to add a new key-value-pair to the file_infos dict
        If values is not passed, a list full of 'preferred_empty_value' that matches the length of file_ids will be created
        If values is passed, it has to be a list of the length of file_id
        """
        if key in self.file_infos.keys():
            raise ValueError("The key you are trying to add is already in file_infos.")
        else:
            length = len(self.file_infos['file_id'])
            if values == None:
                values = [preferred_empty_value] * length
                self.file_infos[key] = values
            elif type(values) != list:
                raise TypeError("'values' has to be 'None' or a list that matches the length of file_infos['file_ids']")
            else:
                if len(values) == length:
                    self.file_infos[key] = values                
                else:
                    raise ValueError("The list of values that you provided does not match the length of file_infos['file_ids']!")
            

    def update_file_infos(self, file_id: str, updates: Dict, preferred_empty_value: Union[bool, str, None]=None) -> None: 
        index = self.file_infos['file_id'].index(file_id)
        for key, value in updates.items():
            if key not in self.file_infos.keys():
                self.add_new_key_to_file_infos(key, preferred_empty_value = preferred_empty_value)
            self.file_infos[key][index] = value

    
    def get_file_ids_to_process(self, input_file_ids: Optional[List], process_tracker_key: str, overwrite: bool) -> List:
        if input_file_ids == None:
            input_file_ids = self.file_infos['file_id']
        if process_tracker_key not in self.file_infos.keys():
            self.add_new_key_to_file_infos(process_tracker_key)
        if overwrite:
            output_file_ids = input_file_ids
        else:
            process_tracker_status = list()
            for file_id in input_file_ids:
                index = self.file_infos['file_id'].index(file_id)
                process_tracker_status.append(self.file_infos[process_tracker_key][index])
            output_file_ids = [elem[0] for elem in zip(input_file_ids, process_tracker_status) if elem[1] == False or elem[1] == None]
        return output_file_ids


    def get_batches_of_file_ids(self, input_file_ids: Optional[List], batch_size: int) -> List[List[int]]:
        if input_file_ids == None:
            input_file_ids = self.file_infos['file_id']        
        if len(input_file_ids) % batch_size == 0:
            total_batches = int(len(input_file_ids) / batch_size)
        else:
            total_batches = int(len(input_file_ids) / batch_size) + 1
        file_ids_per_batch = list()
        for batch in range(total_batches):
            if len(input_file_ids) >= batch_size:
                sampled_file_ids = random.sample(input_file_ids, batch_size)
            else:
                sampled_file_ids = input_file_ids.copy()
            file_ids_per_batch.append(sampled_file_ids)
            for file_id in sampled_file_ids:
                input_file_ids.remove(file_id)
        return file_ids_per_batch    
    
    
    def save_all(self) -> None:
        self.save_csv()
        self.save_file_infos()
        self.save_project_configs()
    
    
    def save_csv(self) -> None:
        df = pd.DataFrame(self.file_infos)
        current_columns = list(df.columns)
        new_columns = list()
        for column_name in current_columns:
            if column_name in RENAMING_DICT.keys():
                new_columns.append(RENAMING_DICT[column_name])
            else: 
                print(f"Warning: {column_name} not yet specified in renaming dictionary")
                new_columns.append(column_name)
        df.columns=new_columns
        df.to_csv(self.results_dir.joinpath(f'{datetime.now().strftime("%Y_%m_%d")}_findmycells_overview_for_user.csv').as_posix())
    
    
    def save_file_infos(self) -> None:
        filepath = self.results_dir.joinpath(f'{datetime.now().strftime("%Y_%m_%d")}_findmycells_project_summary.p').as_posix()
        with open(filepath, 'wb') as io:
            pickle.dump(self.file_infos, io)
            
        
    def save_project_configs(self) -> None:
        project_configs = self.__dict__.copy()
        if 'file_infos' in project_configs.keys():
            project_configs.pop('file_infos')
        if 'only_duplication' in project_configs.keys():
            project_configs.pop('only_duplication')  
            
        filepath = self.results_dir.joinpath(f'{datetime.now().strftime("%Y_%m_%d")}_findmycells_project_configs.p').as_posix()       
        with open(filepath, 'wb') as io:
            pickle.dump(project_configs, io)
    
    
    def load_all(self) -> None:
        result_files = [fname for fname in listdir_nohidden(self.results_dir) if fname.endswith('.p')]
        result_files.sort(reverse = True)
        if len(result_files) < 2:
            raise FileNotFoundError(f"Couldn´t find the required files in {self.results_dir.as_posix()}")
        
        else:
            project_summary_filename = [fname for fname in result_files if fname.endswith('project_summary.p')][0]
            with open(self.results_dir.joinpath(project_summary_filename).as_posix(), 'rb') as io:
                self.file_infos = pickle.load(io)

            project_configs_filename = [fname for fname in result_files if fname.endswith('project_configs.p')][0]
            with open(self.results_dir.joinpath(project_configs_filename).as_posix(), 'rb') as io:
                project_configs = pickle.load(io)
            
            for key, value in project_configs.items():
                if hasattr(self, key) == False:
                    setattr(self, key, value)

    
    def remove_file_id_from_project(self, file_id: str) -> None:
        index = self.file_infos['file_id'].index(file_id)
        original_file_id = self.file_infos['original_file_id'][index]
        # Move all source files, i.e. microscopy image file and roi file(s):
        subdirectories = listdir_nohidden(self.project_root_dir)
        self.check_and_create_remaining_directories(root_dir = self.project_root_dir,
                                                    subdirectory_attributes = {'removed_files_dir': {'foldername': '08_removed_files', 'key_substring': 'removed_files'}})
        for source_data_type in ['microscopy', 'rois']:
            source_filepath = self.file_infos[f'{source_data_type}_filepath'][index]
            if type(source_filepath) == list:
                for filepath in source_filepath:
                    shutil.move(filepath.as_posix(), self.removed_files_dir.as_posix())
            else:
                shutil.move(source_filepath.as_posix(), self.removed_files_dir.as_posix())
        # Delete all files that were already generated from findmycells:
        for directory in [self.preprocessed_images_dir, 
                          self.semantic_segmentations_dir, 
                          self.instance_segmentations_dir, 
                          self.inspected_area_plots_dir,
                          self.inspection_final_label_planes_dir,
                          self.inspection_planes_for_quantification]:
            filenames = listdir_nohidden(directory)
            for filename in filenames:
                if filename.startswith(file_id):
                    os.remove(directory.joinpath(filename).as_posix())
        # Remove from file_infos:
        for key in self.file_infos.keys():
            self.file_infos[key].pop(index)
        # Remove from area_rois_for_quantification:
        if file_id in self.area_rois_for_quantification.keys():
            self.area_rois_for_quantification.pop(file_id)
         
        
    def import_rois_dict(self, file_id: str, rois_dict: Dict) -> None:
        if hasattr(self, 'area_rois_for_quantification') == False:
            self.area_rois_for_quantification = dict()
        self.area_rois_for_quantification[file_id] = rois_dict