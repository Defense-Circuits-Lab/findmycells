# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/api/02_database.ipynb.

# %% auto 0
__all__ = ['Database', 'FileHistory']

# %% ../nbs/api/02_database.ipynb 2
from pathlib import Path, PosixPath, WindowsPath
from typing import Optional, Dict, List, Union
import pandas as pd
from datetime import datetime
from shapely.geometry import Polygon
import pickle


from .configs import ProjectConfigs
from . import utils

# %% ../nbs/api/02_database.ipynb 4
class Database:
    
    def __init__(self, project_configs: ProjectConfigs) -> None:
        self.project_configs = project_configs
        self._initialize_project_in_root_dir()
        self._create_file_infos_as_attr()
        self._create_file_histories_as_attr()
        
        
    def _initialize_project_in_root_dir(self) -> None:
        self._initialize_all_top_level_subdirectories()
        self._initialize_segmentation_tool_subdirectories()
        if len(utils.list_dir_no_hidden(path = self.project_configs.root_dir.joinpath(self.microscopy_images_dir), only_dirs = True)) > 0:
            self._assert_valid_microscopy_image_subdir_tree_structure()
    
          
    def _initialize_all_top_level_subdirectories(self) -> None:
        self._find_or_create_subdir(target_name = 'microscopy_images', keywords = ['microscopy', 'Microscopy'])
        self._find_or_create_subdir(target_name = 'rois_to_analyze', keywords = ['rois', 'ROIs', 'ROIS', 'Rois'])
        self._find_or_create_subdir(target_name = 'preprocessed_images', keywords = ['preprocessed', 'Preprocessed', 'pre-processed'])
        self._find_or_create_subdir(target_name = 'segmentation_tool', keywords = ['tool', 'Tool'])
        self._find_or_create_subdir(target_name = 'semantic_segmentations', keywords = ['semantic', 'Semantic'])
        self._find_or_create_subdir(target_name = 'instance_segmentations', keywords = ['instance', 'Instance'])
        self._find_or_create_subdir(target_name = 'quantified_segmentations', keywords = ['quantified', 'Quantified', 'quantification', 'Quantification'])
        self._find_or_create_subdir(target_name = 'results', keywords = ['results', 'Results'])
        self._find_or_create_subdir(target_name = 'inspection', keywords = ['inspect', 'Inspect'])
        
        
    def _find_or_create_subdir(self, target_name: str, keywords: List[str], parent_dir: Optional[Path]=None) -> None:
        if parent_dir == None:
            parent_dir = self.project_configs.root_dir
        subdir_found = False
        for path in parent_dir.iterdir():
            if path.is_dir():
                for key in keywords:
                    if key in path.name:
                        subdir_found = True
                        subdir_path = path
                        break
        if subdir_found == False:
            subdir_path = parent_dir.joinpath(target_name)
            subdir_path.mkdir()
        setattr(self, f'{target_name}_dir', subdir_path.name)
                                               
    
    def _initialize_segmentation_tool_subdirectories(self) -> None:    
        self._find_or_create_subdir(target_name = 'trained_models',
                                    keywords = ['models'],
                                    parent_dir = self.project_configs.root_dir.joinpath(self.segmentation_tool_dir))
        self._find_or_create_subdir(target_name = 'segmentation_tool_temp',
                                    keywords = ['tmp', 'temp'],
                                    parent_dir = self.project_configs.root_dir.joinpath(self.segmentation_tool_dir))
                                               

    def _create_file_infos_as_attr(self) -> None:
        file_infos = {'file_id': [],
                      'original_filename': [],
                      'main_group_id': [],
                      'subgroup_id': [],
                      'subject_id': [],
                      'microscopy_filepath': [],
                      'microscopy_filetype': [],
                      'rois_present': [],
                      'rois_filepath': [],
                      'rois_filetype': []}
        setattr(self, 'file_infos', file_infos)
        
        
    def _create_file_histories_as_attr(self) -> None:
        setattr(self, 'file_histories', {})
        
        
    def compute_file_infos(self) -> None:
        self._initialize_microscopy_images_subdirectory_tree()
        self._add_new_files_to_database()
        self._identify_removed_files() # ToDo: not implemented yet
        
        
    def _initialize_microscopy_images_subdirectory_tree(self) -> None:
        if len(utils.list_dir_no_hidden(path = self.project_configs.root_dir.joinpath(self.microscopy_images_dir), only_dirs = True)) > 0:
            self._assert_valid_microscopy_image_subdir_tree_structure()
        else:
            self._create_representative_microscopy_image_subdir_tree()
            
            
    def _assert_valid_microscopy_image_subdir_tree_structure(self) -> None:
        microscopy_images_dir_path = self.project_configs.root_dir.joinpath(self.microscopy_images_dir)
        for main_group_id_subdir_path in utils.list_dir_no_hidden(path = microscopy_images_dir_path, only_dirs = True):
            tmp_subgroup_subdir_paths = utils.list_dir_no_hidden(path = main_group_id_subdir_path, only_dirs = True)
            subgroup_dirs_missing_message = f'Invalid microscopy images subdir structure! Expected at least one subdirectory in {main_group_id_subdir_path}.'
            assert len(tmp_subgroup_subdir_paths) > 0, subgroup_dirs_missing_message
            for subgroup_id_subdir_path in tmp_subgroup_subdir_paths:
                tmp_subject_subdir_paths = utils.list_dir_no_hidden(path = subgroup_id_subdir_path, only_dirs = True)
                subject_subdirs_missing_message = f'Invalid microscopy images subdir structure! Expected at least one subdirectory in {subgroup_id_subdir_path}.'
                assert len(tmp_subject_subdir_paths) > 0, subject_subdirs_missing_message
                           
                            
    def _create_representative_microscopy_image_subdir_tree(self) -> None:
        for representative_main_group_id in ['wildtype', 'transgenic']:
            for representative_subgroup_id in ['week_1', 'week_4']:
                if representative_main_group_id == 'wildtype':
                    subject_ids = ['mouse_1', 'mouse_2', 'mouse_3']
                else:
                    subject_ids = ['mouse_4', 'mouse_5', 'mouse_6']
                for representative_subject_id in subject_ids:
                    self._make_subdir_tree(main_group_id = representative_main_group_id,
                                           subgroup_id = representative_subgroup_id,
                                           subject_id = representative_subject_id)
                            
                            
    def _make_subdir_tree(self, main_group_id: str, subgroup_id: str, subject_id: str) -> None:
        microscopy_images_dir = self.project_configs.root_dir.joinpath(self.microscopy_images_dir)
        microscopy_images_dir.joinpath(main_group_id).mkdir(exist_ok = True)
        microscopy_images_dir.joinpath(main_group_id, subgroup_id).mkdir(exist_ok = True)
        microscopy_images_dir.joinpath(main_group_id, subgroup_id, subject_id).mkdir(exist_ok = True)
        
        
    def _add_new_files_to_database(self) -> None:
        microscopy_images_dir_path = self.project_configs.root_dir.joinpath(self.microscopy_images_dir)
        for main_group_id_subdir_path in utils.list_dir_no_hidden(path = microscopy_images_dir_path, only_dirs = True):
            for subgroup_id_subdir_path in utils.list_dir_no_hidden(path = main_group_id_subdir_path, only_dirs = True):
                for subject_id_subdir_path in utils.list_dir_no_hidden(path = subgroup_id_subdir_path, only_dirs = True):
                    for filepath in utils.list_dir_no_hidden(path = subject_id_subdir_path, only_files = True):
                        new_file_found = self._is_this_a_new_file(filepath = filepath)
                        if new_file_found == True:
                            file_id = self._get_next_available_file_id()
                            self._append_details_to_file_infos(file_id = file_id, filepath = filepath)
                            self._add_new_file_history_tracker(file_id = file_id, source_image_filepath = filepath)                                       
        
    
    def _is_this_a_new_file(self, filepath: Path) -> bool:
        subject_subdir_path = filepath.parent
        subgroup_subdir_path = subject_subdir_path.parent
        main_group_id = subgroup_subdir_path.parent.name
        original_filename = filepath.name[:filepath.name.find('.')]
        file_infos_as_df = pd.DataFrame(data = self.file_infos)
        matching_entries_df = file_infos_as_df.loc[(file_infos_as_df['main_group_id'] == main_group_id) &
                                                   (file_infos_as_df['subgroup_id'] == subgroup_subdir_path.name) &
                                                   (file_infos_as_df['subject_id'] == subject_subdir_path.name) &
                                                   (file_infos_as_df['original_filename'] == original_filename)]
        matching_entries_count = matching_entries_df.shape[0]
        if matching_entries_count == 0:
            is_new_file = True
        elif matching_entries_count == 1:
            is_new_file = False
        else:
            conflicting_file_ids = matching_entries_df['file_id'].values
            raise ValueError((f'Found multiple entries in file_infos for {filepath}.'
                              'This is an unexpected behavior and needs to be resolved. Please '
                              'Try to remove the file that was '
                              'reported above by using the steps described in "removing files '
                              'from a findmycells project". Since this process requires you '
                              'to specify the respective file IDs of the files you´d like to '
                              'remove, please find the conflicting IDs below. You have to remove '
                              'at least all but one, yet removing all and then adding one correct '
                              f'again is recommended.\n Conflicting file IDs: {conflicting_file_ids}.'))
        return is_new_file
        
        
    def _get_next_available_file_id(self) -> str:
        if len(self.file_infos['file_id']) > 0:
            next_available_file_id = max([int(file_id_str) for file_id_str in self.file_infos['file_id']]) + 1
        else:
            next_available_file_id = 0
        return str(next_available_file_id).zfill(4)
                                               
    
    
    def _append_details_to_file_infos(self, file_id: int, filepath: Path) -> None:
        subject_subdir_path = filepath.parent
        subgroup_subdir_path = subject_subdir_path.parent
        main_group_subdir_path = subgroup_subdir_path.parent
        self.file_infos['file_id'].append(str(file_id).zfill(4))
        original_filename = filepath.name[:filepath.name.find('.')]
        self.file_infos['original_filename'].append(original_filename)
        self.file_infos['main_group_id'].append(main_group_subdir_path.name)
        self.file_infos['subgroup_id'].append(subgroup_subdir_path.name)
        self.file_infos['subject_id'].append(subject_subdir_path.name)
        self.file_infos['microscopy_filepath'].append(filepath)
        self.file_infos['microscopy_filetype'].append(filepath.suffix)
        corresponding_dir_in_rois_to_analyze_dir = self.project_configs.root_dir.joinpath(self.rois_to_analyze_dir,
                                                                                          main_group_subdir_path.name,
                                                                                          subgroup_subdir_path.name,
                                                                                          subject_subdir_path.name)
        if corresponding_dir_in_rois_to_analyze_dir.is_dir() == False:
            self.file_infos['rois_present'].append(False)
            self.file_infos['rois_filepath'].append('not_available')
            self.file_infos['rois_filetype'].append('not_available')
        else:
            matching_roi_filepaths = []
            for roi_filepath in utils.list_dir_no_hidden(path = corresponding_dir_in_rois_to_analyze_dir, only_files = True):
                if roi_filepath.name[:roi_filepath.name.find('.')] == original_filename:
                    matching_roi_filepaths.append(roi_filepath)
            if len(matching_roi_filepaths) == 0:
                self.file_infos['rois_present'].append(False)
                self.file_infos['rois_filepath'].append('not_available')
                self.file_infos['rois_filetype'].append('not_available')
            elif len(matching_roi_filepaths) == 1:
                self.file_infos['rois_present'].append(True)
                self.file_infos['rois_filepath'].append(matching_roi_filepaths[0])
                self.file_infos['rois_filetype'].append(matching_roi_filepaths[0].suffix)
            else:
                raise ValueError('It seems like you provided more than a single ROI file in '
                                 f'{corresponding_dir_in_rois_to_analyze_dir} that matches the microscopy '
                                 f'image filename: {original_filename}. If you want to quantify image features '
                                 'within multiple ROIs per image, please use RoiSets created with ImageJ as '
                                 'described here: [Documentation link not provided yet - please raise an issue on '
                                 'https://github.com/Defense-Circuits-Lab/findmycells - thank you!')

        
    def _add_new_file_history_tracker(self, file_id: int, source_image_filepath: Union[PosixPath, WindowsPath]) -> None:
        file_id = str(file_id).zfill(4)
        self.file_histories[file_id] = FileHistory(file_id = file_id, source_image_filepath = source_image_filepath)
                                               
        
    def _identify_removed_files(self) -> None:
        file_ids_to_remove = []
        for index, microscopy_filepath in enumerate(self.file_infos['microscopy_filepath']):
            if microscopy_filepath.is_file() == False:
                file_ids_to_remove.append(self.file_infos['file_id'][index])
        for file_id in file_ids_to_remove:
            self.remove_file_id_from_project(file_id = file_id)
                

    def get_file_infos(self, file_id: str) -> Dict:
        assert file_id in self.file_infos['file_id'], f'The file_id you passed ({file_id}) is not a valid file_id!'
        index = self.file_infos['file_id'].index(file_id)
        file_infos = {}    
        for key, list_of_values in self.file_infos.items():
            if len(list_of_values) > 0:
                file_infos[key] = list_of_values[index]
        return file_infos
    
    
    def update_file_infos(self, file_id: str, updates: Dict, preferred_empty_value: Union[bool, str, None]=None) -> None: 
        index = self.file_infos['file_id'].index(file_id)
        for key, value in updates.items():
            if key not in self.file_infos.keys():
                self._add_new_key_to_file_infos(key, preferred_empty_value = preferred_empty_value)
            elif len(self.file_infos[key]) != len(self.file_infos['file_id']):
                if len(self.file_infos[key]) == 0:
                    self._add_new_key_to_file_infos(key, preferred_empty_value = preferred_empty_value)
                else:
                    raise ValueError(f'Length of the list stored under the key "{key}" in file_infos '
                                     'does not match with the lenght of the list stored under the key "file_id".')
            self.file_infos[key][index] = value
            
            
    def _add_new_key_to_file_infos(self, key: str, values: Optional[List]=None, preferred_empty_value: Union[bool, str, None]=None) -> None:
        """
        Allows us to add a new key-value-pair to the file_infos dict
        If values is not passed, a list full of 'preferred_empty_value' that matches the length of file_ids will be created
        If values is passed, it has to be a list of the length of file_id
        """
        assert key not in self.file_infos.keys(), f'The key (= {key}) you are trying to add to file_infos is already in file_infos.'
        assert type(values) in [list, type(None)], '"values" has to be either None or a list of values with the same length as file_infos["file_id"].'
        length = len(self.file_infos['file_id'])
        if values == None:
            values = [preferred_empty_value] * length
            self.file_infos[key] = values
        else:
            assert len(values) == length, '"values" has to be either None or a list of values with the same length as file_infos["file_id"].'
            self.file_infos[key] = values
            
            
    def get_file_ids_to_process(self, input_file_ids: Optional[List[str]], processing_step_id: str, overwrite: bool) -> List[str]:
        if input_file_ids == None:
            input_file_ids = self.file_infos['file_id']
        else:
            assert type(input_file_ids) == list, '"input_file_ids" has to be list of file_ids (given as strings)!'
            for elem in input_file_ids:
                assert elem in self.file_infos['file_id'], f'"input_file_ids" has to be list of file_ids (given as strings)! {elem} not a valid file_id!'
        if overwrite == True:
            file_ids_to_process = input_file_ids
        else:
            file_ids_to_process = []
            for file_id in input_file_ids:
                if processing_step_id not in self.file_histories[file_id].completed_processing_steps.keys():
                    file_ids_to_process.append(file_id)
                else:
                    if self.file_histories[file_id].completed_processing_steps[processing_step_id] == False:
                        file_ids_to_process.append(file_id)
        return file_ids_to_process


    def import_rois_dict(self, file_id: str, rois_dict: Dict[str, Dict[str, Polygon]]) -> None:
        if hasattr(self, 'area_rois_for_quantification') == False:
            self.area_rois_for_quantification = {}
        self.area_rois_for_quantification[file_id] = rois_dict


    def remove_file_id_from_project(self, file_id: str) -> None:
        self._remove_file_id_from_file_infos(file_id = file_id)
        self._remove_file_id_from_file_histories(file_id = file_id)
        self._remove_file_id_from_area_rois(file_id = file_id)
        self._remove_file_id_from_quantification_results(file_id = file_id)
        self._delete_all_associated_files_from_processing_subdirs(file_id = file_id)

        
    def _remove_file_id_from_file_infos(self, file_id: str) -> None:
        index = self.file_infos['file_id'].index(file_id)
        for key in self.file_infos.keys():
            self.file_infos[key].pop(index)
    
    
    def _remove_file_id_from_file_histories(self, file_id: str) -> None:
        if hasattr(self, 'file_histories') == True:
            self.file_histories.pop(file_id)
       
    
    def _remove_file_id_from_area_rois(self, file_id: str) -> None:
        if hasattr(self, 'area_rois_for_quantification') == True:
            if file_id in self.area_rois_for_quantification.keys():
                self.area_rois_for_quantification.pop(file_id)
        
        
    def _remove_file_id_from_quantification_results(self, file_id: str) -> None:
        if hasattr(self, 'quantification_results') == True:
            for quantification_strategy_class_name in self.quantification_results.keys():
                if file_id in self.quantification_results[quantification_strategy_class_name].keys():
                    self.quantification_results[quantification_strategy_class_name].pop(file_id)
                    
                    
    def _delete_all_associated_files_from_processing_subdirs(self, file_id: str) -> None:
        for processing_subdir_attr_id in ['preprocessed_images', 'semantic_segmentations', 'instance_segmentations']:
            processing_subdir_name = getattr(self, f'{processing_subdir_attr_id}_dir')
            processing_subdir_path = self.project_configs.root_dir.joinpath(processing_subdir_name)
            self._delete_matching_files_from_subdir(subdir_path = processing_subdir_path, file_id = file_id)
        quantified_segmentations_subdir_path = self.project_configs.root_dir.joinpath(self.quantified_segmentations_dir)
        all_area_id_subdir_paths = utils.list_dir_no_hidden(path = quantified_segmentations_subdir_path, only_dirs = True)
        for area_id_subdir_path in all_area_id_subdir_paths:
            self._delete_matching_files_from_subdir(subdir_path = area_id_subdir_path, file_id = file_id)
            
            
    def _delete_matching_files_from_subdir(self, subdir_path: Union[PosixPath, WindowsPath], file_id: str) -> None:
        all_filepaths_in_subdir = utils.list_dir_no_hidden(path = subdir_path, only_files = True)
        associated_filepaths = [filepath for filepath in all_filepaths_in_subdir if filepath.name.startswith(file_id)]
        for filepath_to_delete in associated_filepaths:
            filepath_to_delete.delete()

            
    def export_quantification_results(self,
                                      export_as: str='xlsx', # 'xlsx' or 'csv'
                                     ) -> None:
        assert hasattr(self, 'quantification_results'), 'No quantification results present yet, nothing to export.'
        assert export_as in ['xlsx', 'csv'], f'"export_as" has to be either "xlsx" or "csv", not {export_as}.'
        quantified_segmentations_dir_path = self.project_configs.root_dir.joinpath(self.quantified_segmentations_dir)
        all_quantified_areas = utils.list_dir_no_hidden(path = quantified_segmentations_dir_path, only_dirs = True)
        for area_roi_id_dir_path in all_quantified_areas:
            area_roi_id = area_roi_id_dir_path.name
            file_ids_with_quantification_results_in_this_area = self._get_file_ids_with_quantification_results_in_area_id(area_id = area_roi_id)
            results_overview = self._get_results_overview_dataframe_for_export(file_ids_to_include = file_ids_with_quantification_results_in_this_area,
                                                                               area_id = area_roi_id)
            results_dir_path = self.project_configs.root_dir.joinpath(self.results_dir)
            filepath = results_dir_path.joinpath(f'quantified_features_in_{area_roi_id}.{export_as}')
            if export_as == 'xlsx':
                results_overview.to_excel(filepath)
            else:
                results_overview.to_csv(filepath)
    

    def _get_file_ids_with_quantification_results_in_area_id(self, area_id: str) -> List[Optional[str]]:
        file_ids_with_quantification_results_in_this_area = []
        for file_id, quantification_results_overview in self.quantification_results['CountFeaturesInWholeAreaROIsStrat'].items():
            if area_id in quantification_results_overview.keys():
                file_ids_with_quantification_results_in_this_area.append(file_id)
        return file_ids_with_quantification_results_in_this_area


    def _get_results_overview_dataframe_for_export(self, file_ids_to_include: List[Optional[str]], area_id: str) -> pd.DataFrame:
        results_overview = {'group ID': [],
                            'subject ID': [],
                            'subgroup ID': [],
                            'file ID in fmc project': [],
                            'microscopy filepath': [],
                            'roi filepath': []}
        for quantification_strategy in self.quantification_results:
            results_overview[f"quantified features {quantification_strategy}"]=[]
        for file_id in file_ids_to_include:
            file_id_specific_file_infos = self.get_file_infos(file_id = file_id)
            for quantification_strategy in self.quantification_results:
                results_overview[f'quantified features {quantification_strategy}'].append(self.quantification_results[quantification_strategy][file_id][area_id])
            results_overview['group ID'].append(file_id_specific_file_infos['main_group_id'])
            results_overview['subject ID'].append(file_id_specific_file_infos['subject_id'])
            results_overview['subgroup ID'].append(file_id_specific_file_infos['subgroup_id'])
            results_overview['file ID in fmc project'].append(file_id)
            results_overview['microscopy filepath'].append(file_id_specific_file_infos['microscopy_filepath'])
            results_overview['roi filepath'].append(file_id_specific_file_infos['rois_filepath'])
        return pd.DataFrame(data = results_overview)

# %% ../nbs/api/02_database.ipynb 5
class FileHistory:
    
    
    def __init__(self, file_id: str, source_image_filepath: Union[PosixPath, WindowsPath]) -> None:
        self.file_id = file_id
        self.source_image_filepath = source_image_filepath
        self.datetime_added = datetime.now()
        self._initialize_tracked_history()
        self._initialize_tracked_settings()
        self._initialize_completed_processing_steps()
        
        
    def _initialize_tracked_history(self) -> None:
        empty_history = {'processing_step_id': [],
                         'processing_strategy': [],
                         'strategy_finished_at': []}
        empty_history_df = pd.DataFrame(data = empty_history)
        setattr(self, 'tracked_history', empty_history_df)
        
        
    def _initialize_tracked_settings(self) -> None:
        setattr(self, 'tracked_settings', {})


    def _initialize_completed_processing_steps(self) -> None:
        setattr(self, 'completed_processing_steps', {})
        
        
    def track_processing_strat(self, processing_step_id: str, processing_strategy_name: str, strategy_configs: Dict) -> None:
        if processing_step_id not in self.completed_processing_steps.keys():
            self.completed_processing_steps[processing_step_id] = False
        tracked_details = {'processing_step_id': [processing_step_id],
                           'processing_strategy': [processing_strategy_name],
                           'strategy_finished_at': [datetime.now()]}
        tracked_details_df = pd.DataFrame(data = tracked_details)
        self.tracked_history = pd.concat([self.tracked_history, tracked_details_df], ignore_index = True)
        self.tracked_settings[self.tracked_history.index[-1]] = strategy_configs
        
    
    def mark_processing_step_as_completed(self, processing_step_id: str) -> None:
        assert processing_step_id in self.completed_processing_steps.keys(), 'This processing step has not been started yet!'
        self.completed_processing_steps[processing_step_id] = True
