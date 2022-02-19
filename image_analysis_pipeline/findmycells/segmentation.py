from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from PIL import Image
from typing import Tuple, List

from torch.cuda import empty_cache

from deepflash2.learner import EnsembleLearner
from deepflash2.models import get_diameters

from .database import Database

"""

Currently remaining issues:

1.  Due to low memory (Linux distribution), not all files can be processed by the df2 implementation at once (temp files in Linux system tmp directory).
    As work-arround, df2 perdictions will be done on individual batches ( = all single plane images of an original z-stack). 
    This should have no consequences for the regular "binary" segmentations of df2, since the "stats" are calculated on all preprocessed images.
    After initial computation of these stats, however, df2 predictions (binary & instance segmentations) are done on the individual batches.
    This allows to regularly clear all temp files and only consumes some GBs of disk space. The cellpose diameter, however, is calculated on base of all
    masks (binary segmentations) that are present - and for the first batch, these are only the masks of all single planes of file_id 0000.
    
2.  Should make proper use of classes, like in preprocessing.py and quantifications.py. Probably one level less required here, though.
"""

class SegmentationObject:
    
    def __init__(self, database: Database, file_ids: List[str]):
        self.database = database
        self.file_ids = file_ids
        if 'batch_id' in database.file_infos.keys():
            self.batch_id = max(database.file_infos['batch_id']) + 1
        else:
            self.batch_id = 0


    def run_all_segmentation_steps(self):
        for segmentation_strategy in self.database.segmentation_strategies:
            self = segmentation_strategy().run(segmentation_object = self, step = self.database.segmentation_strategies.index(segmentation_strategy))
        
    
    def update_database(self):
        for file_id in self.file_ids:
            self.database.update_file_infos(file_id = file_id, updates = {'batch_id': self.batch_id}, preferred_empty_value = False)
            self.database.update_file_infos(file_id = file_id, updates = {'segmentation_completed': True})                   
        

class SegmentationStrategy(ABC):
    
    @abstractmethod
    def run(self, segmentation_object: SegmentationObject, step: int) -> SegmentationObject:
        # do preprocessing
        segmentation_object.database = self.update_database(database = segmentation_object.database, file_id = segmentation_object.file_ids, step = step)
        return segmentation_object
    
    @abstractmethod
    def update_database(self, database: Database, file_ids: List[str], step: int) -> Database:
        for file_id in file_ids:
            updates = dict()
            updates[f'segmentation_step_{str(step).zfill(2)}'] = 'SegmentationStrategyName'
            # Add additional information if neccessary
            database.update_file_infos(file_id = file_id, updates = updates)
        return database


class Deepflash2BinaryAndInstance(SegmentationStrategy):
    
    def run(self, segmentation_object: SegmentationObject, step: int) -> SegmentationObject:
        if hasattr(segmentation_object.database, 'segmentation_tool_configs') == False:
            segmentation_object.database = self.initialize_deepflash2_as_segmentation_tool(database = segmentation_object.database)
        self.copy_all_files_of_current_batch_to_temp_dir(database = segmentation_object.database, batch_file_ids = segmentation_object.file_ids)
        segmentation_object.database = self.run_binary_segmentations(database = segmentation_object.database)
        segmentation_object.database = self.run_instance_segmentations(database = segmentation_object.database)
        self.move_files(database = segmentation_object.database, batch_file_ids = segmentation_object.file_ids)
        self.delete_temporary_files_and_dirs(database = segmentation_object.database, batch_file_ids = segmentation_object.file_ids)
        segmentation_object.database = self.update_database(database = segmentation_object.database, batch_file_ids = segmentation_object.file_ids, step = step)
        return segmentation_object
    
    
    def initialize_deepflash2_as_segmentation_tool(self, database: Database) -> Database:
        database.segmentation_tool_configs = dict()
        database.segmentation_tool_configs['n_models'] = len([elem for elem in listdir_nohidden(database.trained_models_dir) if elem.endswith('.pth')])
        database.segmentation_tool_configs['ensemble_path'] = database.trained_models_dir
        database.segmentation_tool_configs['stats'] = self.compute_stats(database = database)
        return database
        
        
    def compute_stats(self, database: Database) -> Tuple:
        expected_file_count = sum(database.file_infos['total_planes'])
        actual_file_count = len([image for image in listdir_nohidden(database.preprocessed_images_dir) if image.endswith('.png')])
        if actual_file_count != expected_file_count:
            raise ValueError('Actual and expected counts of preprocessed images don´t match.')
        ensemble_learner = EnsembleLearner(image_dir = database.preprocessed_images_dir.as_posix(), 
                                           ensemble_path = database.segmentation_tool_configs['ensemble_path'])
        stats = ensemble_learner.stats
        del ensemble_learner
        return stats
    
    
    def copy_all_files_of_current_batch_to_temp_dir(self, database: Database, batch_file_ids: List[str]):
        temp_copies_path = database.segmentation_tool_dir.joinpath('temp_copies_of_preprocessed_images')
        for file_id in batch_file_ids:
            files_to_segment = [filename for filename in listdir_nohidden(database.preprocessed_images_dir) if filename.startswith(file_id)]
            if len(files_to_segment) > 0:
                temp_copies_path.mkdir()
                for filename in files_to_segment:
                    filepath_source = database.preprocessed_images_dir.joinpath(filename)
                    shutil.copy(filepath_source.as_posix(), temp_copies_path.as_posix())
    
    
    def run_binary_segmentations(self, database: Database) -> Database:
        image_dir = database.segmentation_tool_dir.joinpath('temp_copies_of_preprocessed_images').as_posix()
        ensemble_learner = EnsembleLearner(image_dir = image_dir,
                                           ensemble_path = database.segmentation_tool_configs['ensemble_path'],
                                           stats = database.segmentation_tool_configs['stats'])
        ensemble_learner.get_ensemble_results(ensemble_learner.files, 
                                              zarr_store = database.segmentation_tool_temp_dir,
                                              export_dir = database.segmentation_tool_dir,
                                              use_tta = True)
        if 'df_ens' not in database.segmentation_tool_configs.keys():
            database.segmentation_tool_configs['df_ens'] = ensemble_learner.df_ens.copy()
        else:
            database.segmentation_tool_configs['df_ens'] = pd.concat([database.segmentation_tool_configs['df_ens'], ensemble_learner.df_ens.copy()])
        del ensemble_learner
        return database
        
    
    def run_instance_segmentations(self, database: Database) -> Database:
        cellpose_diameter = self.get_cellpose_diameter(binary_masks_dir = database.segmentation_tool_dir.joinpath('masks'))
        if 'cellpose_diameter_first_batch' not in database.segmentation_tool_configs.keys():
            database.segmentation_tool_configs['cellpose_diameter_first_batch'] = cellpose_diameter
        database.segmentation_tool_configs['cellpose_diameters_all_batches'].append(cellpose_diameter)

        image_dir = database.segmentation_tool_dir.joinpath('temp_copies_of_preprocessed_images').as_posix()
        empty_cache()
        for row_id in range(database.segmentation_tool_configs['df_ens'].shape[0]):
            ensemble_learner = EnsembleLearner(image_dir = image_dir,
                                               ensemble_path = database.segmentation_tool_configs['ensemble_path'],
                                               stats = database.segmentation_tool_configs['stats'])
            df_single_row = database.segmentation_tool_configs['df_ens'].iloc[row_id].to_frame().T.reset_index(drop=True).copy()
            ensemble_learner.df_ens = df_single_row
            ensemble_learner.cellpose_diameter = database.segmentation_tool_configs['cellpose_diameter_first_batch']
            ensemble_learner.get_cellpose_results(export_dir = database.segmentation_tool_dir)
            del ensemble_learner
            empty_cache()
        return database
        
        
    def get_cellpose_diameter(self, binary_masks_dir: Path) -> float: 
        mask_paths = [binary_masks_dir.joinpath(elem) for elem in listdir_nohidden(binary_masks_dir)]
        masks_as_arrays = []
        for mask_as_image in mask_paths:
            with Image.open(mask_as_image) as image:
                masks_as_arrays.append(np.array(image))
        cellpose_diameter = get_diameters(masks_as_arrays)
        return cellpose_diameter        

    
    def move_files(self, database: Database, batch_file_ids: List[str]):
        binary_masks_path = database.segmentation_tool_dir.joinpath('masks')
        for binary_mask_filename in listdir_nohidden(binary_masks_path):
            filepath_source = binary_masks_path.joinpath(binary_mask_filename)
            shutil.move(filepath_source.as_posix(), database.binary_segmentations_dir.as_posix())
        instance_masks_path = database.segmentation_tool_dir.joinpath('cellpose_masks')
        for instance_mask_filename in listdir_nohidden(instance_masks_path):
            filepath_source = instance_masks_path.joinpath(instance_mask_filename)
            shutil.move(filepath_source.as_posix(), database.instance_segmentations_dir.as_posix())
            
    
    def delete_temporary_files_and_dirs(self, database: Database):
        if hasattr(database, 'low_memory'):
            # This will probably only work with Linux as OS
            if database.low_memory:
                temp_zarr_subdirs = [elem for elem in os.listdir('/tmp/') if 'zarr' in elem]
                if len(temp_zarr_subdirs) > 0:
                    for subdirectory in temp_zarr_subdirs:
                        shutil.rmtree(f'/tmp/{subdirectory}/')
        shutil.rmtree(database.segmentation_tool_temp_dir.as_posix())       
        shutil.rmtree(database.segmentation_tool_dir.joinpath('temp_copies_of_preprocessed_images').as_posix())
        # Reset of df_ens is done in self.update_database() as it resembles an update of the database object
             

    def update_database(self, database: Database, file_ids: List[str], step: int) -> Database:
        for file_id in file_ids:
            updates = dict()
            updates[f'segmentation_step_{str(step).zfill(2)}'] = 'Deepflash2BinaryAndInstance'
            updates['binary_segmentations_done'] = True
            updates['instance_segmentations_done'] = True
            database.update_file_infos(file_id = file_id, updates = updates)
        database.segmentation_tool_configs['df_ens'] = None
        return database
   
    
    







# old code version starts here:
class SegmentationMethod(ABC):
    
    @abstractmethod
    def __init__(self, database: Database, file_ids: List):
        self.database = database
        self.file_ids = file_ids
    
    @abstractmethod
    def run_segmentations(self) -> Database:
        return self.database



class Deepflash2BinarySegmentation(SegmentationMethod):
    # stats can be saved! they just have to be calculated once and then predictions / segmentations can be performed on individual files!!    
    def __init__(self, database: Database):
        self.database = database
        self.image_dir = self.database.deepflash2_dir + 'temp_copies_of_prepro_images/'
        self.ensemble_path = self.database.deepflash2_configs['ensemble_path']
        self.n_models = self.database.deepflash2_configs['n_models']
        self.stats = self.database.deepflash2_configs['stats']
        # Add Warning to log if self.n_models < 3

    
    def run_segmentations(self) -> Database:         
        files_to_segment = os.listdir(self.image_dir)
        
        ensemble_learner = EnsembleLearner(image_dir = self.image_dir, ensemble_path = self.ensemble_path, stats = self.stats)
        ensemble_learner.get_ensemble_results(ensemble_learner.files, 
                                              zarr_store = self.database.deepflash2_temp_dir,
                                              export_dir = self.database.deepflash2_dir,
                                              use_tta = True)

        if self.database.deepflash2_configs['df_ens'] == None:
            self.database.deepflash2_configs['df_ens'] = ensemble_learner.df_ens.copy()
        else:
            self.database.deepflash2_configs['df_ens'] = pd.concat([self.database.deepflash2_configs['df_ens'], ensemble_learner.df_ens.copy()]) # axis?!

        self.database.segmented_file_lists['binary_segmented_files'] = self.database.segmented_file_lists['binary_segmented_files'] + files_to_segment
        del ensemble_learner

        return self.database
        
        
    def clear_temp_data(self):
        # Remove all zarr directories that were created in systems tmp directory:
        temp_zarr_subdirs = [elem for elem in os.listdir('/tmp/') if 'zarr' in elem]
        if len(temp_zarr_subdirs) > 0:
            for subdirectory in temp_zarr_subdirs:
                shutil.rmtree(f'/tmp/{subdirectory}/')



class Deepflash2InstanceSegmentation(SegmentationMethod):
    # stats can be saved! they just have to be calculated once and then predictions / segmentations can be performed on individual files!!
    def __init__(self, database: Database) -> Database:
        self.database = database
        self.image_dir = self.database.deepflash2_dir + 'temp_copies_of_prepro_images/'
        self.ensemble_path = self.database.deepflash2_configs['ensemble_path']
        self.n_models = self.database.deepflash2_configs['n_models']
        self.stats = self.database.deepflash2_configs['stats']
        self.df_ens = self.database.deepflash2_configs['df_ens']
        
    
    def get_cellpose_diameter(self):
        deepflash2_masks_dir = self.database.deepflash2_dir + 'masks/'
        mask_paths = [f'{deepflash2_masks_dir}{elem}' for elem in os.listdir(deepflash2_masks_dir)]
        masks_as_arrays = []
        for mask_as_image in mask_paths:
            with Image.open(mask_as_image) as image:
                masks_as_arrays.append(np.array(image))
        cellpose_diameter = get_diameters(masks_as_arrays)
        return cellpose_diameter
    
    
    def run_segmentations(self) -> Database:
        instance_segmented_files = os.listdir(self.image_dir)
        
        cellpose_diameter = self.get_cellpose_diameter()
        if self.database.deepflash2_configs['cellpose_diameter_first_batch'] == None:
            self.database.deepflash2_configs['cellpose_diameter_first_batch'] = cellpose_diameter
        self.database.deepflash2_configs['cellpose_diameters_all_batches'].append(cellpose_diameter)
        
        empty_cache()
        for row_id in range(self.df_ens.shape[0]):
            ensemble_learner = EnsembleLearner(image_dir = self.image_dir, ensemble_path = self.ensemble_path, stats = self.stats)
            df_single_row = self.df_ens.iloc[row_id].to_frame().T.reset_index(drop=True).copy()
            ensemble_learner.df_ens = df_single_row
            ensemble_learner.cellpose_diameter = self.database.deepflash2_configs['cellpose_diameter_first_batch']
            ensemble_learner.get_cellpose_results(export_dir = self.database.deepflash2_dir)
            del ensemble_learner
            empty_cache()      
        
        self.database.segmented_file_lists['instance_segmented_files'] = self.database.segmented_file_lists['instance_segmented_files'] + instance_segmented_files
        
        return self.database
    
    
    def clear_temp_data(self):
        # Remove all zarr directories that were created in systems tmp directory:
        temp_zarr_subdirs = [elem for elem in os.listdir('/tmp/') if 'zarr' in elem]
        if len(temp_zarr_subdirs) > 0:
            for subdirectory in temp_zarr_subdirs:
                shutil.rmtree(f'/tmp/{subdirectory}/')

        # Remove all zarr directories that were created in designated df2 temp directory:
        shutil.rmtree(self.database.deepflash2_temp_dir[:self.database.deepflash2_temp_dir.rfind('/')])
        

class SegmentationStrategy(ABC):
    
    @abstractmethod
    def run(self, database: Database, file_ids = None) -> Database:
        return database
    
    
class Deepflash2BinaryAndInstanceSegmentationStrategy(SegmentationStrategy):

    def run(self, database: Database, file_ids = None) -> Database:
        self.database = database

        if all(self.database.file_infos['preprocessing_completed']) == False:
            raise TypeError('Not all files have been preprocessed yet! This has to be finished before deepflash2 can be used.')
        
        if hasattr(self.database, 'deepflash2_configs') == False:
            self.database.deepflash2_configs = {'n_models': len([elem for elem in os.listdir(self.database.trained_models_dir) if elem.endswith('.pth')]),
                                                'ensemble_path': Path(self.database.trained_models_dir),
                                                'stats': None,
                                                'df_ens': None,
                                                'cellpose_diameter_first_batch': None,
                                                'cellpose_diameters_all_batches': list()}
            self.database.deepflash2_configs['stats'] = self.compute_stats()
        
        if hasattr(self.database, 'segmented_file_lists') == False:
            self.database.segmented_file_lists = {'binary_segmented_files': list(),
                                                  'instance_segmented_files': list()}
            
        if file_ids == None:
            self.file_ids = self.database.file_infos['file_id']
        else:
            self.file_ids = file_ids
            
        temp_copies_dir = self.database.deepflash2_dir + 'temp_copies_of_prepro_images/'
        for file_id in self.file_ids:
            
            files_to_segment = [filename for filename in os.listdir(self.database.preprocessed_images_dir) if filename.startswith(file_id)]
            files_to_segment = [filename for filename in files_to_segment if filename not in self.database.segmented_file_lists['instance_segmented_files']]
            
            if len(files_to_segment) > 0:
                os.mkdir(temp_copies_dir)
                
                for file in files_to_segment:
                    filepath_source = self.database.preprocessed_images_dir + file
                    shutil.copy(filepath_source, temp_copies_dir)

                df2_binary_seg = Deepflash2BinarySegmentation(self.database)
                self.database = df2_binary_seg.run_segmentations()
                self.database.save_all()

                df2_instance_seg = Deepflash2InstanceSegmentation(self.database)
                self.database = df2_instance_seg.run_segmentations()
                self.database.save_all()

                # Move all files to correct project subdirs:
                filepath_binary_masks = self.database.deepflash2_dir + 'masks/'
                for binary_mask_file in os.listdir(filepath_binary_masks):
                    filepath_source = filepath_binary_masks + binary_mask_file
                    shutil.move(filepath_source, self.database.binary_segmentations_dir)

                filepath_instance_masks = self.database.deepflash2_dir + 'cellpose_masks/'
                for instance_mask_file in os.listdir(filepath_instance_masks):
                    filepath_source = filepath_instance_masks + instance_mask_file
                    shutil.move(filepath_source, self.database.instance_segmentations_dir)

                # Clear all temp data before starting with next file_id
                df2_instance_seg.clear_temp_data()
                self.database.deepflash2_configs['df_ens'] = None
                shutil.rmtree(temp_copies_dir)
        
        return self.database
        
    def compute_stats(self) -> Tuple:
        expected_file_count = sum(self.database.file_infos['total_image_planes'])
        actual_file_count = len([image for image in os.listdir(self.database.preprocessed_images_dir) if image.endswith('.png')])
        
        if actual_file_count != expected_file_count:
            raise ImportError('Actual and expected counts of preprocessed images don´t match.')
            
        ensemble_learner = EnsembleLearner(image_dir = self.database.preprocessed_images_dir, ensemble_path = self.database.deepflash2_configs['ensemble_path'])
        stats = ensemble_learner.stats
        del ensemble_learner
        return stats
            
    
class Segmentor:
    
    def __init__(self, database: Database, file_ids = None):
        self.database = database
        self.file_ids = file_ids
        
    def run_all(self) -> Database:
        self.database = self.database.segmentation_strategy.run(self.database, self.file_ids)
        
        return self.database
        
