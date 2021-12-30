from abc import ABC, abstractmethod
import os
import numpy as np
from pathlib import Path
import shutil
from PIL import Image

from torch.cuda import empty_cache

from deepflash2.learner import EnsembleLearner
from deepflash2.models import get_diameters

from .database import Database


class SegmentationMethod(ABC):
    
    @abstractmethod
    def __init__(self, database: Database):
        self.database = database
    
    @abstractmethod
    def run_segmentations(self) -> Database:
        return self.database




class Deepflash2BinarySegmentation(SegmentationMethod):
    # stats can be saved! they just have to be calculated once and then predictions / segmentations can be performed on individual files!!    
    def __init__(self, database: Database) -> Database:
        self.database = database
        self.ensemble_path = Path(self.database.trained_models_dir)
        self.n_models = len([elem for elem in os.listdir(self.database.trained_models_dir) if elem.endswith('.pth')])
        # Add Warning to log if self.n_models < 3

    
    def run_segmentations(self) -> Database:
        binary_segmented_files = os.listdir(self.database.preprocessed_images_dir)
        ensemble_learner = EnsembleLearner(image_dir = self.database.preprocessed_images_dir, ensemble_path = self.ensemble_path)
        self.database.deepflash2_binary_segmentation_stats = ensemble_learner.stats
        ensemble_learner.get_ensemble_results(ensemble_learner.files, 
                                              zarr_store = self.database.deepflash2_temp_dir,
                                              export_dir = self.database.deepflash2_dir,
                                              use_tta = True)
        self.database.deepflash2_binary_segmentation_df_ens = ensemble_learner.df_ens.copy()
        if hasattr(self.database, 'segmented_file_lists') == False:
            self.database.segmented_file_lists = dict()
        self.database.segmented_file_lists['binary_segmented_files'] = binary_segmented_files
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
        self.ensemble_path = Path(self.database.trained_models_dir)
        self.df_ens_temp = self.database.deepflash2_binary_segmentation_df_ens.copy()
        
    
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
        instance_segmented_files = os.listdir(self.database.preprocessed_images_dir)
        self.database.deepflash2_cellpose_diameter = self.get_cellpose_diameter()
        empty_cache()
        for row_id in range(self.df_ens_temp.shape[0]):
            ensemble_learner = EnsembleLearner(image_dir = self.database.preprocessed_images_dir, ensemble_path = self.ensemble_path)
            df_single_row = self.df_ens_temp.iloc[row_id].to_frame().T.reset_index(drop=True).copy()
            ensemble_learner.df_ens = df_single_row
            ensemble_learner.cellpose_diameter = self.database.deepflash2_cellpose_diameter
            ensemble_learner.get_cellpose_results(export_dir = self.database.deepflash2_dir)
            del ensemble_learner
            empty_cache()
        if hasattr(self.database, 'segmented_file_lists') == False:
            self.database.segmented_file_lists = dict()        
        self.database.segmented_file_lists['instance_segmented_files'] = instance_segmented_files
        
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
    def run(self, database: Database) -> Database:
        return database
    
    
class Deepflash2BinaryAndInstanceSegmentationStrategy(SegmentationStrategy):
    # stats can be saved! they just have to be calculated once and then predictions / segmentations can be performed on individual files!!    
    def run(self, database: Database) -> Database:
        # Should check whether all file_ids are already tagged with "preprocessing_completed" == True !
        self.database = database
        df2_binary_seg = Deepflash2BinarySegmentation(self.database)
        self.database = df2_binary_seg.run_segmentations()
        df2_instance_seg = Deepflash2InstanceSegmentation(self.database)
        self.database = df2_instance_seg.run_segmentations()
        df2_instance_seg.clear_temp_data()
        
        # Has to take care that the files end up in the correct directories! -> binary_segmentations & instance_segmentations, respectively
        return self.database
        
 
    
class Segmentor:
    
    def __init__(self, database: Database):
        self.database = database
        
    def run_all(self) -> Database:
        self.database = self.database.segmentation_strategy.run(self.database)
        
        return self.database
        
