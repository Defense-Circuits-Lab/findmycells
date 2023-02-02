# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/07_postprocessing_00_specs.ipynb.

# %% auto 0
__all__ = ['PostprocessingStrategy', 'PostprocessingObject']

# %% ../../nbs/07_postprocessing_00_specs.ipynb 2
from abc import abstractmethod
from typing import Dict, List
from skimage import io

from ..core import ProcessingObject, ProcessingStrategy
from ..configs import DefaultConfigs
from .. import utils

# %% ../../nbs/07_postprocessing_00_specs.ipynb 4
class PostprocessingStrategy(ProcessingStrategy):
    
    """
    Extending the `ProcssingStrategy` base class for postprocessing as processing subtype.
    """
    
    @property
    def processing_type(self):
        return 'postprocessing' 

# %% ../../nbs/07_postprocessing_00_specs.ipynb 5
class PostprocessingObject(ProcessingObject):
    
    """
    Extending the `ProcessingObject` base class for postprocessing as processing subtype.
    """
    
    @property
    def processing_type(self):
        return 'postprocessing'
    
    @property
    def widget_names(self):
        widget_names = {'segmentations_to_use': 'Dropdown',
                        'overwrite': 'Checkbox',
                        'autosave': 'Checkbox',
                        'show_progress': 'Checkbox'}
        return widget_names

    @property
    def descriptions(self):
        descriptions = {'segmentations_to_use': 'continue with semantic or instance segmentations',
                        'overwrite': 'overwrite previously processed files',
                        'autosave': 'autosave progress after each file',
                        'show_progress': 'show progress bar and estimated computation time'}
        return descriptions
    
    @property
    def tooltips(self):
        return {} 
    
    @property
    def default_configs(self) -> DefaultConfigs:
        default_values = {'segmentations_to_use': 'instance',
                          'overwrite': False,
                          'autosave': True,
                          'show_progress': True}
        valid_types = {'segmentations_to_use': [str],
                       'overwrite': [bool],
                       'autosave': [bool],
                       'show_progress': [bool]}
        valid_options = {'segmentations_to_use': ('semantic', 'instance')}
        default_configs = DefaultConfigs(default_values = default_values,
                                         valid_types = valid_types,
                                         valid_value_options = valid_options)
        return default_configs
    
    
    def _processing_specific_preparations(self) -> None:
        self.file_id = self.file_ids[0]
        self.file_info = self.database.get_file_infos(file_id = self.file_id)
        self.rois_dict = self.database.area_rois_for_quantification[self.file_id]
        self.segmentations_per_area_roi_id = {}
        
        
    def load_segmentations_masks_for_postprocessing(self, segmentations_to_use: str) -> None:
        assert segmentations_to_use in ['semantic', 'instance'], f'"segmentations_to_use" has to be either "semantic" or "instance", not {segmentations_to_use}!'
        if segmentations_to_use == 'semantic':
            masks_dir_path = self.database.project_configs.root_dir.joinpath(self.database.semantic_segmentations_dir)
        else:
            masks_dir_path = self.database.project_configs.root_dir.joinpath(self.database.instance_segmentations_dir)
        self.postprocessed_segmentations = utils.load_zstack_as_array_from_single_planes(path = masks_dir_path, file_id = self.file_id)
            
    
    def save_postprocessed_segmentations(self) -> None:
        for area_roi_id in self.segmentations_per_area_roi_id.keys():
            for plane_index in range(self.segmentations_per_area_roi_id[area_roi_id].shape[0]):
                image = self.segmentations_per_area_roi_id[area_roi_id][plane_index]
                target_dir_path = self.database.project_configs.root_dir.joinpath(self.database.quantified_segmentations_dir, area_roi_id)
                if target_dir_path.is_dir() == False:
                    target_dir_path.mkdir()
                filepath = target_dir_path.joinpath(f'{self.file_id}-{str(plane_index).zfill(3)}_postprocessed_segmentations.png')
                io.imsave(filepath, image, check_contrast=False)


    def _add_processing_specific_infos_to_updates(self, updates: Dict) -> Dict:
        return updates
