import os

from .database import Database
from .preprocessing import Preprocessor
from .segmentation import Segmentor

class Project:
    def __init__(self, user_input: dict):
        self.root_dir = user_input['project_root_dir']
        self.database = Database(user_input)
        
    def save_status(self):
        self.database.save_all()
    
    def load_status(self):
        self.database.load_all()
    
    def preprocess(self, file_ids = None):
        if 'preprocessing_completed' not in self.database.file_infos.keys():
            self.database.add_new_key_to_file_infos('preprocessing_completed')
        if file_ids == None:
            all_file_ids = self.database.file_infos['file_id']
            preprocessing_infos = self.database.file_infos['preprocessing_completed']
            file_ids = [elem[0] for elem in zip(all_file_ids, preprocessing_infos) if elem[1] == False or elem[1] == None]
        preprocessor = Preprocessor(file_ids, self.database)
        self.database = preprocessor.run_individually()
    
    def run_segmentation(self):
        segmentor = Segmentor(self.database)
        self.database = segmentor.run_all()
    
    def run_quantificatons(self):
        pass
        
        
      
        

        
        
        
        
        
        
        


            
            
            
