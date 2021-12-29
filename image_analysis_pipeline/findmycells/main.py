import os

from .database import Database
from .preprocessing import Preprocessor

class Project:
    def __init__(self, user_input: dict):
        self.root_dir = user_input['project_root_dir']
        self.database = Database(user_input)
        
    def save_status(self):
        self.database.save_all()
    
    def load_status(self):
        self.database.load_all()
    
    def preprocess(self, file_ids):
        if 'preprocessing_completed' not in self.database.file_infos.keys():
            self.database.add_new_key_to_file_infos('preprocessing_completed')
        preprocessor = Preprocessor(file_ids, self.database)
        self.database = preprocessor.run_individually()
    
    def run_segmentation(self):
        pass
    
    def run_quantificatons(self):
        pass
        
        
      
        

        
        
        
        
        
        
        


            
            
            
