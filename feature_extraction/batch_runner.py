import os

class BatchRunner:
'''Runs a job on all files in a specified directory'''

    def __init__():
        self.path = "./"

    def __init__(folder_path):
        self.path = folder_path

    def set_path(folder_path):
        self.path = folder_path

    def run_batch(self, f, save=False, output_ext = ".jpg", output_folder="batch_output"):
        '''
        Runs function f on all files in folder_path
        '''
        for root, dirs, files in os.walk("./enhanced_image"):
            for filename in files:
                out = f(filename)
                if save:
                    outfile = open(filename+output_ext, 'w')
                    outfile.write(filename)
        
