import glob
import os
import pandas as pd
import pathlib
import shutil
from tso_functions import get_location_info, load_project_dirs

# Load directories
project_dirs = load_project_dirs()
indiv_files_dir = pathlib.Path(project_dirs.get('individual_files_dir'))

#%% Script for renaming individual TSO source files

def get_subdirs(dirname):
    datum = []
    locatie = []
    for item in dirname.iterdir():
        if item.is_dir():
            datum.append(item.name)
            locatie.append(item)
    
    output = pd.DataFrame(list(zip(datum, locatie)), columns=['Datum', 'Locatie']) \
               .sort_values(by='Datum') \
               .reset_index(drop=True)
    return output

# i_dir is the position of the subdir in the list of subdirs
def rename_data_files(tso_subdirs, dirname, i_dir):
    path = dirname / 'F*.dat'
    files = glob.glob(str(path))
    
    if len(files) == 0:
        raise ValueError(f'No F***.dat files found in directory {dirname}')
        
    new_filenames = []
    for file in files:
        
        print('\nInlezen ' + file)
        
        donar_found = 0
        
        with open(file, 'r') as f:
            raw_text = f.readlines()
            
        for num, line in enumerate(raw_text):
            # set all strings to lower case for matching
            line = line.lower()
        
            if 'donar' in line:
                donar_found = 1
                print(line.strip())
                splitted_line = line.split()
                loc_code = splitted_line[-1].split('-')[0]
                loc_number = splitted_line[-1].split('-')[-1]
                dir_name = tso_subdirs['Locatie'].iloc[i_dir]
                
                # Steenbergen (Volkerak)
                if loc_number == 'steenbgn':
                    loc_number = '10'
                    
                # Oesterdam (Zoommeer)
                if loc_number == 'oestdm':
                    loc_number = '36'
                
                # Scharendijke (Grevelingen)
                if loc_number == 'schardkdppt':
                    loc_number = '03'
                    
                # Dreischor (Grevelingen)
                if loc_number == 'dreis' or loc_number == 'dreisr':
                    loc_number = '13'
                    
                # Herkingen (Grevelingen)
                if loc_number == 'herkgn':
                    loc_number = '16'
                
                # Extra measurment points in VM with name 'z-01'
                if loc_code.lower() == 'z':
                    loc_number = f'{loc_code}_{loc_number}'
                    
                # zero fill loc_number to always print leading zeros
                loc_number = loc_number.zfill(4)
                
                # first rename with n in front of name, otherwise
                # files will be renamed to existing filenames that still
                # need to be renamed and therefore will be overwritten/skipped
                new_filename = dir_name / f'F{loc_number}.dat'
                
                if new_filename == file:
                    print('Filename is correct already, file not renamed')
                else:                         
                    new_filename_n = dir_name / f'nF{loc_number}.dat'
                    
                    if not new_filename_n.exists(): 
                        shutil.copy(file, new_filename_n)                        
                    else:
                        raise ValueError(f'File {new_filename} already exists, file not renamed. Check duplicate location names in source files')
                        
                    os.remove(file)  
                    print('File renamed to: ' + str(new_filename))
                                        
                new_filenames.append(new_filename)
                
        if donar_found == 0:
            print(f'Donar location not found, file {file} not renamed')
            
    path_n = dirname / 'nF*.dat'
    files_n = glob.glob(str(path_n))
    
    # final renaming
    for file_n in files_n:
        new_filename = file_n.replace('nF', 'F')
        new_filename_path = pathlib.Path(new_filename)
        os.rename(file_n, new_filename_path)  

def change_naarmat_file(tso_dir, tso_last_dir, naarmat_filenames, new_date):
    for naarmat_filename in naarmat_filenames:
        source = tso_dir / naarmat_filename
        
        naarmat_filename = naarmat_filename[1:]

        destination = tso_last_dir / naarmat_filename
        shutil.copy(source, destination)
        
        with open(destination, 'r') as f:
            raw_text = f.readlines()
            
        for num, line in enumerate(raw_text):
            if 'DATUM=' in line:
                old_date = line.split('=')[-1].strip()
                new_line = line.replace(old_date, new_date)
                raw_text[num] = new_line                  
                    
        with open(destination, 'w', newline='') as f:
            f.writelines(raw_text)
            print(f'\n{old_date} replaced by {new_date} in {naarmat_filename}')            
       
def proces_individual_files(location, date='latest'):
    
    # Location data same for all locations in VZM
    if location == 'vzm':
        _, _, location_data = get_location_info('anka')
        tso_locations = ['anka', 'volk', 'zoom']
    else:
        location_code, location_name, location_data = get_location_info(location)
        tso_locations = [location]
    
    tso_dir = indiv_files_dir / location_data                
    tso_subdirs = get_subdirs(tso_dir)
    
    # maak een lijst met alle subdirs in hoofddir van locatie en gebruik de meest recente
    tso_last_dir = tso_subdirs['Locatie'].iloc[-1]
    
    rename_data_files(tso_subdirs, tso_last_dir, -1)
    
    new_date = tso_subdirs['Datum'].iloc[-1]
    naarmat_filename = [f'_Naarmat{location}' for location in tso_locations]
    change_naarmat_file(tso_dir, tso_last_dir, naarmat_filename, new_date)