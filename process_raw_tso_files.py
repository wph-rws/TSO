import glob
import os
import pathlib
import shutil

import numpy as np
import pandas as pd

from tso_functions import get_location_info, load_project_dirs, determine_filetype

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

def clean_filenames(dirname, donar_code):
    
    for f in dirname.iterdir():    
        
        if f.is_file():        
            if donar_code in f.name.lower():
                splitted_fn = f.name.lower().split('-')
                
                # Filename is in correct format, e.g: GTS-20_20250626.csv or 1148734_GTS-20_20250626.csv
                if len(splitted_fn) > 1:
                    new_name = donar_code.upper() + '-' + splitted_fn[1]
                    
                # Filename is in incorrect format, e.g: gts21.csv 
                elif len(splitted_fn) == 1:
                    
                    # Get file number from filename, without extension
                    loc_num = splitted_fn[0].lower().split('.')[0].replace(donar_code, '')
                    
                    # Read headers and first row of data to get date
                    first_row = pd.read_csv(f, nrows=1, sep=';')
                    mdate = pd.to_datetime(first_row.iloc[0]['DATUM'], format='%Y%m%d')     
                    
                    # Make new filename, e.g.: GTS-20_20250626.csv
                    new_name = donar_code.upper() + '-' + loc_num + '_' + mdate.strftime('%Y%m%d') + '.csv'
                    
                f.rename(f.with_name(new_name))  
   
            else:
                if 'steen' in f.name.lower():
                    new_name = f.name.replace('STEEN', f'{donar_code.upper()}-10')  
                    new_name = donar_code.upper() + new_name.lower().split(donar_code, 1)[1]
                    f.rename(f.with_name(new_name))
            
            print(f'File renamed {f.name}  →  {new_name}')

# i_dir is the position of the subdir in the list of subdirs
def rename_data_files(tso_subdirs, dirname, i_dir):
    
    path = dirname / 'F*.dat'
    files = glob.glob(str(path))
    
    if len(files) == 0:
        raise ValueError(f'No F***.dat files found in directory {dirname}')
        
    new_filenames = []
    for file in files:
        
        print('\nRead ' + file)
        
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
        
# %%

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

# %%
      
def proces_individual_files(location, measurement_date='latest'):
    
    # Location data same for all locations in VZM
    if location == 'vzm':
        _, _, location_data, donar_code = get_location_info('anka')
        tso_locations = ['anka', 'volk', 'zoom']
    else:
        location_code, location_name, location_data, donar_code = get_location_info(location)
        tso_locations = [location]
    
    tso_dir = indiv_files_dir / location_data                
    tso_subdirs = get_subdirs(tso_dir)
    
    # Make list with all subdirs in main dir of location and use the selected subdir    
    if isinstance(measurement_date, str) and measurement_date.lower() == 'latest':
        i_dir = -1 
    elif isinstance(measurement_date, (int, np.integer)) and measurement_date < 10000:
        i_dir = measurement_date
    else:
        try:                       
            matching_dir = tso_subdirs[tso_subdirs['Datum'] == measurement_date]
            
            # Check the number of matches
            num_matches = len(matching_dir)            
            
            if num_matches == 0:
                raise ValueError(f'No matches found for "{measurement_date}"')
            elif num_matches > 1:
                raise ValueError(f'Multiple matches found for "{measurement_date}"')
                
            i_dir = tso_subdirs.iloc[matching_dir.index].index[0]
    
        except ValueError as e:
            print(e)
            return None        
        
    tso_selected_dir = tso_subdirs['Locatie'].iloc[i_dir]
    
    # Determine filetype (dat or csv)
    filetype = determine_filetype(tso_selected_dir)
    
    # Only rename files if filetype is .dat, .csv already has location in its name 
    if filetype == 'dat':    
        rename_data_files(tso_subdirs, tso_selected_dir, i_dir)
        
        # Creation of naarmat-file only for *.dat files, *.csv will not work
        new_date = tso_subdirs['Datum'].iloc[-1]
        naarmat_filename = [f'_Naarmat{location}' for location in tso_locations]
        change_naarmat_file(tso_dir, tso_selected_dir, naarmat_filename, new_date)
        
    elif filetype == 'csv':
        clean_filenames(tso_selected_dir, donar_code)