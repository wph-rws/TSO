import pandas as pd
import datetime

def read_individual_files(location, filelist):

    # Create empty variables
    fn_short = []    
    raw_text_files = []
    width_df = []
    header_rows = {}
    loc_empty_line = {}
    measurement_date = 0

    # Define measurment info
    measurement_info = {
        'loc_names' : [],
        'meetvaartuig' : [],
        'meetdatum': [],
        'meetgebied' : [],
        'startpunt' : [],
        'eindpunt' : [],
        'projectcode' : [],
        'opdrachtgever' : [],
        'beheerder' : [],
        'bemonst. inst': [],
        'waarnemer(s)': [],
        }
    
    # Read all individual files
    for i, filename in enumerate(filelist['filename']):
        
        fn_short = filelist['fn_short'].iloc[i]
    
        with open(filename, 'r') as f:
            raw_text = f.readlines()
            raw_text_files.append(raw_text)
    
            for num, line in enumerate(raw_text):
    
                # Set all strings to lower case for matching
                line = line.lower()
    
                # Number of measurement point
                if '#00 donarlocatie:' in line:                
                   
                    # Clean string to be able to append named locations
                    cleaned_line = line.replace('#00 donarlocatie:', '').strip().upper()
                    
                    # Exception for Grevelingen
                    if 'SCH' in cleaned_line:
                        cleaned_line = 'SCH'
                    
                    # Split number from string
                    measurement_info['loc_names'].append(cleaned_line.split('-')[-1])                   
                                        
                if '# vertikaal' in line:
                    # Split string, lose first 3 items, join last ones
                    measurement_info['meetvaartuig'].append(' '.join(line.split()[3:]).upper())
                    
                # Measurement date:
                if '*' in line:
                    datetime_string = line.split('*')[-1].strip()                     
                    try:
                        measurement_date = datetime.datetime.strptime(datetime_string, '%d-%m-%Y %H:%M:%S')
                        measurement_info['meetdatum'].append(measurement_date)
                    except ValueError:
                        continue                                       
                 
                # Observer
                if '#08 waarnemer' in line:          
                    measurement_info['waarnemer(s)'].append((line.split(':')[-1]).strip().capitalize())
                    
                # Measurement location
                if '#21 meetgebied' in line:
                    measurement_info['meetgebied'].append((line.split(':')[-1]).strip().capitalize())
                    
                # Measurement location
                if '#22 startpunt' in line:
                    measurement_info['startpunt'].append((line.split(':')[-1]).strip())
                        
                # Measurement location
                if '#23 eindpunt' in line:
                    measurement_info['eindpunt'].append((line.split(':')[-1]).strip())
                    
                # Projectcode
                if '#06 projectcode' in line:
                    measurement_info['projectcode'].append((line.split(':')[-1]).strip())                        
                    
                # Measurement location
                if '#51 opdrachtgever' in line:
                    measurement_info['opdrachtgever'].append((line.split(':')[-1]).strip())                    

                # Measurement location
                if '#52 beheerder' in line:
                    measurement_info['beheerder'].append((line.split(':')[-1]).strip())          
    
                # Find location of first empy line
                if line in ['\n', '\r\n']:
                    loc_empty_line[fn_short] = num
    
                    # First character is a ~ which messes up column names
                    header_row = raw_text[loc_empty_line[fn_short]+1].split()[1:]
    
                    # Replace 'T' with 'Temp'
                    header_row = ['Temp' if item == 'T' else item for item in header_row]
                    width_df.append(len(header_row))
    
                    # Save header row of individual file
                    header_rows[fn_short] = header_row
    
                    # Break because empty line has been found and header rows extracted
                    break
    
            # # Check if headers are the same (now disabled because in VZM there can be a difference in number of columns)
            # if i > 0 and header_rows[i-1] != header_rows[i]:
            #     raise ValueError(f'Header row "{filelist[i]}" is not equal to header row "{filelist[i-1]}"')         
             
    # Count occurences and select most frequently mentioned one to display in combined datafile
    to_count = list(measurement_info.keys())
    for item in to_count:
        if item not in  ('loc_names', 'meetdatum') and measurement_info[item]:
            measurement_info[item] = pd.DataFrame(measurement_info[item]).value_counts().index[0]
    
    return measurement_info, header_rows, loc_empty_line