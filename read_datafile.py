import numpy as np
import pandas as pd
import collections
from io import StringIO 
from tso_functions import MPNaam, find_filename_datafile

#%% Read data file of measurement

def read_datafile(location, measurement_date='latest', ignored_points={}, plot_mode='single'):    
   
    # Find filename
    filename = find_filename_datafile(location, measurement_date, plot_mode)
    
    # Read file mpnaam
    mpnaam = MPNaam(location)

    startrows = []    
    df_colnames = collections.OrderedDict()
    df_colnames_raw = collections.OrderedDict()
    meetdatum = []
    date_found = 0    

    cols_raw = []
    cols = []
    
    no_data_in_file = False
  
    with open(filename, 'r') as f:
        raw_text = f.readlines()       
        
        for num, line in enumerate(raw_text):
            
            # set all strings to lower case for matching
            line = line.lower()
            
            # splits regel in delen
            splitted_line = line.split()      
            
            # if-statement to define column names
            if '>' in splitted_line[0]:
                
                # raw column names, useful for debugging
                cols_raw.append(splitted_line)
                
                # determine actual column names   
                # force lines with x and y in it to same column name
                if '>x' in line:
                    cols.append('x-coord')
                elif '>y' in line:
                    cols.append('y-coord')
                # force line with time in it to same column name
                elif '>tijd' in line:
                    cols.append('tijd')
                elif 'vaarsnelheid' in line:
                    cols.append('vaarsnelheid')
                elif 'insitu troeb' in line:
                    cols.append('insitu troebelheid')
                elif 'echolood' in line:
                    cols.append('echolood')
                elif 'dichtheid' in line:
                    cols.append('dichtheid')
                 # catch not defined parameters
                elif 'nog niet gedefinieerd' in line:
                    cols.append(splitted_line[0].strip('>'))
                # if length string == 1, then that text as column name
                elif len(splitted_line) == 1:
                    cols.append(line.split('>')[1].rstrip())
                # older datafiles of measurments (around 1995) can have two types of
                # oxygen mg/l    
                elif '300.5' in splitted_line[1]:
                    max_len_line = np.min([len(splitted_line) - 1, 5])
                    cols.append(splitted_line[2] + '_oud_' + splitted_line[max_len_line])
                # oxygen percentage    
                elif 'o2%' in splitted_line[0] or '%o2' in splitted_line[0]:
                    cols.append('zuurstof_perc')
                # there can be two types of echoscope: ship and sensor    
                elif 'ech' in splitted_line[0]:
                    cols.append(splitted_line[0][1:])
                # there can be two types of conductivity (geleidendheid)   
                elif 'gel' in splitted_line[0]:
                    if len(splitted_line) > 2:
                        cols.append(splitted_line[0][1:] + '_' + splitted_line[1])
                    else:
                        cols.append(splitted_line[1].rstrip('\n'))
                # if not catched by previous code, then column name is second part of string
                else:
                    cols.append(splitted_line[1].rstrip('\n'))  
            if '*' in splitted_line[0]:
                if len(splitted_line) > 1:            
                    # print(f'Datum is {splitted_line[1]}')   
                    meetdatum.append(splitted_line[1])
                    date_found = 1
                    
            # if-statement om start data te bepalen
            if splitted_line[0].startswith('-'):
                # data moet minimaal 7 kolommen breed zijn
                if len(splitted_line) >= 7:
                    loc_start_data = num
                    print(f'Reading file "{filename}"')
                    startrows.append(loc_start_data)
                    
                    break
                
                else:
                    print(f"File {filename} doesn't have enough columns, file will be skipped")
                    no_data_in_file = True
                    break
           
        # Go the next file if not enough data in file (not enough columns)            
        if no_data_in_file:    
            raise ValueError(f'Not enough data in file {filename}')             
                    
        # Splits data from header
        raw_text_body = raw_text[loc_start_data:]
        
        # Filter rows that start with '#'
        filtered_text_body = [line for line in raw_text_body if not line.startswith('#')]
        
        # Data back to str for reading with StringIO
        filtered_text_pandas = '\n'.join(filtered_text_body)
                    
        if date_found == 0:
            print('Date not found')
            meetdatum.append(np.nan)
        
        df_colnames[filename] = cols
        df_colnames_raw[filename] = cols_raw   
    
        try:
            df = pd.read_csv(StringIO(filtered_text_pandas), sep='\s+', header=None, na_values=9999)
        except Exception as e:
            print(f'First try reading data failed, error:\n{e} \nTry againt with adjusted number of columns')
            
        try:
            max_columns = max(len(line.split()) for line in StringIO(filtered_text_pandas))
            df = pd.read_csv(StringIO(filtered_text_pandas), sep='\s+', header=None, na_values=9999, names=np.arange(0, max_columns))     
        except Exception as e:
            raise ValueError(f'Second try reading data also failed, error:\n{e}')
        
        if len(df.columns) > len(cols):
            a = len(df.columns)
            b = len(cols)
            print(f'Number of columns of data [{a}] is larger than number of column names[{b}]')
            
            for i in range(len(df.columns)):
                if df[df.columns[i]].isnull().values.all() == 1:                
                    df.drop([df.columns[i]], inplace=True, axis=1)
                    print(f'Column {i} removed because all NaN')
        
        # Set column names                    
        df.columns = cols
        
        # Drop duplicate rows, there are a some datafiles where there is 
        # duplicate data, for example in 'gr20230711' and 'gr20230810'
        df = df.drop_duplicates(keep='first')
        
        # Make date array with length of number of rows with measurement values
        # based on the last measurement date
        datum_array = pd.to_datetime(np.repeat(meetdatum[-1], len(df)))
        df.insert(0, 'datum', datum_array)        

        # Create intervals for number of measurements per measurement location
        # times that start with '-' indicate start of measurement location
        df['tijd'] = df['tijd'].astype(int)
        start = df.index[df['tijd'] <= 0].values
        end = np.append(start[1:], df.index[-1] + 1)
        value = np.arange(1, len(start) + 1)
        
        df_interval = pd.DataFrame(data=[start, end, value]).T
        df_interval.columns = ['start', 'end', 'value']         

        # Make series with intervals based on values from value array
        intervals = df_interval.set_index(pd.IntervalIndex.from_arrays(df_interval['start'], df_interval['end'], closed='left'))['value']
         
        # Make dataframe with length of measurements + 1
        mp_values = pd.DataFrame(np.arange(end[-1] + 1))
         
        # Map values from intervals to values (numbers) of measurement locations
        mp_values['value'] = mp_values[0].map(intervals)         
        df.insert(1, 'meetpunt', mp_values['value'].iloc[0:-1])
         
        # remove '-' by taking absolute value and round depth to precision of 0.25m
        df['tijd'] = np.abs(df['tijd'])        
        df['diepte'] = np.round((df['diepte'] * 4), 0) / 4                        
                
        df['berekend_meetpunt'] = pd.Categorical(
            [None] * len(df),  # Create a list of None values with the same length as df
            categories=mpnaam.df['loc_name'].tolist(),
            ordered=True
        )   
        
        # Initialize an empty list to hold 'meetpunt' numbers to remove if 
        # distance to nearest predefined point is too large
        mp_nummers_to_remove = []      
        
        dist_lim = 500  # distance limit in meters
        
        # Check coordinates for each unique measurement point
        for mp_nummer in df['meetpunt'].unique():            
            
            # Calculate mean coordinate of measurement per point
            x_temp = df['x-coord'][df['meetpunt'] == mp_nummer].mean()
            y_temp = df['y-coord'][df['meetpunt'] == mp_nummer].mean()
            
            abs_dist = ((x_temp - mpnaam.df['x']) ** 2 + (y_temp - mpnaam.df['y']) ** 2) ** 0.5
            closest_loc_name = mpnaam.df['loc_name'].loc[abs_dist.idxmin()]
            
            # Correction for DREIS/DREI because DREI is not in mpnaam
            if abs_dist.min() > dist_lim and closest_loc_name.lower() == 'dreis':
                print('Measurement at location "DREI" instead of "DREIS"')
                df.loc[df['meetpunt'] == mp_nummer, 'berekend_meetpunt'] = 'DREIS'
            
            elif abs_dist.min() > dist_lim:
                
                # Check if this measurement point has already been marked as "ignore distance limit"
                if mp_nummer in ignored_points and ignored_points[mp_nummer]:
                    df.loc[df['meetpunt'] == mp_nummer, 'berekend_meetpunt'] = closest_loc_name
                    print(f'Point {int(mp_nummer)} already set to ignore distance limit, kept as {closest_loc_name}.')
                else:
                    # Ask the user if they want to ignore the distance limit for this point
                    response = input(f'Distance for point {int(mp_nummer)} with x,y = ({int(x_temp)}, {int(y_temp)}) to nearest point {closest_loc_name} is {int(abs_dist.min())} m (> {dist_lim} m). Ignore limit for this point for all parameters? (y/n): ')
                    if response.strip().lower() in ['y', 'yes']:
                        
                        # Remember that this point should ignore the limit in the future
                        ignored_points[mp_nummer] = True
                        df.loc[df['meetpunt'] == mp_nummer, 'berekend_meetpunt'] = closest_loc_name
                        print(f'Point {int(mp_nummer)} kept in data (ignoring distance limit).')
                    else:        
                        mp_nummers_to_remove.append(mp_nummer)
                        print(f'Point {int(mp_nummer)} removed from data because distance ({int(abs_dist.min())} m) exceeds limit.')
            else:
                df.loc[df['meetpunt'] == mp_nummer, 'berekend_meetpunt'] = closest_loc_name
                
        # Remove the rows that exceed the distance limit
        df = df[~df['meetpunt'].isin(mp_nummers_to_remove)]
               
        # Sort dataframe
        df.sort_values(by=['berekend_meetpunt', 'diepte'], inplace=True)
        df['berekend_meetpunt'] = df['berekend_meetpunt'].astype(str)
        
        # Move 'berekend_meetpunt' to 3rd position
        cols = df.columns.tolist()
        cols.insert(2, cols.pop(cols.index('berekend_meetpunt')))
        df = df[cols]
        
        # Find 'berekend meetpunt' with multiple values for 'meetpunt',
        # which means same coordinates for two or more points in datafile
        counts = df.groupby('berekend_meetpunt')['meetpunt'].nunique()
        problematic = counts[counts > 1]
        
        if not problematic.empty:
            error_message = "Error: The following 'berekend_meetpunt' values have multiple 'meetpunt' values:"
            for problem_points in problematic.index:
                sub_df = df[df['berekend_meetpunt'] == problem_points]
                problem_points_coords = sub_df.groupby('meetpunt')[['berekend_meetpunt', 'tijd', 'x-coord', 'y-coord']].first()
                   
                error_message = "Error: The following 'berekend_meetpunt' values have multiple 'meetpunt' values:\n"
                error_message += problem_points_coords.to_string()
                
                print(error_message)               
                raise ValueError("Data contains conflicting 'meetpunt' values for the same 'berekend_meetpunt'.")
        
        # Find missing loc_names, convert to sets for efficient comparison
        berekend_meetpunt_set = set(df['berekend_meetpunt'].astype(str))
        loc_name_set = set(mpnaam.df['loc_name'].astype(str))

        missing_loc_names = loc_name_set - berekend_meetpunt_set
        
        # Remove missing locations from mpnaam
        mpnaam.df = mpnaam.df[~mpnaam.df['loc_name'].astype(str).isin(missing_loc_names)]
        
        # Catch missing locations Z1, Z2 and Z3 for Veerse Meer because almost
        # always missing
        z_locations_missing = any(name.startswith('Z') for name in missing_loc_names)
        
        if missing_loc_names and not z_locations_missing:
            missing_locations = ', '.join(map(str, missing_loc_names))
            print(f'Location(s): "{missing_locations}" removed from mpnaam')

        meetdatum = meetdatum[0]        
             
    return df, meetdatum, mpnaam, ignored_points
