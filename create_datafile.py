from tso_functions import generate_filelist, write_datafile

"""
This script processes measurement data by generating a file list and then writing
a corresponding data file for one or more measurement locations. Two main functions are used:
    - generate_filelist
    - write_datafile

Usage details:
1. Select one or more locations by modifying the 'locations' list. For example:
       locations = ['grev']
   You can also process multiple locations at once, e.g.,
       locations = ['anka', 'volk', 'zoom']
       
2. Measurement Date Options:
   - If no measurement date is provided, the functions will use the latest available measurement
     from the directory with the datafiles for the given location.
   - To process a specific measurement date, you can:
       a) Provide the date as a string in the format 'YYYYMMDD', e.g.,
              measurement_date = '20240404'
       b) Provide an integer indicating the measurement order:
              - Use -1 for the latest measurement,
              - Use -2 for the second to last, etc.
          Example: measurement_date = -2

Script Structure:
-----------------------------------------------------------
# Block 1: Process each selected location with the default (latest) measurement.

-----------------------------------------------------------
# Block 2: Process using a specific measurement date provided as an integer.
    Example: measurement_date = -2 is the second to last measurement

-----------------------------------------------------------
# Block 3: Example of processing a specific location with a specific date.

"""

#%%

# locations = ['anka']
locations = ['grev']
# locations = ['grno']
# locations = ['kvgt']
# locations = ['veer']
# locations = ['volk']
# locations = ['zoom']

# locations = ['anka', 'volk', 'zoom']
# locations = ['anka', 'kvgt', 'grev', 'grno', 'veer', 'volk', 'zoom']
# measurement_date = '20240404'

#%%

for location in locations:
    filelist = generate_filelist(location)
    write_datafile(location, filelist)

#%%

measurement_date = -2

for location in locations:
    filelist = generate_filelist(location, measurement_date)
    write_datafile(location, filelist, measurement_date)

# %% With measurement date
# location = 'veer'
# measurement_date = '20240731'
# filelist = generate_filelist(location, measurement_date)
# write_datafile(location, filelist, measurement_date)