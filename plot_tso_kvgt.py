"""
For processing a new measurement, first run 'rename_raw_tso_files.py' and then 
create a new datafile with 'create_datafile.py'

If the function is called without measurement date, the latest available
measurement date is chosen, to chose a specific measurement date, 
two options are available:
    - Specify the date, for example : measurement_date = '20240601'
    - Specify an integer, where -1 is the latest, -2 is the second latest, etc.,
      example: measurement_date = -1
    
If plot_mode 'multiple' is used, the latest six measurements are used

"""

from plot_measurement import process_plots

location = 'kvgt'
process_plots(location)

#%%

location = 'kvgt'
process_plots(location, plot_mode='multiple')