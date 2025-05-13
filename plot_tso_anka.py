"""
For processing a new measurement, first run 'process_locations.py', and then 
create a new data file using 'create_datafile.py'.

If the function is called without a specified measurement date, the latest available
measurement date will be used. To select a specific measurement date, you have two options:
    - Specify the date as a string in 'YYYYMMDD' format, for example: measurement_date = '20240601'.
    - Specify an integer, where:
        - -1 corresponds to the latest measurement date,
        - -2 corresponds to the second latest measurement date, and so on. For example: measurement_date = -5.

If 'plot_mode' is set to 'multiple', by default the latest six measurements are used for plotting.
However, you can adjust this by using the 'multiple_offset' parameter to select a different set of six consecutive measurements. For example:
    - If 'multiple_offset = 2', the function will select six measurements starting from the third-most recent measurement.
    - Similarly, 'multiple_offset = 7' will select the six measurements starting from the eight-most recent, and so on.
"""

from plot_measurement import process_plots

location = 'anka'
process_plots(location)

#%%

location = 'anka'
process_plots(location, plot_mode='multiple')