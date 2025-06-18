from process_raw_tso_files import proces_individual_files

# =======================================================================
# Select which location to process by uncommenting the appropriate line:
# location = 'vzm'    # Volkerak-Zoommeer (combines data from anka, volk, and zoom)
location = 'grev'   # Grevelingenmeer
# location = 'veer'   # Veerse Meer
# location = 'kvgt'   # Kanaal Gent-Terneuzen
# =======================================================================

#%% Main processing call

# Run the processing for the selected location.
# Make sure you have uncommented one of the location assignments at the top of the script.

proces_individual_files(location)