This codebase has been built to handle the processing of water quality measurements for Rijkswaterstaat Zee en Delta. These measurements are called 'TSO-measurements', after the most important parameters in the dataset: 'Temperature', 'Salinity' and 'Oxygen'. 

The history of the measurments goes back to the end of the 1970's and the processing of the data has always been done at a local branch of Rijkswaterstaat in Zeeland. A Fortran library has been created and updated till a few years ago to process the data and make plots to visualize the data.
However, the Fortran codebase goes back more than thirty years, with a lot of legacy code and a fair amount of technical debt. As a result, it is hard to maintain or change the codebase. Therefore, a Python version of the code was created. The Python code uses the same structure of input files
as the Fortran code did and is able to generate datafiles that can be processed with the Fortran code and viceversa, as long as no new parameters like 'Phycocyanine' are measured or the units of the measurement have been changed, like the units for 'Turbidity', which have been changed from 'NTU' to 'FNU'.
In those cases, only the new Python code is able to produce visualisations. The Fortran code is only available locally on servers of Rijkswaterstaat, this code will not be published because it requires a lot of local libraries, including non-open source (commercial) software and a lot of effort to make
it run on other systems/servers.

To maintain this backward compatibility, the processing code is more complex than what it would have been if it had been written without any constraints. The Python code has been tested on a lot of datafiles from the past few years, but probably will not work correctly on very old datafiles.
There have been changes to the numbering of the measurement points, the measurement trajectory, the structure of the datafiles, etc. It is not the intention to make the scripts work with all the old datafiles, that would be too much work and most likely result in an even more complex codebase.
All datafiles that are available have been uploaded, so anyone can read the data and create code to read these files when they wish to do so.

The processed data is published on this website:

https://waterberichtgeving.rws.nl/owb/regio/regio-zeeuwse-wateren/zeeland-metingen/zeeland-metingen-tso

The scripts can be used from the command line, but that may require commenting / uncommenting of parts of the code. It may be easier to use it in a IDE like Spyder or create notebooks.
