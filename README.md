TSO Water Quality Processing

This codebase has been built to handle the processing of water quality measurements for Rijkswaterstaat Zee en Delta. These measurements, called TSO-measurements, focus on the three most important parameters in the dataset: Temperature, Salinity, and Oxygen.
Project Background

The history of these measurements dates back to the late 1970s. Traditionally, data processing was performed at a local branch of Rijkswaterstaat in Zeeland using a Fortran library. Over more than thirty years, this Fortran code has accumulated a lot of legacy code and technical debt, making it hard to maintain or modify.

To overcome these challenges, a Python version of the processing code was developed. The Python code:

    Maintains compatibility: It uses the same structure of input files as the Fortran code, allowing generated datafiles to be used interchangeably (as long as no new parameters—like Phycocyanine—are introduced or units—such as the change for Turbidity from NTU to FNU—are altered).
    Provides enhanced visualization: When new parameters or unit changes occur, only the Python code can generate the required visualizations.
    Preserves legacy functionality: The Fortran code remains available locally on Rijkswaterstaat servers, but it will not be published due to its reliance on local libraries, including proprietary software, and the complexity of running it on other systems.

Note: To maintain backward compatibility, the processing code is more complex than it might have been otherwise. While the Python code has been thoroughly tested on recent datafiles, it may not work correctly with very old datafiles due to changes in measurement point numbering, trajectories, and datafile structures. All available datafiles have been uploaded so that anyone can read the data and develop their own readers if necessary.

The processed data is published on the official website:
Zeeland Metingen TSO
Running the Scripts

The scripts can be executed from the command line. However, you might need to comment or uncomment certain parts of the code for specific tasks. For a more interactive experience, consider using an IDE like Spyder or creating Jupyter notebooks.
Setting Up the Conda Environment

This project uses Conda to manage its dependencies. All required packages (with their exact versions) are listed in the environment.yaml file.
How to Create the Environment

    Install Conda:
    If you haven’t already, install Miniconda or Anaconda.

    Clone the Repository:
    Make sure you have cloned this repository to your local machine.

    Create the Environment:
    Open a terminal, navigate to the project directory, and run:

conda env create -f environment.yaml

This command creates a new Conda environment (named tso, as specified in the file) with all the required packages and versions.

Activate the Environment:
Once the environment is created, activate it by running:

    conda activate tso

    Run the Project:
    With the environment active, you can now run the project’s scripts and applications as described above.

Note on the prefix Field

The environment.yaml file contains a prefix line (e.g., prefix: C:\Users\Wilbert\anaconda3\envs\tso). This line specifies the absolute path to the environment on my computer and is specific to my setup. You can safely delete this line for portability. When you create the environment on your system, Conda will ignore the prefix or create the environment in your default location.
