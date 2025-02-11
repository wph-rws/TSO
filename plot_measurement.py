import datetime
import numpy as np
import pandas as pd
import pyproj
import matplotlib.pyplot as plt
import cmocean
import yaml
import pathlib

from scipy.interpolate import griddata
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist

from tso_functions import load_tso_parameters, load_project_dirs, get_location_info
from read_datafile import read_datafile
from plot_kaartje import plot_kaartje, transform_points, get_location_data_for_map

#%%
def plot_parameter(ax, df, mpnaam, vmin, vmax, colorstep, colorstep_factor, cmap, parameter, measurement_date, plot_manager, apply_smoothing=True):
    
    # Access properties from PlotManager instance
    clabel_fontsize = plot_manager.clabel_fontsize
    tick_fontsize = plot_manager.tick_fontsize
    dot_size_depth = plot_manager.dot_size_depth
    location = plot_manager.location
    
    fig = plot_manager.fig

    # Load available TSO-parameters
    parameters = load_tso_parameters()       
      
    # Make copy of mpnaam.df and to preserve original mpnaam.df 
    mpnaam_df = mpnaam.df.copy()   
    df = df.copy()
    
    # Filter mpnaam_df on points that are in the dataframe
    mpnaam_df = mpnaam_df[mpnaam_df['loc_name'].isin(df['berekend_meetpunt'].unique().astype(str))]
    mpnaam_df = mpnaam_df.set_index('loc_name')
    
    # Map location number to corresponding distance
    mpnaam_df.index = mpnaam_df.index.astype(str)    
        
    berekend_meetpunt = df['berekend_meetpunt'].astype(str)
    df['dist'] = berekend_meetpunt.map(mpnaam_df['dist'])
    
    # Vector for ax.fill
    z_max_sensor = df.groupby('meetpunt').agg({'diepte': 'max'}).reset_index()
    z_fill = np.max([mpnaam_df['max_depth'], z_max_sensor['diepte']], axis=0) 
    
    # Get description and unit of parameter
    graph_parameter = parameter
    description_parameter = parameters[parameter]['description'].lower()
    unit_parameter = parameters[parameter]['unit']
    graph_parameter_and_unit = f'{description_parameter.capitalize()} {unit_parameter}'
    clabel_fmt = '%1.0f'
    
    if description_parameter == 'chloride':
        df[description_parameter] = (df[description_parameter] / 1000).round(1)
        graph_parameter_and_unit = 'Chloridegehalte (g/l)'
        clabel_fmt = '%1.1f'
    elif description_parameter == 'temperatuur':
        graph_parameter_and_unit = 'Temperatuur ($^\circ$C)'
    elif description_parameter == 'zuurstof':
        graph_parameter = '$O_2$'
    elif parameter.lower() == 'ph':
        graph_parameter_and_unit = 'Zuurgraad'
        
    # Filter dataframe on 'notna' to determine if it is only 1 measurement point    
    # Check if the description_parameter column exists    
    value_not_found = -9999
    if description_parameter in df.columns:
        
        # Filter the DataFrame for non-null values in the description_parameter column
        filtered_df = df[df[description_parameter].notna()]
    else:
        # Add the column with the description_parameter name and fill with NaN
        df[description_parameter] = value_not_found
        print(f'Warning: "{description_parameter}" column not found, added column with values of {value_not_found}')
        filtered_df = df

    # Check if only one unique 'meetpunt' exists
    only_one_mp = filtered_df['meetpunt'].nunique() == 1

    # If only 1 measurement with values, create small grid around values to display in graph
    if only_one_mp:
        
        # Create new rows with distance -100 and +100
        df_above = filtered_df.copy()
        df_above['dist'] -= 300
        
        df_below = filtered_df.copy()
        df_below['dist'] += 300
        
        # Concatenate the new rows with the original DataFrame and overwrite it
        df_one_mp = pd.concat([df_above, filtered_df, df_below])
        
        # Create dataframes for x,y and parameter
        x = df_one_mp['dist']
        y = df_one_mp['diepte'] * -1
        z = df_one_mp[description_parameter]
        
        ngridx = 150
        ngridy = 100
        
        xi = np.linspace(0, np.max(x), ngridx)
        yi = np.linspace(np.min(y), np.max(y), ngridy)
            
    else:
        
        # Put data parameter in separate arrays
        x = df['dist']
        y = df['diepte'] * -1
        z = df[description_parameter]         
        
        ngridx = 900
        ngridy = 600
        
        xi = np.linspace(0, np.max(x), ngridx)
        yi = np.linspace(np.min(y), np.max(y), ngridy)
    
    # Filter out NaNs from the input data, otherwise blank regions in interpolation
    z = np.array(z)
    mask = ~np.isnan(z)
    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
        
    zi = griddata((x_filtered, y_filtered), z_filtered, (xi[None, :], yi[:, None]), method='linear')    
    
    # Apply Gaussian smoothing (if more than 1 measurment points are available)
    if apply_smoothing and not only_one_mp:
        
        # Determine the indices of columns that contain at least one non-NaN value
        valid_columns = np.where(np.any(~np.isnan(zi), axis=0))[0]
        
        # Identify the first and last valid column indices
        first_valid_col = valid_columns[0]
        last_valid_col = valid_columns[-1]       
    
        # Fill NaN values in the interpolated grid
        # Forward Fill and Backward Fill
        zi = (pd.DataFrame(zi)
              .ffill(axis=0)
              .bfill(axis=0)
              .ffill(axis=1)
              .bfill(axis=1)
              .values)
        
        # Apply Gaussian smoothing to the interpolated grid.
        # Gaussian smoothing replaces each grid value with a weighted average of its neighbors.
        # Mathematically, if f(x, y) represents the original grid, the smoothed value fₛ(x, y) is computed as:
        #
        #     fₛ(x, y) = ∬ f(u, v) Gₛ(x-u, y-v) du dv,
        #
        # where the Gaussian kernel Gₛ is defined by:
        #
        #     Gₛ(x, y) = (1 / (2πσ²)) * exp(-(x² + y²) / (2σ²)).
        #
        # This is equivalent to convolving the original grid f with the Gaussian kernel Gₛ:
        #
        #     fₛ = f * Gₛ.
        #
        # The parameter σ (sigma) controls the spread of the kernel. A larger σ means that the smoothing
        # operation averages over a broader neighborhood, which can blur sharp features. If σ is set too high,
        # even regions containing actual measured values may be excessively averaged with surrounding points 
        # (or interpolated regions), potentially leading to a loss of detail or local features effectively
        # 'disappearing', especially if NaNs from neighboring regions propagate into areas with measurements.
        #
        # In this implementation, Gaussian smoothing (with sigma=8) is applied to reduce noise, but be cautious:
        # over-smoothing may obscure important variations in the measured data and sigma=8 is a fairly heavy
        # smooting setting
        zi = gaussian_filter(zi, sigma=8)
        
        # Set all columns before the first valid column and after the last valid column
        # to NaN so that no data is introducted due to filling and filtering
        zi[:, :first_valid_col] = np.nan
        zi[:, last_valid_col+1:] = np.nan
   
    # Remove duplicate indices that have been made for griddata, so only dots for actual
    # measurement values are plotted with scatter plot
    if only_one_mp:
        df_scatter = pd.DataFrame({
            'x': x_filtered.values,
            'y': y_filtered.values,
            'original_index': x_filtered.index
            })
        df_scatter['group_count'] = df_scatter.groupby('original_index').cumcount()
        df_scatter = df_scatter[df_scatter['group_count'] == 1]
        
        x_filtered = df_scatter['x']
        y_filtered = df_scatter['y']     
        
    # Format title of subplot
    title = f'{graph_parameter}     {graph_parameter_and_unit}          {measurement_date}'
               
    ymax_mpnaam = (np.ceil(mpnaam.df['max_depth']/2).round()*2).max()
    ymax_data = (np.ceil(df['diepte']/2).round()*2).max()
    ymax = np.max([ymax_mpnaam, ymax_data])
    ylim = [-ymax, 0]
        
    levels = np.arange(vmin, vmax+colorstep, colorstep)
    
    # Filter zi for values close to a contour level, cause this results in a lot of 
    # contour lines that clutter the figure
    tolerance = 0.02
    for level in levels:
        mask = np.abs(zi - level) < tolerance
        zi[mask] = level
       
    # Test for extrema in data versus levels
    if np.all(np.isnan(zi)):
        zmax = np.max(levels)
        zmin = np.min(levels)
    else:
        zmax = np.nanmax(zi)
        zmin = np.nanmin(zi)

    parname = parameters[parameter]['description']
    contour_color = 'black'
    
    if zmax > np.max(levels):
        print(f'Maximum for parameter {parname} = {zmax:.2f}, which is higher than the maximum of the colormap ({levels[-1]})')
        if parname.lower() == 'troebelheid':
            new_max_level = np.ceil(zmax / 10) * 10
            levels = np.append(levels, new_max_level)
            contour_color = 'red'
    if zmin < np.min(levels):
        print(f'Minimum for parameter {parname} = {zmin:.2f}, which is lower than the minimum of the colormap ({levels[0]})')
    
    quadcontourset = ax.contourf(xi, yi, zi, levels, vmin=vmin, vmax=vmax, cmap=cmap)
    contour_lines = ax.contour(xi, yi, zi, levels, colors=contour_color, linewidths=0.5)  

    # Create an axis on the right side for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.18)

    cbar = fig.colorbar(
        ScalarMappable(norm=quadcontourset.norm, cmap=quadcontourset.cmap),
        cax=cax,
        ticks=np.arange(vmin, vmax+1, colorstep*colorstep_factor),
        boundaries=levels,
        values=(levels[:-1] + levels[1:]) / 2)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    ax.set_ylim(ylim)
    lighter_lightgrey = '#dcdcdc'
    ax.fill_between(mpnaam_df['dist'], -1.01*z_fill, ax.get_ylim()[0], color=lighter_lightgrey, alpha=1, zorder=3)
    ax.plot(mpnaam_df['dist'], -1.01*z_fill, color='darkgrey', linewidth=1, zorder=4)
    
    # Overlay between mp30 and mp34 for Zoommeer
    if location == 'zoom':  
       ax.fill_between(mpnaam_df['dist'].loc['30':'34'], ax.get_ylim()[0], ax.get_ylim()[1], color='white', alpha=1, zorder=2)
       
    # Generate a list of label positions
    label_positions = []
    vertex_lengths = []
    
    #%% Custom code to calculate number of contour labels
    for segment in contour_lines.allsegs:
            
        # Get points of the path object (Numpy array inside a list)
        if len(segment[0]) > 0:
            vertices = segment[0]
        else:
            continue                
        
        # Calculate the total contour length by summing pairwise distances
        if len(vertices) > 1:
            
            # Calculate pairwise distances between consecutive vertices
            distances = pdist(vertices)
            
            # Sum the distances to get the total vertex length
            vertex_length = np.sum(distances)
            vertex_lengths.append(vertex_length)
            
        # If there's only one vertex, the length is 0    
        else: 
            vertex_length = 0
            vertex_lengths.append(vertex_length)
        
        # Ensure vertex_length is at least 1 to avoid issues with log10(0)
        vertex_length = np.nanmax([1, vertex_length])
        
        # Define the candidate functions based on the area of applicability:
        # For short contours: use the 10th root (f_short)
        f_short = np.max([1, vertex_length ** (1/10) - 2])
        
        # For long contours: use the base-10 logarithm (f_long)
        f_long = np.log10(vertex_length)
        
        # Define parameters for the logistic blending weight:
        L0 = 1e9    # Transition scale: the approximate vertex length where blending is 50/50
        k = 0.7     # Steepness of the transition: higher k means a sharper transition
        
        # Compute the logistic weight:
        # The weight is near 0 for small vertex lengths (favoring f_short)
        # and near 1 for large vertex lengths (favoring f_long).
        weight = 1 / (1 + np.exp(-k * (np.log(vertex_length) - np.log(L0))))
        
        # Blend the two methods using the weight:
        blended_labels = (1 - weight) * f_short + weight * f_long
        
        # Round the blended result to get an integer number of labels
        num_labels = int(np.round(blended_labels))
        
        # print(f'Vertex length is {vertex_length:.0f}, calculated {num_labels} labels')
        
        # Evenly select indices along the vertices for label placement
        indices = np.linspace(0, len(vertices) - 1, num=num_labels, dtype=int)
        label_positions.extend(vertices[indices])                              
    
    # If more than 3 custom label positions are calculated, manual label positions,
    # for less than 3 it results in errors for finding nearest contour
    if len(label_positions) > 3:        
        ax.clabel(contour_lines, inline=True, fontsize=clabel_fontsize, colors=contour_color,
            fmt=clabel_fmt, manual=label_positions, inline_spacing=3, zorder=1)      
    else:
        ax.clabel(contour_lines, inline=True, fontsize=clabel_fontsize, colors=contour_color,
                  fmt=clabel_fmt, inline_spacing=3, zorder=1)    
    #%%
    
    # Do not plot dots for -9999 values that have been created for parameters that are not
    # present in the dataset to avoid an empty ax for that parameter
    if not any(df[description_parameter] == value_not_found):
        ax.scatter(x_filtered, y_filtered, marker='o', c='black', s=dot_size_depth, zorder=4)
    ax.set_title(title, fontsize=plot_manager.label_fontsize)

    # Set xticks and xtick labels
    ax.set_xticks(mpnaam_df['dist'])
    ax.set_xticklabels([str(i) for i in df['berekend_meetpunt'].unique()])
    ax.tick_params(axis='both', labelsize=tick_fontsize)

#%%

# Function to calculate the area of a polygon given its vertices
def polygon_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

#%%

def plot_map_tso(mpnaam, ax, plot_manager):
    
    # from pyproj import Transformer
    
    # # Create a transformer object for converting RDNew to ETRS89 / UTM zone 31N
    # transformer = Transformer.from_crs("EPSG:28992", "EPSG:25831", always_xy=True)   
    
    xll, yll = plot_manager.xll, plot_manager.yll    
    # xll, yll = transformer.transform(xll, yll)
    
    dx, dy = plot_manager.dx, plot_manager.dy
    rotation_angle = plot_manager.rotation_angle
    
    # Define the desired origin for rotation for the red dots in the map
    origin_x, origin_y = xll + dx / 2, yll + dy / 2  # Center of the polygon
    
    # if plot_manager.gebiedscode.upper() in ('AK TSO', 'KG TSO'):
    #     xm_transformed, ym_transformed = transform_points(mpnaam.df['x_etrs'], mpnaam.df['y_etrs'], origin_x, origin_y, rotation_angle)  
    # else:
    #     xm_transformed, ym_transformed = transform_points(mpnaam.df['x'], mpnaam.df['y'], origin_x, origin_y, rotation_angle)
        
    xm_transformed, ym_transformed = transform_points(mpnaam.df['x'], mpnaam.df['y'], origin_x, origin_y, rotation_angle)

    # Subplot map
    img_data, bbox = [], []
    ax, img_data, bbox = plot_kaartje(plot_manager.location, ax, img_data, type_kaart='map',
                                      force_refresh_image=False, debug=False, rotation_angle=rotation_angle)
    ax.tick_params(axis='both', which='major', labelsize=plot_manager.tick_fontsize)
    
    # Add annotations to the scatter plot
    for i, (x_pos, y_pos) in enumerate(zip(xm_transformed, ym_transformed), start=0):
        ax.annotate(mpnaam.df.iloc[i]['loc_name'], (x_pos, y_pos), textcoords='offset points', xytext=(0,5), ha='center', fontsize=plot_manager.clabel_fontsize)
    
    if plot_manager.location == 'zoom':
        ax.plot(xm_transformed[:3], ym_transformed[:3], color='black', linewidth=1.0)
        ax.plot(xm_transformed[3:], ym_transformed[3:], color='black', linewidth=1.0)
    else:
        ax.plot(xm_transformed, ym_transformed, color='black', linewidth=1.0)
        
    # Transformation from RD --> ETRS --> RD causes some rotation
    # is fixed by custom cropping of the map, not an ideal solution, but
    # only working option for now where the coordinates for the ship route 
    # still align with the map
    xlim_map = ax.get_xlim()
    ylim_map = ax.get_ylim()

    if plot_manager.location == 'kvgt' and rotation_angle == -90:
        xo = 325
        yo = 1000
    elif plot_manager.location == 'anka' and rotation_angle == -90:
        xo = 165
        yo = 550
    else:
        xo = 0
        yo = 0
    
    # Apply new limits
    ax.set_xlim(xlim_map[0] - xo, xlim_map[1] + xo)
    ax.set_ylim(ylim_map[0] - yo, ylim_map[1] + yo)
       
    ax.scatter(xm_transformed, ym_transformed, color='red', s=8, zorder=2)
    ax.set_title('Overzicht gevaren route', fontsize=plot_manager.label_fontsize) 

#%% Class PlotManager

class PlotManager:
    def __init__(self, location, measurement_date='latest', plot_mode='single', parameters_multiple=None):
        
        # Load common and location-specific parameters from YAML
        with open('plot_parameters.yaml', 'r') as f:
            self.params = yaml.safe_load(f)
            
        # Load project dirs
        project_dirs = load_project_dirs()
        self.output_dir = pathlib.Path(project_dirs.get('output_dir'))

        self.location_code, self.location_name, self.location_data = get_location_info(location)
        self.location = location  
        self.plot_mode = plot_mode

        # If the caller did not provide parameters for multiple mode, use a default list
        if parameters_multiple is None:
            self.parameters_multiple = ['salinity', 'temperature', 'oxygen', 'ph']
        else:
            self.parameters_multiple = parameters_multiple

        self.clabel_fontsize = 4         
        self.datetime_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.figures = []  # Storage list for 'multiple' mode
        self.ignored_points = {} # Points for which question about distance limit is already answered

        # Plotting mode
        if plot_mode == 'single':
            self.n_axis = 4    
            self.df, self.meetdatum_output, self.mpnaam, _ = read_datafile(location, measurement_date)
            self.suptitle_fontsize = 16
            self.label_fontsize = 9
            self.tick_fontsize = 7
            self.dot_size_depth = 0.4 
            self.space_last_axis = 0.25  # 25% of space for the map on the last axis  
            self.filename = self.output_dir / f'{self.location_code.upper()} TSO - {self.meetdatum_output} - {self.datetime_now}.pdf'
            self.create_figure()
            
        elif plot_mode == 'multiple':
            self.n_axis = 7  # 6+1 axis, last one for map
            self.date_range = list(np.arange(-self.n_axis + 1, 0))
            self.iter_dates = iter(self.date_range)   
            self.suptitle_fontsize = 13
            self.label_fontsize = 7
            self.dot_size_depth = 0.025 
            self.tick_fontsize = 5
            self.space_last_axis = 0.28  # 28% of space for the map on the last axis  
            self.filename = self.output_dir / f'{self.location_code.upper()} TSO - Laatste {len(self.date_range)} metingen - {self.datetime_now}.pdf'
            # Figures will be created in plot_multiple_parameters
            
        else:
            raise ValueError('Unknown plot mode')

        # Set location-specific variables
        self.set_location_variables(self.location)
        
        # Set gebiedscode
        self.gebiedscode = f'{self.location_code} TSO'
        
        # Convert xll and yll to ETRS for 'anka' and 'kvgt'
        if location in ('anka', 'kvgt'):
            self.xll, self.yll = self.convert_xll_yll(self.xll, self.yll)

    def convert_xll_yll(self, xll, yll):
        transformer = pyproj.Transformer.from_crs('EPSG:28992', 'EPSG:28992', always_xy=True)
        xll_etrs, yll_etrs = transformer.transform(xll, yll)
        return xll_etrs, yll_etrs

    def set_location_variables(self, location):
        
        # Load common parameters from YAML
        common_params = self.params['common']
        
        self.oxy_min = common_params['oxy_min']
        self.oxy_max = common_params['oxy_max']
        self.oxy_colorstep = common_params['oxy_colorstep']
        self.oxy_colorstep_factor = common_params['oxy_colorstep_factor']

        self.temp_min = common_params['temp_min']
        self.temp_max = common_params['temp_max']
        self.temp_colorstep = common_params['temp_colorstep']
        self.temp_colorstep_factor = common_params['temp_colorstep_factor']

        self.ph_min = common_params['ph_min']
        self.ph_max = common_params['ph_max']
        self.ph_colorstep = common_params['ph_colorstep']
        self.ph_colorstep_factor = common_params['ph_colorstep_factor']

        # Load location-specific parameters from YAML
        loc_params = self.params['locations'].get(location)
        if loc_params is None:
            raise ValueError(f'Unknown location: {location}')
        
        # Retrieve map-related parameters using your existing function.
        self.xll, self.yll, self.dx, self.dy = get_location_data_for_map(location)
        self.rotation_angle = loc_params['rotation_angle']

        # Set salinity parameters for the location
        self.sal_min = loc_params['sal_min']
        self.sal_max = loc_params['sal_max']
        self.sal_colorstep = loc_params['sal_colorstep']
        self.sal_colorstep_factor = loc_params['sal_colorstep_factor']

        # Handle extra settings for certain locations
        if location == 'volk':
            if 'additional_parameters' in loc_params:
                self.parameters_multiple.extend(loc_params['additional_parameters'])
            self.turbid_min = loc_params.get('turbid_min')
            self.turbid_max = loc_params.get('turbid_max')
            self.turbid_colorstep = loc_params.get('turbid_colorstep')
            self.turbid_colorstep_factor = loc_params.get('turbid_colorstep_factor')
            
            # Allow location-specific override of common pH and oxygen limits:
            self.ph_max = loc_params.get('ph_max', self.ph_max)
            self.oxy_max = loc_params.get('oxy_max', self.oxy_max)

        elif location == 'zoom':
            self.ph_max = loc_params.get('ph_max', self.ph_max)
            self.oxy_max = loc_params.get('oxy_max', self.oxy_max)

    def create_figure(self):
        
        # Define the proportions for the axes        
        space_remaining = 1 - self.space_last_axis
        space_per_axis = space_remaining / (self.n_axis - 1)
        
        # Create the figure and axes using gridspec_kw to set the relative heights
        self.fig, self.axes = plt.subplots(self.n_axis, 1, figsize=(8.27, 11.69), 
                                           gridspec_kw={'height_ratios': [space_per_axis] * (self.n_axis - 1) + [self.space_last_axis]})
        self.axis_iterator = iter(self.axes)
        self.fig.suptitle(self.location_name, fontsize=self.suptitle_fontsize)
        
        # Adjust tight_layout parameters        
        if self.plot_mode == 'single':
            self.fig.tight_layout(pad=1.2, h_pad=0.9, rect=[0, 0, 0.98, 0.98])
        elif self.plot_mode == 'multiple':
            self.fig.tight_layout(pad=0.5, h_pad=0.4, rect=[0, 0, 0.98, 0.98])

    def reset_iterators(self):
        self.iter_dates = iter(self.date_range)
        self.axis_iterator = iter(self.axes)

    def next_axis(self, read_data=True):
        try:
            next_axis = next(self.axis_iterator)
            if self.plot_mode == 'multiple' and read_data:
                self.df, self.meetdatum_output, self.mpnaam, self.ignored_points = read_datafile(self.location, next(self.iter_dates), 
                                                                                            self.ignored_points, plot_mode='multiple')
            return next_axis
        except StopIteration:
            raise IndexError('No more axes available')

    # Definition of avalaible plot modes, expand when necessary
    def plot_salinity(self):
        plot_parameter(self.next_axis(), self.df, self.mpnaam, self.sal_min, self.sal_max, self.sal_colorstep, 
                       self.sal_colorstep_factor, cmocean.cm.haline_r, 'CL-', self.meetdatum_output, self)

    def plot_temperature(self):
        plot_parameter(self.next_axis(), self.df, self.mpnaam, self.temp_min, self.temp_max, self.temp_colorstep,
                       self.temp_colorstep_factor, 'jet', 'Temp', self.meetdatum_output, self)

    def plot_oxygen(self):
        plot_parameter(self.next_axis(), self.df, self.mpnaam, self.oxy_min, self.oxy_max, self.oxy_colorstep, 
                       self.oxy_colorstep_factor, cmocean.cm.balance_r, 'O2', self.meetdatum_output, self)
    def plot_ph(self):
        plot_parameter(self.next_axis(), self.df, self.mpnaam, self.ph_min, self.ph_max, self.ph_colorstep, 
                       self.ph_colorstep_factor, 'RdBu', 'pH', self.meetdatum_output, self)
        
    def plot_turbidity(self):
        plot_parameter(self.next_axis(), self.df, self.mpnaam, self.turbid_min, self.turbid_max, self.turbid_colorstep, 
                       self.turbid_colorstep_factor, cmocean.cm.turbid, 'TURBID', self.meetdatum_output, self)    
    
    def plot_overview_map(self):
        plot_map_tso(self.mpnaam, self.next_axis(read_data=False), self)

    def plot_multiple_parameters(self):
        for parameter in self.parameters_multiple:
            self.create_figure()
            self.reset_iterators()
            for _ in self.date_range:
                if parameter == 'salinity':
                    self.plot_salinity()
                elif parameter == 'temperature':
                    self.plot_temperature()
                elif parameter == 'oxygen':
                    self.plot_oxygen()
                elif parameter == 'ph':
                    self.plot_ph()
                elif parameter == 'turbidity':
                    self.plot_turbidity()
                else:
                    raise ValueError(f'Unknown parameter: {parameter}')
            self.plot_overview_map()
            self.figures.append(self.fig)

    def save_figure(self):       
        
        if self.plot_mode == 'single':
            self.fig.savefig(self.filename, format='pdf', dpi=300)
            plt.close(self.fig)
            print(f'Figure saved as "{self.filename}"')
        elif self.plot_mode == 'multiple':
            with PdfPages(self.filename) as pdf:
                for fig in self.figures:
                    pdf.savefig(fig, dpi=300)
                    plt.close(fig)
            print(f'All figures saved as "{self.filename}"')
        else:
            raise ValueError(f'Unknown plot mode: {self.plot_mode}')

            
#%%
            
def process_plots(location, measurement_date='latest', plot_mode='single', parameters_multiple=None):
    
    plot_manager = PlotManager(location, measurement_date=measurement_date, plot_mode=plot_mode, parameters_multiple=parameters_multiple)
    
    if plot_manager.plot_mode == 'single':
        plot_manager.plot_salinity()
        plot_manager.plot_temperature()
        plot_manager.plot_oxygen()
        plot_manager.plot_overview_map()
    elif plot_manager.plot_mode == 'multiple':
        plot_manager.plot_multiple_parameters()
    else:
        raise ValueError(f'Unknown plot mode: {plot_manager.plot_mode}')
        
    plot_manager.save_figure()