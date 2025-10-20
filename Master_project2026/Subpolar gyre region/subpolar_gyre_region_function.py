import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

def plot_ocean_variable_maps(
    netcdf_file,
    variable_name,
    lat_min=45,
    lat_max=65,
    lon_min=-60,
    lon_max=-10,
    start_year=1850,
    end_year=2014,
    period_length=20,
    output_file=None,
    title=None,
    cmap_colors=None,
    units=None,
    figsize=(18, 15)
):
    """
    Create spatial maps of ocean variables for multiple time periods.
    
    Parameters:
    -----------
    netcdf_file : str
        Path to the NetCDF file
    variable_name : str
        Name of the variable to plot (e.g., 'tos', 'sos', 'zos')
    lat_min, lat_max : float
        Latitude range for region selection
    lon_min, lon_max : float
        Longitude range for region selection
    start_year, end_year : int
        Time range for analysis
    period_length : int
        Length of each averaging period in years (default: 20)
    output_file : str, optional
        Output filename (auto-generated if None)
    title : str, optional
        Overall figure title (auto-generated if None)
    cmap_colors : list, optional
        List of colors for colormap (auto-selected if None)
    units : str, optional
        Units for the variable (auto-detected if None)
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    
    # Variable-specific defaults
    variable_config = {
        'tos': {
            'colors': ['#08306b', '#2171b5', '#6baed6', '#c6dbef', 
                      '#fee5d9', '#fcae91', '#fb6a4a', '#cb181d', '#67000d'],
            'units': '°C',
            'long_name': 'Sea Surface Temperature'
        },
        'sos': {
            'colors': ['#54278f', '#756bb1', '#9e9ac8', '#bcbddc', '#dadaeb',
                      '#e5f5e0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32'],
            'units': 'PSU',
            'long_name': 'Sea Surface Salinity'
        },
        'zos': {
            'colors': ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
                      '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'],
            'units': 'm',
            'long_name': 'Sea Surface Height'
        },
        'mlotst': {
            'colors': ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4',
                      '#1d91c0', '#225ea8', '#253494', '#081d58'],
            'units': 'm',
            'long_name': 'Mixed Layer Depth'
        }
    }
    
    # Set defaults based on variable if not provided
    if cmap_colors is None and variable_name in variable_config:
        cmap_colors = variable_config[variable_name]['colors']
    elif cmap_colors is None:
        cmap_colors = ['#440154', '#31688e', '#35b779', '#fde724']  # viridis-like
    
    if units is None and variable_name in variable_config:
        units = variable_config[variable_name]['units']
    elif units is None:
        units = 'units'
    
    # Load the NetCDF file
    print(f"Loading NetCDF file: {netcdf_file}")
    ds = xr.open_dataset(netcdf_file, use_cftime=True)
    
    # Check if variable exists
    if variable_name not in ds.variables:
        print(f"Available variables: {list(ds.variables)}")
        raise ValueError(f"Variable '{variable_name}' not found in dataset")
    
    print(f"Dataset loaded. Shape: {ds[variable_name].shape}")
    print(f"Coordinates: {list(ds.coords)}")
    
    # Extract coordinates (2D for curvilinear grids)
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    data = ds[variable_name]
    
    print(f"\nLatitude range: {lats.min():.2f} to {lats.max():.2f}")
    print(f"Longitude range: {lons.min():.2f} to {lons.max():.2f}")
    
    # Handle longitude if it's in 0-360 format
    if lons.max() > 180:
        print("Converting longitude from 0-360 to -180-180...")
        lons = np.where(lons > 180, lons - 360, lons)
    
    # Create mask for region
    print(f"\nSelecting region: {lat_min}-{lat_max}°N, {lon_min}-{lon_max}°E")
    mask = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
    
    # Find bounding box - handle both (j,i) and (y,x) dimension names
    spatial_dims = [d for d in data.dims if d != 'time']
    dim1, dim2 = spatial_dims[0], spatial_dims[1]
    
    indices = np.where(mask)
    idx1_min, idx1_max = indices[0].min(), indices[0].max()
    idx2_min, idx2_max = indices[1].min(), indices[1].max()
    
    print(f"Grid indices: {dim1}=[{idx1_min}:{idx1_max}], {dim2}=[{idx2_min}:{idx2_max}]")
    
    # Select region
    data_region = data.isel({dim1: slice(idx1_min, idx1_max+1), 
                             dim2: slice(idx2_min, idx2_max+1)})
    lats_region = lats[idx1_min:idx1_max+1, idx2_min:idx2_max+1]
    lons_region = lons[idx1_min:idx1_max+1, idx2_min:idx2_max+1]
    
    # Refine mask for the selected region
    mask_region = ((lats_region >= lat_min) & (lats_region <= lat_max) & 
                   (lons_region >= lon_min) & (lons_region <= lon_max))
    
    # Create time periods
    periods = [(year, min(year + period_length - 1, end_year)) 
               for year in range(start_year, end_year, period_length)]
    
    print(f"\nCreating maps for {len(periods)} periods:")
    for start, end in periods:
        print(f"  {start}-{end}")
    
    # Create figure
    n_periods = len(periods)
    n_cols = 3
    n_rows = int(np.ceil(n_periods / n_cols))
    
    fig = plt.figure(figsize=figsize)
    
    # Create colormap
    cmap = LinearSegmentedColormap.from_list('custom', cmap_colors, N=100)
    
    # Calculate all period means for consistent color scale
    print("\nCalculating period means...")
    all_means = []
    for start, end in periods:
        period_data = data_region.sel(time=slice(f'{start}', f'{end}'))
        period_mean = period_data.mean(dim='time').values
        period_mean_masked = np.where(mask_region, period_mean, np.nan)
        all_means.append(period_mean_masked)
    
    # Get valid values for color scale
    valid_values = np.concatenate([m[~np.isnan(m)] for m in all_means])
    vmin = np.percentile(valid_values, 2)
    vmax = np.percentile(valid_values, 98)
    
    print(f"{variable_name} range across all periods: {vmin:.2f} to {vmax:.2f} {units}")
    
    # Create maps
    for idx, (start, end) in enumerate(periods):
        print(f"\nProcessing period {start}-{end}...")
        
        period_mean = all_means[idx]
        regional_mean = np.nanmean(period_mean)
        
        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, 
                             projection=ccrs.PlateCarree())
        
        # Plot data
        im = ax.pcolormesh(lons_region, lats_region, period_mean,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap,
                           vmin=vmin,
                           vmax=vmax,
                           shading='auto')
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', alpha=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5, zorder=1)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, 
                          color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Set extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], 
                      crs=ccrs.PlateCarree())
        
        # Title
        ax.set_title(f'{start}-{end}\nMean: {regional_mean:.2f} {units}',
                     fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label=f'{variable_name} ({units})')
    
    # Overall title
    if title is None:
        if variable_name in variable_config:
            var_name = variable_config[variable_name]['long_name']
        else:
            var_name = variable_name.upper()
        title = f'{var_name} - North Atlantic Subpolar Gyre\n{period_length}-Year Averages ({start_year}-{end_year})'
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    # Save figure
    if output_file is None:
        output_file = f'{variable_name}_subpolar_gyre_{period_length}yr_maps.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as: {output_file}")
    
    ds.close()
    print("Done!")
    
    return fig


def plot_regional_yearly_mean(
    netcdf_file,
    variable_name,
    lat_min=45,
    lat_max=65,
    lon_min=-60,
    lon_max=-10,
    start_year=None,
    end_year=None,
    output_file=None,
    title=None,
    marker='o',
    figsize=(10,5)
):
    """
    Compute and plot a yearly mean time series for a specific region
    on a curvilinear grid.

    Parameters
    ----------
    netcdf_file : str
        Path to the NetCDF file
    variable_name : str
        Variable to analyze (e.g., 'tos', 'sos')
    lat_min, lat_max, lon_min, lon_max : float
        Region bounds
    start_year, end_year : int, optional
        Time range to select
    output_file : str, optional
        File to save the figure
    title : str, optional
        Plot title
    marker : str
        Marker style for plot
    figsize : tuple
        Figure size

    Returns
    -------
    tos_region_yearly : xarray.DataArray
        Yearly mean time series for the region
    """
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt

    # Open dataset
    ds = xr.open_dataset(netcdf_file, use_cftime=True)
    
    if variable_name not in ds.variables:
        raise ValueError(f"Variable {variable_name} not found in dataset")
    
    data = ds[variable_name]

    # Extract lat/lon arrays
    lats = ds['latitude'].values
    lons = ds['longitude'].values

    # Convert lon if needed
    if lons.max() > 180:
        lons = np.where(lons > 180, lons - 360, lons)
    
    # Create mask
    mask = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)

    # Identify spatial dimensions
    spatial_dims = [d for d in data.dims if d != 'time']
    dim1, dim2 = spatial_dims[0], spatial_dims[1]

    # Find bounding box indices
    indices = np.where(mask)
    idx1_min, idx1_max = indices[0].min(), indices[0].max()
    idx2_min, idx2_max = indices[1].min(), indices[1].max()

    # Subset data for region
    data_region = data.isel({dim1: slice(idx1_min, idx1_max+1),
                             dim2: slice(idx2_min, idx2_max+1)})
    lats_region = lats[idx1_min:idx1_max+1, idx2_min:idx2_max+1]
    
    # Refine mask
    mask_region = ((lats_region >= lat_min) & (lats_region <= lat_max) &
                   (lons[idx1_min:idx1_max+1, idx2_min:idx2_max+1] >= lon_min) &
                   (lons[idx1_min:idx1_max+1, idx2_min:idx2_max+1] <= lon_max))
    
    # Apply mask
    data_region = data_region.where(mask_region)

    # Cosine latitude weighting
    weights = np.cos(np.deg2rad(lats_region))
    weights = weights.where(~np.isnan(data_region))
    weights = weights / weights.sum()

    # Weighted spatial mean
    region_mean = (data_region * weights).sum(dim=[dim1, dim2])

    # Select time range
    if start_year is not None and end_year is not None:
        region_mean = region_mean.sel(time=slice(f'{start_year}', f'{end_year}'))

    # Resample yearly
    region_yearly = region_mean.resample(time='Y').mean()

    # Plot
    plt.figure(figsize=figsize)
    region_yearly.plot(marker=marker)
    if title is None:
        title = f'{variable_name.upper()} - Yearly Mean ({lat_min}-{lat_max}N, {lon_min}-{lon_max}E)'
    plt.title(title)
    plt.ylabel(variable_name)
    plt.xlabel('Year')
    plt.grid(True)
    
    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    ds.close()
    return region_yearly
