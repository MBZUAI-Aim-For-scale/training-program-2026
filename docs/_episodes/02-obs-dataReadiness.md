---
title: "Demo 2"
order: 2
teaching: 1
exercises: 0
questions:
- "How do we construct ENACTS using rain gauges and satellite products?"
- "How can we effectively grid different levels of station data with satellite observations?"
- "What critical insights can be derived from local rain gauge analyses?"
objectives:
- "Understand the step-by-step process of constructing ENACTS by combining gauge and satellite products."
- "Showcase methodologies for gridding different levels of stations with satellite data."
- "Perform and interpret targeted rain gauge analyses."
keypoints:
- "ENACTS construction relies on the precise integration of ground-based rain gauges and satellite products."
- "Gridding methodologies must account for varying levels of station density to ensure spatial accuracy."
- "Rain gauge analysis provides essential ground-truth context for validating observational data readiness."
---

# Observational Data Readiness

This episode is a hands-on exercise in the Climate Data Tools (CDT) software.

## Download Data

### Download Country Administrative Boundaries

The menu **Data Preparation** ‣ **Download Country boundary** allows you to download country outlines and administrative subdivisions for countries in Africa from [GADM data](https://gadm.org/index.html).

![CDT country boundary download dialog](../fig/cdt_1.png)

1. Select the region where the country is located.
2. Select the country from the drop-down list.
3. Select the administrative subdivisions to download:
   - **0** is the country boundaries,
   - **1** is the first subdivision (example: Region),
   - **2** is the second subdivision (example: District).
4. Specify the folder to save the downloaded ESRI shapefile, either by clicking the browse button <img src="../fig/cdt_dots.png" alt="browse button" height="20" style="margin:0 -4px;vertical-align:middle"> or by typing the full path to the folder.

### Download Satellite Rainfall Estimates Data

The menu **Data Preparation** ‣ **Download Data** ‣ **Download Satellite Rainfall Estimates data** allows you to download satellite rainfall products at temporal resolutions from 30 minutes to monthly.

![CDT satellite rainfall download dialog](../fig/cdt2.png)

1. Select the temporal resolution of the satellite rainfall estimate product to download.
2. For minute or hourly data, select the time step.
3. Select the satellite rainfall estimate product to download.
4. Set the date and time range of the data to download. See [Setting date range](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/cdt_data_input.html#cdt.date.range) for more details.
5. If the product requires a login, enter the user name and password of your account. The login website is shown next to the **Login to** label; if you do not have an account yet, you can sign up there.
6. Click this button to display information about the data. It opens your default web browser with a link to the data or relevant information about it.
7. If this checkbox is available, check it to download the data from the IRI Data Library. The Data Library lets you subset the data before downloading it, which makes the download faster.
8. Set the bounding box of the area to download: type the minimum and maximum longitude and latitude in decimal degrees.
9. Specify the folder to save the downloaded rainfall data, either with the browse button <img src="../fig/cdt_dots.png" alt="browse button" height="10" style="vertical-align:middle"> or by typing the full path to the folder.

A folder containing the downloaded data is created under the directory you provided in step 9. It is named after the satellite rainfall estimate product and the time step of the data.

## Spatial Interpolation

Spatial interpolation estimates the unknown value at a point from measured points in the study area. To interpolate point data (station observations) into gridded data, use the menu **Gridding** ‣ **Spatial Interpolation**. It displays a tabbed widget on the left panel where you enter the input data, set the interpolation parameters and display maps of the interpolated data.

The **Input Data** tab lets you specify the input station data and the interpolation parameters.

![CDT spatial interpolation input data tab](../fig/cdt3.png)

1. Select the temporal resolution of the input data in CDT station data format. The available temporal resolutions are minutes, hourly, daily, pentad, dekadal, monthly and others (seasonal, annual, …).
2. For minute and hourly data, select the time step of the input data.
3. Select the CDT station data to interpolate from the drop-down list if it is already loaded, or open it with the browse button <img src="../fig/cdt_dots.png" alt="browse button" height="10" style="vertical-align:middle">.
4. Set the start and end date of the period to interpolate. See [setting date range](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/cdt_data_input.html#cdt.date.range) for more details.
5. Click **Interpolation Parameters** to set the parameters used to interpolate the station data. This opens a dialog box.

   > CDT has 7 spatial interpolation methods: Inverse Distance Weighted, Ordinary Kriging, Universal Kriging, Modified Shepard interpolation, Spheremap interpolation, Nearest Neighbor, and Nearest Neighbor with elevation (3D).
   >
   > By default CDT uses a local interpolation technique, so you have to provide the minimum and maximum number of stations used to interpolate a grid point, and the maximum distance within which stations are selected. See the [spatial interpolation methods](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/interpolation_methods.html#interp.local) for further details, and the [interpolation dialog](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/interpolation_dialog.html) to select the method and set the interpolation parameters.

6. Click **Create Grid for Interpolation** to provide the grid used to interpolate the data. It displays the [grid creation](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/interpolation_grid.html) dialog box.
7. If the variable cannot be negative or below a certain value, check **Change negative values to** and enter the replacement value in the input field that becomes active.
8. To mask the grid outside given polygons, check **Blank grid outside the Shapefile**. This activates the drop-down list below it, where you provide the shapefile to use. See the [blanking options](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/blanking_options.html) to control the buffer outside the polygons.
9. Specify the folder to save the output, either with the browse button <img src="../fig/cdt_dots.png" alt="browse button" height="10" style="vertical-align:middle"> or by typing the full path to the folder.

When you have set everything in this tab, click **Interpolate** to start the interpolation.

This creates a folder named `INTERPOLATION_<file name of station data>` under the output folder you provided in step 9. Inside this folder, two directories and one file are created:

- `CDTDATASET`: a directory containing all the files CDT needs. For ordinary and universal kriging, the fitted variogram models are saved here in the file `Fitted_variogram.rds`.
- `DATA_NetCDF`: a directory containing the interpolated data in netCDF format.
- `SpatialInterpolation.rds`: an index file. You need it to reload the interpolation outputs if you want to display them later.

The **Maps** tab displays maps of the interpolated data.

![CDT spatial interpolation maps tab](../fig/cdt4.png)

1. If you already ran the interpolation and want to display its maps, check **Data already computed**. This activates the **Load Data** button; use it to browse to the file `SpatialInterpolation.rds`.
2. Enter the date you want to plot, then click **PLOT** to display the map. Use `<<` to display the map for the previous date and `>>` for the next date.
3. Select the type of map for the station data: **Points** or **Pixels**. See [the type of plot for station data](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/plot_types.html#plot.type.stn).
4. Select the type of map for the gridded data: **Pixels** or **FilledContour**. See [the type of plot for gridded data](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/plot_types.html#plot.type.grd).
5. Click **Options** to change the color bar, title and labels of the maps. For details, see [map options](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/plot_options.html#map.options).

## Merging Data

### Bias Adjustment

All estimated gridded data (satellite rainfall estimates, model output, …) are subject to errors from sensors, algorithms and chance. There are two main sources of error: random error and systematic bias. These discrepancies between the gridded estimates and the station observations need to be corrected before further analysis. Systematic bias can be reduced or eliminated with statistical bias correction methods.

#### Computing Bias Coefficients

CDT corrects systematic bias using historical gridded estimates and station observations. Climatological bias coefficients are first computed for each month, or at the temporal resolution of the data.

The light blue color of the menu means you only need to compute the bias coefficients once. You will need a long, quality-controlled station series to compute them. Once the coefficients are calculated, you can reuse them to correct the systematic bias of gridded estimates from the same source.

The menu **Gridding** ‣ **Merging Climate Data** ‣ *name of the climate variable* ‣ **Compute Bias Coefficients** computes the climatological bias coefficients for rainfall, temperature, relative humidity, radiation and pressure data.

![CDT compute bias coefficients dialog](../fig/cdt5.png)

1. Select the temporal resolution of the data: daily, pentad, dekadal or monthly.
2. Click **Set Bias Base Period** to specify the period and the minimum number of years used to compute the bias coefficients. All the parameters are set in the [base period setting](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/selecting_period.html#base.period) dialog.
3. Select the file containing the station data (in CDT station data format) from the drop-down list if it is already loaded, or open it with the browse button <img src="../fig/cdt_dots.png" alt="browse button" height="10" style="vertical-align:middle">.
4. Type the full path to the folder containing the netCDF files, or browse to it with the button on the right.
5. Provide a sample netCDF file and set the filename format of the netCDF files in the [netCDF settings](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/cdt_data_input.html#cdt.ncdf.setting) dialog, opened with the **Settings** button.
6. Type the full path to the folder to save the computed bias coefficients, or use the browse button.
7. Select the method used to compute the bias coefficients. CDT has 4 methods: Multiplicative Bias Time Step Variable, Multiplicative Bias for Each Month, Quantile Mapping with Fitted Distribution, and Quantile Mapping with Empirical Distribution. See [CDT bias adjustment methods](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/bias_adjustment_methods.html) for more information.
8. Enter the minimum number of non-missing values used to calculate the bias coefficients or fit the distribution. Even with a minimum number of years set in step 2, some dekads and months may contain missing values, so this sets the minimum length of data used for the computation.
9. Click **Bias Interpolation Parameters** to open the [interpolation dialog](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/interpolation_dialog.html), where you select the interpolation method and set all its parameters.
10. Click **Create Grid for Interpolation** to create the grid used to interpolate the data. It displays the [grid creation](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/interpolation_grid.html) dialog box.

Click the **Run** button to start the bias coefficient calculation.

A folder named `BIAS_Data_<file name of station data>` is created under the output folder you provided in step 6. It contains a file named `BIAS_PARAMS.rds` with the parameters used to compute the bias coefficients, plus one or more files that depend on the method:

- **Multiplicative Bias Time Step Variable**: netCDF files named `STN_GRD_Bias_Coeff_<n>.nc`, where `n` is the index of the bias coefficient. The total number of coefficients N is 365 for daily data, 72 for pentadal data, 36 for dekadal data and 12 for monthly data.
- **Multiplicative Bias for Each Month**: netCDF files named `STN_GRD_Bias_Coeff_<n>.nc`, where `n` is the month.
- **Quantile Mapping with Fitted Distribution**: netCDF files named `STN_Bias_Pars_<distname>_<n>.nc`, containing the computed and interpolated distribution parameters for the station data, and `GRD_Bias_Pars_<distname>_<n>.nc` for the gridded data. `distname` is the fitted distribution (`BernGamma` for rainfall, `Gaussian` for the other variables) and `n` is the month.
- **Quantile Mapping with Empirical Distribution**: a file named `GRID_DATA.rds` containing the quantile functions of the empirical distribution.

### Merging Climate Data

The menu **Gridding** ‣ **Merging Climate Data** ‣ *name of the climate variable* ‣ **Merging Data** merges station observations with gridded data. The climate variable is one of rainfall, temperature, relative humidity, radiation or pressure.

The **Input Data** tab specifies the input station and gridded data, the period over which the data will be merged, and the folder to save the output.

![CDT merging input data tab](../fig/cdt6.png)

1. Select the temporal resolution of the data: daily, pentad, dekadal or monthly.
2. Set the start and end date of the period over which the data will be merged. See [setting date range](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/cdt_data_input.html#cdt.date.range) for more details.
3. Select the file containing the station data (in CDT station data format) from the drop-down list if it is already loaded, or open it with the browse button <img src="../fig/cdt_dots.png" alt="browse button" height="10" style="vertical-align:middle">.
4. Type the full path to the folder containing the netCDF files, or browse to it with the button <img src="../fig/cdt_dots.png" alt="browse button" height="10" style="vertical-align:middle"> on the right.
5. Provide a sample netCDF file and set the filename format of the netCDF files in the [netCDF settings](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/cdt_data_input.html#cdt.ncdf.setting) dialog, opened with the **Settings** button.
6. Type the full path to the folder to save the merged data, or use the browse button <img src="../fig/cdt_dots.png" alt="browse button" height="10" style="vertical-align:middle">.
7. Specify the netCDF filename format of the output merged files.

The **Merging Parameters** tab selects the method and sets all the parameters for the merging.

![CDT merging parameters tab](../fig/cdt7.png)

1. Select the merging method. CDT has 4 merging methods: Simple Bias Adjustment, Regression Kriging, Cressman Scheme and Barnes Scheme. See [CDT merging methods](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/merging_methods.html) for more information.

   > For Regression Kriging, you have to provide the auxiliary variables to use. A frame is displayed where you select one or more of: elevation, slope, aspect, longitude and latitude. If you include elevation, slope or aspect, you need to provide elevation data (in netCDF format); slope and aspect are computed from that digital elevation model.

   ![CDT regression kriging auxiliary variables](../fig/cdt8.png)

2. Specify the number of nested runs to perform, then enter the pass ratio for each run, separated by commas. The number of runs must equal the number of pass ratios. See [CDT merging nested run](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/merging_methods.html#merging.nested.run) for more information.
3. Click **Merging Interpolation Parameters** to open the [interpolation dialog](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/interpolation_dialog.html), where you select the interpolation method and set all its parameters.

   ![CDT merging interpolation parameters dialog](../fig/cdt9.png)

   > By default, CDT uses a fixed search radius defined by `maxdist`: all stations used to interpolate a grid node are selected within that radius. Alternatively, CDT can select the stations for each grid node from `nmin` and `nmax` alone, so `maxdist` varies: small where stations are dense and large where they are sparse. Check **Use variable radius of influence for each grid** to use a variable `maxdist`. See [variable search radius](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/interpolation_methods.html#var.maxdist) for more details.

4. Click **Create Grid for Interpolation** to create the grid used to interpolate the data. It displays the [grid creation](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/interpolation_grid.html) dialog.
5. Check **Blank merged data using ESRI shapefiles** to mask the grid outside the polygons of an ESRI shapefile (blanking). Before running the merging, set how to add or remove the buffer outside the polygons. See [blanking options](https://iri.columbia.edu/~rijaf/CDTUserGuide/html/blanking_options.html) for more details.
6. For rainfall data, check **Apply Rain-no-Rain Mask** to mask the areas where the stations observed no rain. Specify the threshold that defines no rain: a binary field is created, set to zero where rainfall is below the threshold and one above it. To prevent steep gradients in the interpolated mask, check **Smooth Rain-no-Rain Mask** to apply spatial smoothing to it.

Before running the merging, change the merging options if needed with the R function `merging.options()`. You can change the minimum number of stations used for the merging, for regression kriging, or for fitting the variogram if you use ordinary kriging for the interpolation. You can also change the size of the block used for spatial smoothing. The default values of `merging.options()` are:

```
merging.options()
## $mrgMinNumberSTN
## [1] 10
##
## $rkMinNumberSTN
## [1] 20
##
## $vgmMinNumberSTN
## [1] 20
##
## $useLocalInterpolation
## [1] TRUE
##
## $powerWeightIDW
## [1] 2
##
## $powerWeightShepard
## [1] 0.7
##
## $powerWeightBarnes
## [1] 0.5
##
## $addCoarseGrid
## [1] TRUE
##
## $resCoarseGrid
## [1] 0.5
##
## $saveGridBuffer
## [1] FALSE
##
## $dirGridBuffer
## [1] "~"
##
## $saveRnoR
## [1] FALSE
##
## $dirRnoR
## [1] "~"
##
## $RnoRModel
## [1] "logit"
##
## $RnoRCutOff
## [1] 3
##
## $RnoRaddCoarse
## [1] TRUE
##
## $RnoRUseMerged
## [1] FALSE
##
## $RnoRSmoothingPixels
## [1] 2
##
## $blockType
## [1] "gaussian"
##
## $blockSize
## [1] 1 1
```

`dirGridBuffer` and `dirRnoR` default to your home directory. To get help on the available arguments of this function, type the following into the R console:

```
?merging.options
```

Click the **Run** button to start the merging. A folder named `MERGED_<name of climate variable>_Data_<start date>_<end date>` is created under the output folder you provided. It contains the netCDF files of the merged data and a log file.
