---
title: Demo 2 
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

<!-- MathJax -->
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# Observational Data Readiness

## Download Data
## Download Country Administrative Boundaries

The menu [**Data Preparation** ‣ **Download Country boundary**]() allows you to download country outlines and administrative subdivisions for countries in Africa from [GADM data](https://gadm.org/index.html).

![CDT Download Interface](../fig/cdt_1.png)

1. Select the region where the country is located.
2. Select the country from the drop-down list.
3. Select the administrative subdivisions to download.
   
   **0** is the country boundaries,
   
   **1** is the first subdivision (example: Region),
   
   **2** is the second subdivision (example: District),
   
4. Specify the folder to save the downloaded ESRI shapefile by browsing to it using the browse button <img src="../fig/cdt_dots.png" height="20"> or typing the full path to the folder.


## Download Satellite Rainfall Estimates data 

The menu Data Preparation ‣ Download Data ‣ Download Satellite Rainfall Estimates data   allows to download satellite rainfall products from 30 minutes to monthly temporal resolution.

![CDT Download Interface](../fig/cdt2.png)


1.	Select the temporal resolution of the satellite rainfall estimate product to download.
2.	In case of minutes or hourly data, select here the time step.
3.	Select the satellite rainfall estimate product to download.
4.	Set the date and time range of the data to download. See Setting date range for more details.
5.	If the satellite rainfall estimate product requires a login to download the data, enter here the user name and the password you use to log in to your account. The URL of the website for login is displayed next to the label Login to. If you do not have an account yet, you can sign up to this URL.
6.	Click this button to display the information about the data. It will open your default web browser and display the link to the data or relevant information about the data.
7.	In case of this checkbox is activated, check it if you want to download the data from the Data Library of IRI. The IRI Data Library allows you to subset the data before downloading it into your computer, and it makes the download faster.
8.	Set the bounding box of the area to be downloaded. Type in the fields the minimum and maximum of the longitude and latitude in degree decimal.
9.	Specify the folder to save the downloaded rainfall data by browsing it from the button  or typing the full path to the folder.

A folder containing the downloaded data will be created under the directory you provided in (9). This folder is created from the name of the satellite rainfall estimate product and the time step of the data


## Spatial Interpolation

A spatial interpolation is the estimation of the unknown value of a point by using the measured points in the study area. To interpolate a spatial points data (stations observation) into gridded data, use the menu Gridding ‣ Spatial Interpolation. It displays a tabbed widget on the left panel, allows to enter the inputs data, set the interpolation parameters and display maps of the interpolated data.

The tab Input Data allows to specify the input station data and the interpolation parameters.

![CDT Download Interface](../fig/cdt3.png)


1.	Select the temporal resolution of the input data in CDT stations data format. The available temporal resolution are: minutes, hourly, daily, pentad, dekadal, monthly data and others (seasonal, annual, …).
2.	In case of minutes and hourly, select here the time step of the input data.
3.	Select the CDT station data to interpolate with the drop down list if it is already loaded, or open it through   …  .
4.	Set the start and end date of the period to interpolate. See setting date range for more details.
5.	Click on the button   Interpolation Parameters   to set the parameters to be used to interpolate the station data. It opens a dialog box.
 > CDT has 7 spatial interpolation methods: Inverse Distance Weighted, Ordinary Kriging, Universal Kriging, Modified Shepard interpolation, Spheremap interpolation method, Nearest Neighbor and Nearest Neighbor with elevation - 3D.

 > By default CDT uses a local interpolation technique, therefore, you have to provide the minimum and maximum number of stations used to interpolate a grid point, and the maximum distance within which the stations are selected. See the spatial interpolation methods for further details, and the interpolation dialog to select the method to be used and set the interpolation parameters.

6.	Click on the button   Create Grid for Interpolation   to provide the grid to be used to interpolate the data. It will display the grid creation dialog box.
7.	In the case where the variable to be interpolated cannot be negative or below a certain value, check    Change negative values to    , enter the value you want to change the negative values in the activated input field.
8.	To mask the grid outside a given polygons, check    Blank grid outside the Shapefile    , this will activate the drop down list below allowing you to provide the Shapefile to be used. See the blanking options to control the buffer outside the polygons.
9.	Specify the folder to save the output by browsing it from the button  or typing the full path to the folder.

When you finish to set everything in this tab, click on the button Interpolate to start the interpolation process.


This creates a folder named __INTEPORLATION__<file name of station data>__under the folder you provided (9) to save the outputs. Inside this folder 2 directories and 1 file are created:
•	__CDTDATASET__: directory containing all the required files used by CDT; for ordinary and universal kriging, the fitted variogram models are saved under this folder inside the file Fitted_variogram.rds
•	__DATA_NetCDF__:directory containing the interpolated data in netCDF format
•	__SpatialInterpolation.rds__: an index file, you will need to reload the outputs of the interpolation if you want to display it later


The tab Maps allows to display the maps of the interpolated data.

![CDT Download Interface](../fig/cdt4.png)


1.	If you have already performed the interpolation before and you want to display the maps of the interpolated data,
check   Data already computed , this will activated the button Load Data allowing you to load the index file of the interpolated data. Browse the file SpatialInterpolation.rds via the open dialog box.
2.	Enter the date you want to plot in the input fields then click the button PLOT to display the map. You can use the button `<<` to display the map of the previous interpolated data and `>>` for the next date.
4.	Select the type of map to display for the stations data, available options are Points and Pixels. See the type of plot for station data.
5.	Select the type of map to display for the gridded data, available options are Pixels and FilledContour. See the type of plot for gridded data.
6.	Click on the button Options to change the color bar, title and labels of the maps. For more details on how to change the options, see map options.


## Merging Data

### 1. Bias Adjustment  {-}


All estimated gridded data (satellite rainfall estimates, models output, …) are subject to different types of errors, including errors from sensors, algorithms and random errors.
There are two main sources errors for the estimated gridded data: random error and systematic bias. These discrepancies between the estimated gridded data and the station observations need to be corrected before further analysis. The systematic bias can be reduced or eliminated by applying statistical bias correction methods.

__Computing Bias coefficients__

CDT uses an approach based on historical data from the estimated gridded data and station observations to correct the systematic bias. A climatological bais coefficients are first computed for each month or at the temporal resolution of the data.
The color light blue of the menu means that you only need to compute the bias coefficients once. Indeed, you will need a long series and quality controlled station data to compute the climatological bias coefficients. Once the coefficients are calculated you can use it to correct the systematic bias from the same source of the estimated gridded data.


__Climate data__

The menu Gridding ‣ Merging Climate Data ‣ <name of the climate variable > ‣ Compute Bias Coefficients   allows to compute the climatological bias coefficients for rainfall, temperature, relative humidity, radiation and pressure data.

![CDT Download Interface](../fig/cdt5.png)


1. Select the temporal resolution of the data. Available temporal resolution are: daily, pentad, dekadal and monthly data.
2.	Click on the button     Set Bias Base Period     to specify the period and minimum number of year to be use to compute the bias coefficients. All the parameters can be set in the base period setting dialog.
3.	Select the file containing the station data (in CDT station data format) to use from the drop down list if it is already loaded, or open it through  .
4.	Type the full path to the folder containing the netCDF files or browse it from the button  on the right.
5.	Provide a netCDF sample file and change the filename format of the netCDF file through the netCDF settings dialog by clicking on the button   Settings  .
6.	Type the full path to the folder to save the computed bias coefficients, or use the browse button  .
7.	Select the method to use to compute the bias coefficients. CDT has 4 methods to compute the bias coefficients: Multiplicative Bias Time Step Variable, Multiplicative Bias for Each Month, Quantile Mapping with Fitted Distribution, and Quantile Mapping with Empirical Distribution. See CDT bias adjustment methods section for more information.
8.	Enter the minimum length of non missing values to be used to calculate the bias coefficients or to fit the distribution. Although you already specify the minimum number of year needed to compute the bias coefficients, it is possible that some dekads and months contain missing values, you need to specify here the minimum length of data to be used to perform the computation.
9.	Click on the button     Bias Interpolation Parameters     to open the interpolation dialog to select the interpolation method to be used and set the all parameters for the interpolation.
10.	Click on the button     Create Grid for Interpolation     to create the grid to be used to interpolate the data. It will display the grid creation dialog box.


