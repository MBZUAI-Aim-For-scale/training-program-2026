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

The menu **Data Preparation** ‣ **Download Country boundary** allows you to download country outlines and administrative subdivisions for countries in Africa from [GADM data](https://gadm.org/index.html).

![CDT Download Interface](../fig/cdt_1.png)

1. Select the region where the country is located.
2. Select the country from the drop-down list.
3. Select the administrative subdivisions to download.
   
   **0** is the country boundaries,
   
   **1** is the first subdivision (example: Region),
   
   **2** is the second subdivision (example: District),
   
4. Specify the folder to save the downloaded ESRI shapefile by browsing to it using the browse button ![browse button]({{ site.baseurl }}/fig/cdt_dots.png) or typing the full path to the folder.


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


