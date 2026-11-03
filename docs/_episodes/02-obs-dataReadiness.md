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
   
4. Specify the folder to save the downloaded ESRI shapefile by browsing to it using the browse button ![](../fig/cdt_dots.png =20x20) or typing the full path to the folder.


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

