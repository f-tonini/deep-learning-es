# -*- coding: utf-8 -*-
"""
ArcGIS Python Toolbox (".pyt" file)
@author: Francesco Tonini <ftonini84@gmail.com>
"""

import arcpy, sys, os, cv2, random, time
import tensorflow as tf
import numpy as np
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
from shapely.geometry import Point
from urllib.error import URLError, HTTPError
from urllib.request import urlopen
from fastparquet import ParquetFile
import importlib
# importlib.reload(sys)  # force reload of the module
# importlib.reload(os)  # force reload of the module
# importlib.reload(cv2)  # force reload of the module
# importlib.reload(tensorflow)  # force reload of the module
# importlib.reload(numpy)  # force reload of the module
# importlib.reload(pandas)  # force reload of the module
# importlib.reload(URLError)  # force reload of the module
# importlib.reload(HTTPError)  # force reload of the module
# importlib.reload(urlopen)  # force reload of the module

arcpy.env.overwriteOutput = True
#arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(3857)
#arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(4326)

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Cultural ES Toolbox"
        self.alias = "deeplearningES"
        # List of tool classes associated with this toolbox
        self.tools = [CulturalES]

class CulturalES(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Cultural Ecosystem Services"
        self.description = "Cultural ecosystem services are the defined as the nonmaterial benefits people obtain from ecosystems through spiritual enrichment, " + \
                           "cognitive development, reflection, recreation, and aesthetic experiences. This tools maps the density of natural landscapes from user-uploaded " + \
                           "photos on the Flickr social media platform"
        self.canRunInBackground = False
        self.RETRAINED_LABELS_TXT_FILE_LOC = os.path.join(os.getcwd(), "retrained_labels.txt")
        self.RETRAINED_GRAPH_PB_FILE_LOC = os.path.join(os.getcwd(), "retrained_graph.pb")

    def getParameterInfo(self):
        """Define parameter definitions"""
        
        # Input Area of Interest Parameter
        in_aoi = arcpy.Parameter(
            displayName="Area of Interest",
            name="in_aoi",
            datatype="GPFeatureRecordSetLayer",
            parameterType="Required",
            direction="Input")

        # Use __file__ attribute to find the .lyr file (assuming the
        #  .pyt and .lyr files exist in the same folder)
        in_aoi.value = os.path.join(os.path.dirname(__file__),
                                    'AOI_poly_es.lyrx')
                                    
        # Input Ecosystem Service Type Parameter
        in_es_type = arcpy.Parameter(
            displayName="Ecosystem Service Type",
            name="in_es_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        
        in_es_type.value = "Nature"

        #Use the following lines instead if you want to have a dropdown menu with multiple values!
        #in_es_type.filter.type = "ValueList"
        #in_es_type.filter.list = ["Nature", "Wildlife"]

        # Input Sample Size Parameter
        in_n_photos = arcpy.Parameter(
            displayName="Sample Fraction",
            name="in_n_photos",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")

        in_n_photos.value = 0.3
        
        # Input Start Year Parameter
        in_start_year = arcpy.Parameter(
            displayName="Start Year (>= 2005)",
            name="in_start_year",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")

        # Input End Year Parameter
        in_end_year = arcpy.Parameter(
            displayName="End Year (<= 2017)",
            name="in_end_year",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")

        # Input Probability Threshold Parameter
        in_prob_threshold = arcpy.Parameter(
            displayName="Probability Threshold",
            name="in_prob_threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")

        in_prob_threshold.value = 0.5

        # Input Grid Type
        in_grid_type = arcpy.Parameter(
            displayName="Grid Type",
            name="in_grid_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input")

        in_grid_type.filter.type = "ValueList"
        in_grid_type.filter.list = ["HEXAGON", "SQUARE", "TRIANGLE"]
        in_grid_type.value = "HEXAGON"

        # Input Probability Threshold Parameter
        in_cell_size = arcpy.Parameter(
            displayName="Cell Size",
            name="in_cell_size",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input")

        # Derived Output Features Parameter
        out_features = arcpy.Parameter(
            displayName="Output Density Layer",
            name="out_features",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output")

        # Use __file__ attribute to find the .lyr file (assuming the
        #  .pyt and .lyr files exist in the same folder)
        #out_features.symbology = os.path.join(os.path.dirname(__file__), 
        #                            'raster_symbology.lyrx')

        parameters = [in_aoi, in_es_type, in_n_photos, in_start_year, in_end_year, in_prob_threshold, in_grid_type, in_cell_size, out_features]
                  
        return parameters

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        if parameters[2].value and parameters[2].value <= 0:
            parameters[2].setErrorMessage("The sample fraction must be > 0 and <= 1!")
        elif parameters[2].value and parameters[2].value > 1:
            parameters[2].setErrorMessage("The sample fraction must be > 0 and <= 1!")
        else:
            parameters[2].clearMessage()

        if parameters[3].value and parameters[3].value < 2005:
            parameters[3].setErrorMessage("The start year cannot be smaller than 2005!")
        else:
            parameters[3].clearMessage()

        if parameters[4].value and parameters[4].value > 2017:
            parameters[4].setErrorMessage("The end year cannot be greater than 2017!")
        else:
            parameters[4].clearMessage()
                    
        if parameters[3].value and parameters[4].value:
            if parameters[3].value > parameters[4].value:
                parameters[3].setErrorMessage("The start year cannot be greater than the end year!")
                parameters[4].setErrorMessage("The end year cannot be smaller than the start year!")
            else:
                parameters[3].clearMessage()
                parameters[4].clearMessage()

        return

    def checkIfNecessaryPathsAndFilesExist(self):
        if not os.path.exists(self.RETRAINED_LABELS_TXT_FILE_LOC):
            arcpy.AddMessage('ERROR: RETRAINED_LABELS_TXT_FILE_LOC "' + self.RETRAINED_LABELS_TXT_FILE_LOC + '" does not seem to exist')
            return False

        if not os.path.exists(self.RETRAINED_GRAPH_PB_FILE_LOC):
            arcpy.AddMessage('ERROR: RETRAINED_GRAPH_PB_FILE_LOC "' + self.RETRAINED_GRAPH_PB_FILE_LOC + '" does not seem to exist')
            return False
            
        return True

    def create_heatmap(self, geo_data, aoi_ext, grid_type, cell_size, out_map):
        arcpy.AddMessage("Creating point density file and heatmap...")
        # # GeoDataFrame needs a shapely object, so we create a new column Coordinates 
        # as a tuple of Longitude and Latitude
        #geo_data.drop_duplicates(subset=['lon', 'lat'], keep='last', inplace=True)
        geo_data['Coordinates'] = list(zip(geo_data['lon'], geo_data['lat']))
        # Transform tuples to Point
        geo_data['Coordinates'] = geo_data['Coordinates'].apply(Point)
        geo_data.drop(["lon", "lat"], axis=1, inplace=True)
        # Now, we can create the GeoDataFrame by setting geometry with the coordinates created previously
        crs = {'init': 'epsg:4326'} #WGS84 CODE
        gdf = gpd.GeoDataFrame(geo_data, crs=crs, geometry='Coordinates')
        pnt_feature = os.path.join(arcpy.env.scratchFolder, 'naturePnt.shp')
        gdf.to_file(pnt_feature)

        #Project Points to Web Mercator (meters)
        # pnt_feature_prj = os.path.join(arcpy.env.scratchFolder, 'naturePnt_prj.shp')
        # arcpy.Project_management(pnt_feature, pnt_feature_prj, arcpy.SpatialReference(3857))

        arcpy.AddMessage("Creating tessellated area...")
        out_grid_FC = os.path.join(arcpy.env.scratchFolder, 'gridAOI.shp')
        if cell_size != None:
            arcpy.GenerateTessellation_management(Output_Feature_Class=out_grid_FC, Extent=aoi_ext, Shape_Type=grid_type, Size=cell_size, Spatial_Reference=arcpy.SpatialReference(4326)) #WGS84 CODE
        else:
            arcpy.GenerateTessellation_management(Output_Feature_Class=out_grid_FC, Extent=aoi_ext, Shape_Type=grid_type, Spatial_Reference=arcpy.SpatialReference(4326)) #WGS84 CODE
        
        arcpy.AddMessage("Spatial join and calculate point counts...")
        arcpy.SpatialJoin_analysis(out_grid_FC, pnt_feature, out_map, match_option="COMPLETELY_CONTAINS")

        return

    def classify_images(self, aoi_ext, sample_frac, startYear, endYear, prob_threshold):
        arcpy.AddMessage("Starting image classification module...")
        if not self.checkIfNecessaryPathsAndFilesExist():
            return
    
        # load the graph from file
        with tf.gfile.FastGFile(self.RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
            # instantiate a GraphDef object
            graphDef = tf.GraphDef()
            # read in retrained graph into the GraphDef object
            graphDef.ParseFromString(retrainedGraphFile.read())
            # import the graph into the current default Graph, note that we don't need to be concerned with the return value
            _ = tf.import_graph_def(graphDef, name='')

        with tf.Session() as sess:
            
            #1. READ PHOTO DATABASE
            arcpy.AddMessage("Fetching photos from database...")
            file_parquet = './flickrDB_small_mod.parq'
            pf = ParquetFile(file_parquet)
            photoDB_df = pf.to_pandas()

            #2. QUERY PHOTO DATABASE BY EXTENT OF USER-DEFINED AREA OF INTEREST AND YEARS RANGE
            photoDB_df = photoDB_df[(photoDB_df[u'latitude'] <= aoi_ext['YMax']) & (photoDB_df[u'latitude'] >= aoi_ext['YMin']) \
                                    & (photoDB_df[u'longitude'] <= aoi_ext['XMax']) & (photoDB_df[u'longitude'] >= aoi_ext['XMin']) \
                                    & (photoDB_df[u'year'] <= int(endYear)) & (photoDB_df[u'year'] >= int(startYear))]

            #3. SAMPLE DOWN NUMBER OF PHOTOS BASED ON USER-DEFINED SAMPLE SIZE
            photoDB_df = photoDB_df.sample(frac=float(sample_frac))
            if photoDB_df.empty:
                arcpy.AddError("No photos can be found within the selected spatial extent! Please modify your selection")
                arcpy.ExecuteError
            else:
                links = photoDB_df[u'url_sq']
                lat = photoDB_df[u'latitude']
                lon = photoDB_df[u'longitude']

                #4. CLASSIFY TARGET LABELS USING INCEPTION V3 DNN MODEL FROM GOOGLE 
                # ---model retrained previously---
                arcpy.AddMessage("Running neural network for image classification...")
                pred_lst = []
                for row_lat, row_lon, row_link in zip(lat, lon, links):
                # if the file does not end in .jpg or .jpeg (case-insensitive), continue with the next iteration of the for loop
                    if not (row_link.lower().endswith(".jpg") or row_link.lower().endswith(".jpeg")):
                        continue
                    #open image from url
                    try:
                        url_response = urlopen(row_link)
                    except:
                        arcpy.AddWarning("Photo link does not exist anymore!") 
                        continue
                    else:
                        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
                        #openCVImage = cv2.imread(img_array)
                        openCVImage = cv2.imdecode(img_array, -1)
                        # if we were not able to successfully open the image, continue with the next iteration of the for loop
                        if openCVImage is None:
                            arcpy.AddMessage("Unable to open " + row_link + " as an OpenCV image")
                            continue
                        # get the final tensor from the graph
                        finalTensor = sess.graph.get_tensor_by_name('final_result:0')
                        # convert the OpenCV image (numpy array) to a TensorFlow image
                        try:
                            tfImage = np.array(openCVImage)[:, :, 0:3]
                            # run the network to get the predictions    
                            predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})
                            prob_nature = predictions[0][0]
                            label = 1 if prob_nature >= float(prob_threshold) else 0
                            if label == 1:
                                pred_lst.append([row_lat, row_lon, label])
                            else:
                                continue
                        except:
                            arcpy.AddError("Could not compute predictions!")
                            raise arcpy.ExecuteError

                colNames = ['lat', 'lon', 'label']
                predict_df = pd.DataFrame(pred_lst, columns = colNames) 

            # write the graph to file so we can view with TensorBoard
            arcpy.AddMessage("Saving final graph file...")
            tfFileWriter = tf.summary.FileWriter("./output_graph")
            tfFileWriter.add_graph(sess.graph)
            tfFileWriter.close()

        return predict_df
        
    def execute(self, parameters, messages):
        """The source code of the tool."""
        inAOI = parameters[0].valueAsText
        esType = parameters[1].valueAsText
        sampleFrac = parameters[2].valueAsText
        startYear = parameters[3].valueAsText
        endYear = parameters[4].valueAsText
        pThreshold = parameters[5].valueAsText
        grdType = parameters[6].valueAsText
        cellSize = parameters[7].valueAsText
        outFC = parameters[8].valueAsText

        try:
            feature_lyr = r"in_memory\aoi_fs"
            #Create Feature Layer from User-defined AOI polygon
            arcpy.MakeFeatureLayer_management(inAOI, out_layer=feature_lyr)
            #Read the Extent (Xmin, Xmax, Ymin, Ymax)
            desc = arcpy.Describe(feature_lyr)
            extent = desc.extent
            aoi_ext = {
                'XMin': extent.XMin, 
                'XMax': extent.XMax, 
                'YMin': extent.YMin, 
                'YMax': extent.YMax
            }

            self.checkIfNecessaryPathsAndFilesExist()
            #run image classification using Google inception CNN
            pred_df = self.classify_images(aoi_ext, sampleFrac, startYear, endYear, pThreshold)
            #create a point density (heatmap) from the classified images and return output to ArcPro
            if not cellSize:
                cellSize = None
            else:
                cellSize = float(cellSize)
            self.create_heatmap(pred_df, extent, grdType, cellSize, outFC)

        except arcpy.ExecuteError:    
            arcpy.AddError(arcpy.GetMessages(2))   

        return
