import pandas as pd
import numpy as np
import warnings
import os
import simplekml
warnings.filterwarnings('ignore')
import pickle
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from dijkstra import *
import networkx as nx
import math

class TerrainLookup():
    def __init__(self):
        with open('CA_Terrain_rev.pickle', 'rb') as handle:
            self.terrain = pickle.load(handle)

    def distance_haversine(self, A, B):
        #Raman - distance in miles
        radius=6371000
        dLat = np.radians(B[0]-A[0])
        dLon = np.radians(B[1]-A[1])
        lat1 = np.radians(A[0])
        lat2 = np.radians(B[0])
        a = np.sin(dLat/2) * np.sin(dLat/2) + np.sin(dLon/2) * np.sin(dLon/2) * np.cos(lat1) * np.cos(lat2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return c * radius * 0.000621371

    def bearing(self, A, B):
        dLon = np.radians(B[1]-A[1])
        lat1 = np.radians(A[0])
        lat2 = np.radians(B[0])
        y = np.sin(dLon) * np.cos(lat2)
        x = np.cos(lat1)* np.sin(lat2) - np.sin(lat1) * np.cos(lat2)* np.cos(dLon)
        return round(np.degrees(np.arctan2(y, x)) % 360,2)

    def dist_coord(self, lat1, lon1, brng, d):
        #Raman - d in meters
        R = 6378.1 #Radius of the Earth
        brng = np.radians(brng)
        d= d/1000   
        lat1 = np.radians(lat1) #Current lat point converted to radians
        lon1 = np.radians(lon1) #Current long point converted to radians   
        lat2 = np.arcsin( np.sin(lat1)*np.cos(d/R) + np.cos(lat1)*np.sin(d/R)*np.cos(brng))   
        lon2 = lon1 + np.arctan2(np.sin(brng)*np.sin(d/R)*np.cos(lat1), np.cos(d/R)-np.sin(lat1)*np.sin(lat2))  
        lat2 = np.degrees(lat2)
        lon2 = np.degrees(lon2)  
        return lat2, lon2

    def get_file_name(self, lat, lon):
        #returns filename
        #only works for N and W (North America)
        ns = "N"
        ew = "W"
        hgt_file = "N" + str(abs(int(lat))) + "W" + str(abs(int(lon)) + 1)
        return(hgt_file)

    def get_terrain(self, lat, lon):
        #returns terrain in feet
        SAMPLES = 3601
        file = self.get_file_name(lat, lon)
        if file not in self.terrain:
            print(lat, " ", lon, " not found")
            return(-32768)
        lat = abs(lat)
        lon = abs(lon)
        elevations = self.terrain[file]
        lat_row = int(round((lat - int(lat)) * (SAMPLES - 1), 0))
        lon_row = int(round((lon - int(lon)) * (SAMPLES - 1), 0))
        return(elevations[SAMPLES - 1 - lat_row, SAMPLES - 1 - lon_row].astype(int) * 3.28084)

    def check_terrain(self, BS_Lat, BS_Long, BS_Height, Cust_Lat, Cust_Long, Cust_Height):
        BS_Elev = self.get_terrain(BS_Lat, BS_Long)
        BS_Ant = BS_Height
        Cust_Elev = self.get_terrain(Cust_Lat, Cust_Long)
        Cust_Ant = Cust_Height
        BS_H = (BS_Elev + BS_Ant) * 0.3048 #converting to meters
        Cust_H = (Cust_Elev + Cust_Ant) * 0.3048 #converting to meters
        if BS_H >= Cust_H:
            A_Lat = BS_Lat
            A_Long = BS_Long
            A_H = BS_H
            B_Lat = Cust_Lat
            B_Long = Cust_Long
            B_H = Cust_H
        else:
            B_Lat = BS_Lat
            B_Long = BS_Long
            B_H = BS_H
            A_Lat = Cust_Lat
            A_Long = Cust_Long
            A_H = Cust_H
        dist = self.distance_haversine([BS_Lat, BS_Long], [Cust_Lat, Cust_Long])
        dist_m = dist * 1609.34 #converting to meters
        steps = int(dist_m/160.934) #every 0.1 miles in meters
        step_size = dist_m/steps
        bear = self.bearing([B_Lat, B_Long], [A_Lat, A_Long])
        tower_angle = np.arctan((A_H - B_H)/dist_m)
        block = False
        for n1 in range(1, steps): 
            z = n1 * step_size
            y = B_H + (z * np.tan(tower_angle))
            step_lat, step_long = self.dist_coord(B_Lat, B_Long, bear, z)
            t = self.get_terrain(step_lat, step_long) #feet
            y = y*3.28084 #converting to feet from meter
            if t >= y:
                block = True
                break
        return(block)

    def check_terrain_with_combined_customer_height_and_terrain(self, BS_Lat, BS_Long, BS_Height, BS_Elev, Cust_Lat, Cust_Long, Cust_Height):
        #BS_Elev = self.get_terrain(BS_Lat, BS_Long)
        BS_Ant = BS_Height
        #Cust_Elev = self.get_terrain(Cust_Lat, Cust_Long)
        Cust_Ant = Cust_Height
        BS_H = (BS_Elev + BS_Ant) * 0.3048 #converting to meters
        Cust_H = (Cust_Ant) * 0.3048 #converting to meters
        if BS_H >= Cust_H:
            A_Lat = BS_Lat
            A_Long = BS_Long
            A_H = BS_H
            B_Lat = Cust_Lat
            B_Long = Cust_Long
            B_H = Cust_H
        else:
            B_Lat = BS_Lat
            B_Long = BS_Long
            B_H = BS_H
            A_Lat = Cust_Lat
            A_Long = Cust_Long
            A_H = Cust_H
        dist = self.distance_haversine([BS_Lat, BS_Long], [Cust_Lat, Cust_Long])
        dist_m = dist * 1609.34 #converting to meters
        steps = int(dist_m/160.934) #every 0.1 miles in meters
        #print(steps)
        step_size = dist_m/steps
        bear = self.bearing([B_Lat, B_Long], [A_Lat, A_Long])
        tower_angle = np.arctan((A_H - B_H)/dist_m)
        block = False
        for n1 in range(1, steps): 
            z = n1 * step_size
            y = B_H + (z * np.tan(tower_angle))
            step_lat, step_long = self.dist_coord(B_Lat, B_Long, bear, z)
            t = self.get_terrain(step_lat, step_long) #feet
            y = y*3.28084 #converting to feet from meter
            #print(step_lat, step_long, t, y)
            if t >= y:
                block = True
                break
        return(block)

    def kml_coord(self, BS_Lat, BS_Long, BS_Elev, BS_Ant, Cust_Lat, Cust_Long, Cust_Elev, Cust_Ant):
        BS_H = (BS_Elev + BS_Ant) * 0.3048
        Cust_H = (Cust_Elev + Cust_Ant) * 0.3048
        if BS_H >= Cust_H:
            A_Lat = BS_Lat
            A_Long = BS_Long
            A_H = BS_H
            B_Lat = Cust_Lat
            B_Long = Cust_Long
            B_H = Cust_H
        else:
            B_Lat = BS_Lat
            B_Long = BS_Long
            B_H = BS_H
            A_Lat = Cust_Lat
            A_Long = Cust_Long
            A_H = Cust_H
        dist = self.distance_haversine([BS_Lat, BS_Long], [Cust_Lat, Cust_Long])
        dist_m = dist * 1609.34
        steps = int(dist_m/321.869)
        step_size = dist_m/steps
        bear = self.bearing([B_Lat, B_Long], [A_Lat, A_Long])
        tower_angle = np.arctan((A_H - B_H)/dist_m)
        coord = [(B_Long, B_Lat, B_H)]
        for n1 in range(1, steps):
            z = n1 * step_size
            y = B_H + (z * np.tan(tower_angle))
            step_lat, step_long = self.dist_coord(B_Lat, B_Long, bear, z)
            coord.append((step_long, step_lat, y))
        coord.append((A_Long, A_Lat, A_H))
        return(coord)

    def kml_coord_combined_customer_height_and_elevation(self, BS_Lat, BS_Long, BS_Elev, BS_Ant, Cust_Lat, Cust_Long, Cust_Ant):
        BS_H = (BS_Elev + BS_Ant) * 0.3048
        Cust_H = (Cust_Ant) * 0.3048
        if BS_H >= Cust_H:
            A_Lat = BS_Lat
            A_Long = BS_Long
            A_H = BS_H
            B_Lat = Cust_Lat
            B_Long = Cust_Long
            B_H = Cust_H
        else:
            B_Lat = BS_Lat
            B_Long = BS_Long
            B_H = BS_H
            A_Lat = Cust_Lat
            A_Long = Cust_Long
            A_H = Cust_H
        dist = self.distance_haversine([BS_Lat, BS_Long], [Cust_Lat, Cust_Long])
        dist_m = dist * 1609.34
        steps = int(dist_m/321.869)
        step_size = dist_m/steps
        bear = self.bearing([B_Lat, B_Long], [A_Lat, A_Long])
        tower_angle = np.arctan((A_H - B_H)/dist_m)
        coord = [(B_Long, B_Lat, B_H)]
        for n1 in range(1, steps):
            z = n1 * step_size
            y = B_H + (z * np.tan(tower_angle))
            step_lat, step_long = self.dist_coord(B_Lat, B_Long, bear, z)
            coord.append((step_long, step_lat, y))
        coord.append((A_Long, A_Lat, A_H))
        return(coord)


    def kml_coord_all_combined_height(self, BS_Lat, BS_Long, BS_Ant, Cust_Lat, Cust_Long, Cust_Ant):
        BS_H = (BS_Ant) * 0.3048
        Cust_H = (Cust_Ant) * 0.3048
        if BS_H >= Cust_H:
            A_Lat = BS_Lat
            A_Long = BS_Long
            A_H = BS_H
            B_Lat = Cust_Lat
            B_Long = Cust_Long
            B_H = Cust_H
        else:
            B_Lat = BS_Lat
            B_Long = BS_Long
            B_H = BS_H
            A_Lat = Cust_Lat
            A_Long = Cust_Long
            A_H = Cust_H
        dist = self.distance_haversine([BS_Lat, BS_Long], [Cust_Lat, Cust_Long])
        dist_m = dist * 1609.34
        steps = int(dist_m/321.869)
        step_size = dist_m/steps
        bear = self.bearing([B_Lat, B_Long], [A_Lat, A_Long])
        tower_angle = np.arctan((A_H - B_H)/dist_m)
        coord = [(B_Long, B_Lat, B_H)]
        for n1 in range(1, steps):
            z = n1 * step_size
            y = B_H + (z * np.tan(tower_angle))
            step_lat, step_long = self.dist_coord(B_Lat, B_Long, bear, z)
            coord.append((step_long, step_lat, y))
        coord.append((A_Long, A_Lat, A_H))
        return(coord)

    def path_profile(self, BS_Lat, BS_Long, BS_Ant, Cust_Lat, Cust_Long, Cust_Ant):
        BS_Elev = self.get_terrain(BS_Lat, BS_Long) #feet
        Cust_Elev = self.get_terrain(Cust_Lat, Cust_Long) #feet
        BS_H = (BS_Elev + BS_Ant) #feet
        Cust_H = (Cust_Ant + Cust_Elev) #feet

        dist = self.distance_haversine([BS_Lat, BS_Long], [Cust_Lat, Cust_Long])

        distance_granularity = 100 #miles

        bear = self.bearing([BS_Lat, BS_Long], [Cust_Lat, Cust_Long])

        distance_points = [0]
        elevation_points = [BS_Elev]

        for n1 in range(1, distance_granularity):
             
            z = n1 * (dist/distance_granularity)

            step_lat, step_long = self.dist_coord(BS_Lat, BS_Long, bear, z*1609.34)

            t = self.get_terrain(step_lat, step_long) #feet
            
            distance_points.append(z) #Distance in miles
            elevation_points.append(t) #Elevation in feet

        distance_points.append(dist)
        elevation_points.append(Cust_Elev)

        df = pd.DataFrame(list(zip(distance_points, elevation_points)), columns =['Distance', 'Elevation'])

        fig = px.area(df, x="Distance", y="Elevation", color_discrete_sequence =['#49be25'])
        fig.add_shape(type="line", x0=distance_points[0], y0=BS_H, x1=distance_points[-1], y1=Cust_H, line=dict(color='Red', dash='dash'))
        fig.update_layout(title='Path Profile', xaxis_title='Distance (mi)', yaxis_title='Elevation (ft.)')

        return(fig)
        

    def coverage(self, proposed_location, a_height, b_height, max_distance, distance_granularity, azimuth_granularity):

        #proposed_location = (33.76786792166864, -118.19983418598554)

        #max_distance = 3200 # meters
        #distance_granularity = 100
        #azimuth_granularity = 360
        #a_height = 100 #in feet
        #b_height = 10 #in feet
        #a_terrain = self.get_terrain(proposed_location[0], proposed_location[1])

        test_locations_terrain_check = np.empty((azimuth_granularity, distance_granularity))
        test_locations_azmiuth = np.empty((azimuth_granularity, distance_granularity))
        test_locations_distance = np.empty((azimuth_granularity, distance_granularity))
        test_locations_terrain_check[:] = np.nan
        test_locations_azmiuth[:] = np.nan
        test_locations_distance[:] = np.nan
        coverage_coordinates = []

        for azimuth in tqdm(range(azimuth_granularity)):
            for distance in range(distance_granularity):
                current_azimuth = azimuth * 360 / azimuth_granularity
                current_distance = distance * max_distance / distance_granularity

                test_locations_azmiuth[azimuth, distance] = current_azimuth
                test_locations_distance[azimuth, distance] = current_distance

                current_coordinates = self.dist_coord(proposed_location[0], proposed_location[1], current_azimuth, current_distance)

                terrain_check = self.check_terrain(proposed_location[0], proposed_location[1], a_height, current_coordinates[0], current_coordinates[1], b_height)

                if terrain_check == True:
                    #Terrain is blocked
                    test_locations_terrain_check[azimuth, distance:] = 1
                else:
                    test_locations_terrain_check[azimuth, distance] = 0
                    coverage_coordinates.append(current_coordinates)

        x = pd.DataFrame()

        x["Azimith"] = test_locations_azmiuth.flatten()
        x["Distance (mi)"] = test_locations_distance.flatten() * 0.000621371
        x["Terrain"] = test_locations_terrain_check.flatten()  

        x["Terrain"].replace({1:np.nan}, inplace=True)

        x.dropna(inplace=True)

        fig = px.scatter_polar(x, r="Distance (mi)",
                                theta="Azimith",
                                color = "Distance (mi)",
                                color_continuous_scale=px.colors.sequential.Turbo,
                                #color_discrete_sequence=px.colors.sequential.Turbo,
                                width=1000, height=1000,
                                range_r=[0, max_distance*0.000621371], range_theta=[0, 360])
                                #template="plotly_dark",)

        fig.update_layout(
            title = 'LOS Coverage',
            showlegend = False
        )

        return(fig)

    
    def terrain_3d(self, a_lat, a_lon, a_height, b_lat, b_lon, b_height):

        a_elev = self.get_terrain(a_lat, a_lon)
        b_elev = self.get_terrain(b_lat, b_lon)

        dist = self.distance_haversine([a_lat, a_lon], [b_lat, b_lon])
        bear = self.bearing([a_lat, a_lon], [b_lat, b_lon])

        if bear > 180:
            bear_rev = bear - 180
        else:
            bear_rev = bear + 180

        new_dist = 1.2 * dist

        b_lat_new, b_lon_new = self.dist_coord(a_lat, a_lon, bear, new_dist*1609.34)
        a_lat_new, a_lon_new = self.dist_coord(b_lat, b_lon, bear_rev, new_dist*1609.34)

        l1 = min(a_lat, b_lat)
        l2 = max(a_lat, b_lat)
        g1 = min(a_lon, b_lon)
        g2 = max(a_lon, b_lon)

        NW = [l2, g1]
        NE = [l2, g2]
        SW = [l1, g1]
        SE = [l1, g2]

        ns_dist_granularity = 100
        ew_dist_granularity = 100

        ns_dist = self.distance_haversine([l1, g1], [l2, g1])
        ew_dist = self.distance_haversine([l1, g1], [l1, g2])

        test_locations_terrain_check = np.empty((ns_dist_granularity, ew_dist_granularity))
        test_locations_lat = np.empty((ns_dist_granularity, ew_dist_granularity))
        test_locations_long = np.empty((ns_dist_granularity, ew_dist_granularity))
        test_locations_terrain_check[:] = np.nan
        test_locations_lat[:] = np.nan
        test_locations_long[:] = np.nan

        a_dist = np.Inf
        b_dist = np.Inf

        a_loc_index = [0, 0]
        b_loc_index = [0, 0]

        for ns_dist_index in range(ns_dist_granularity):

            left_most_lat, left_most_long = self.dist_coord(NW[0], NW[1], 180, ns_dist_index * ns_dist / ns_dist_granularity * 1609.34)

            for ew_dist_index in range(ew_dist_granularity):

                

                test_lat, test_long = self.dist_coord(left_most_lat, left_most_long, 90, ew_dist_index * ew_dist / ew_dist_granularity * 1609.34)

                test_locations_terrain_check[ns_dist_index, ew_dist_index] = self.get_terrain(test_lat, test_long)
                test_locations_lat[ns_dist_index, ew_dist_index] = test_lat
                test_locations_long[ns_dist_index, ew_dist_index] = test_long

                a_current_dist = self.distance_haversine([a_lat, a_lon], [test_lat, test_long]) * 1609.34

                if a_current_dist < a_dist:
                    a_dist = a_current_dist
                    a_loc_index = [ns_dist_index, ew_dist_index]

                b_current_dist = self.distance_haversine([b_lat, b_lon], [test_lat, test_long]) * 1609.34

                if b_current_dist < b_dist:
                    b_dist = b_current_dist
                    b_loc_index = [ns_dist_index, ew_dist_index]

        z = test_locations_terrain_check
        sh_0, sh_1 = z.shape
        x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)

        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])

        y = [a_loc_index[0]/sh_0, b_loc_index[0]/sh_0]
        x = [a_loc_index[1]/sh_1, b_loc_index[1]/sh_1]
        z = [a_height + a_elev, b_height + b_elev]


        fig.add_scatter3d(x=x, y=y, z=z, mode='markers+lines', line=dict(color='red', width=5), surfaceaxis=1, surfacecolor="red")

        fig.update_layout(title='Elevation Profile', height=1000,)
                        #margin=dict(l=65, r=50, b=65, t=90))

        #update axes labels
        fig.update_layout(scene = dict(zaxis = dict(title='Elevation (ft.)', ticksuffix=' ft.')))

        return(fig)

    def all_los_paths(self, df_sites, max_distance, number_core_sites, random_seed):

        np.random.seed(random_seed)

        nodes = df_sites['Site ID'].tolist()

        sites_shortest_path = {}
        sites_shortest_path_length = {}

        core_site_indices = np.random.choice(df_sites.index.max()+1, number_core_sites, replace=False).tolist()

        core_site_names = df_sites.loc[core_site_indices, 'Site ID'].tolist()

        fig = go.Figure(go.Scattermapbox(
            mode = "markers",
            lon = [0],
            lat = [0],
            marker = {'size': 10, 'color': 'red'},
            name = "Core Site",
            legendgroup="Core Site"))


        fig.add_trace(go.Scattermapbox(
            mode = "markers",
            lon = [0],
            lat = [0],
            marker = {'size': 10, 'color': 'blue'},
            name = "Site",
            legendgroup = "Site",))

        fig.add_trace(go.Scattermapbox(
            mode = "lines",
            lon = [0, 0],
            lat = [0, 0],
            marker = {'size': 10, 'color': 'black'},
            name = "Path",
            legendgroup = "Path",))

        for index in range(df_sites.index.min(), df_sites.index.max()):
            for index2 in range(index+1, df_sites.index.max()+1):

                current_distance = self.distance_haversine([df_sites.loc[index, "Latitude"], df_sites.loc[index, "Longitude"]], [df_sites.loc[index2, "Latitude"], df_sites.loc[index2, "Longitude"]])

                if current_distance <= max_distance:
                    if self.check_terrain(df_sites.loc[index, "Latitude"], df_sites.loc[index, "Longitude"], df_sites.loc[index, "Height (ft)"], df_sites.loc[index2, "Latitude"], df_sites.loc[index2, "Longitude"], df_sites.loc[index2, "Height (ft)"]) == False:
                        
                        a_lat = df_sites.loc[index, "Latitude"]
                        b_lat = df_sites.loc[index2, "Latitude"]
                        a_lon = df_sites.loc[index, "Longitude"]
                        b_lon = df_sites.loc[index2, "Longitude"]

                        fig.add_trace(go.Scattermapbox(
                                mode = "lines",
                                lon = [a_lon, b_lon],
                                lat = [a_lat, b_lat],
                                marker = {'size': 10, 'color': "green"},
                                legendgroup="Path",
                                showlegend=False,))

        fig.add_trace(go.Scattermapbox(
            mode = "markers",
            lon = df_sites[df_sites['Site ID'].isin(core_site_names)]['Longitude'],
            lat = df_sites[df_sites['Site ID'].isin(core_site_names)]['Latitude'],
            marker = {'size': 10, 'color': 'red'},
            #text = df_sites[df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            #name = df_sites[df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            #hoverinfo = 'name',
            hovertext = df_sites[df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            legendgroup="Core Site",
            showlegend=False,))

        fig.add_trace(go.Scattermapbox(
            mode = "markers",
            lon = df_sites[~df_sites['Site ID'].isin(core_site_names)]['Longitude'],
            lat = df_sites[~df_sites['Site ID'].isin(core_site_names)]['Latitude'],
            marker = {'size': 10, 'color': 'blue'},
            #text = df_sites[df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            #name = df_sites[df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            #hoverinfo = 'name',
            hovertext = df_sites[df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            legendgroup="Site",
            showlegend=False,))

        fig.update_layout(
            margin ={'l':0,'t':0,'b':0,'r':0},
            mapbox = {
                'center': {'lon': -118.267311, 'lat': 33.866938},
                'style': "carto-positron",
                'zoom': 11.6,},
            #autosize=True,
            width=900,
            height=900,
            legend=dict(x = 0, y = 1))

        return(fig)

    def shortest_paths(self, df_sites, max_distance, number_core_sites, random_seed):

        np.random.seed(random_seed)

        nodes = df_sites['Site ID'].tolist()

        sites_shortest_path = {}
        sites_shortest_path_length = {}

        core_site_indices = np.random.choice(df_sites.index.max()+1, number_core_sites, replace=False).tolist()

        core_site_names = df_sites.loc[core_site_indices, 'Site ID'].tolist()

        colors = ['FFFFB300','FF803E75','FFFF6800','FFA6BDD7','FFC10020','FFCEA262','FF817066','FF007D34','FFF6768E','FF00538A','FFFF7A5C',
        'FF53377A','FFFF8E00','FFB32851','FFF4C800','FF7F180D','FF93AA00','FF593315','FFF13A13','FF232C16']

        colors = ["#" + x[2:] for x in colors]

        for core_site_index in core_site_indices:
            init_graph = {}

            for node in nodes:
                init_graph[node] = {}

            for index in range(df_sites.index.min(), df_sites.index.max()):
                for index2 in range(index+1, df_sites.index.max()+1):

                    current_distance = self.distance_haversine([df_sites.loc[index, "Latitude"], df_sites.loc[index, "Longitude"]], [df_sites.loc[index2, "Latitude"], df_sites.loc[index2, "Longitude"]])

                    if current_distance <= max_distance:
                        if self.check_terrain(df_sites.loc[index, "Latitude"], df_sites.loc[index, "Longitude"], df_sites.loc[index, "Height (ft)"], df_sites.loc[index2, "Latitude"], df_sites.loc[index2, "Longitude"], df_sites.loc[index2, "Height (ft)"]) == False:
                            
                            init_graph[df_sites.loc[index, "Site ID"]][df_sites.loc[index2, "Site ID"]] = max_distance + current_distance

            graph = Graph(nodes, init_graph)

            previous_nodes, shortest_path = dijkstra_algorithm(graph=graph, start_node=df_sites.loc[core_site_index, "Site ID"])

            for index in range(df_sites.index.min(), df_sites.index.max()+1):
                path = get_path(previous_nodes, shortest_path, start_node=df_sites.loc[core_site_index, "Site ID"], target_node=df_sites.loc[index, "Site ID"])

                if len(path) > 0:
                    if df_sites.loc[index, "Site ID"] not in sites_shortest_path_length:
                        sites_shortest_path_length[df_sites.loc[index, "Site ID"]] = shortest_path[df_sites.loc[index, "Site ID"]]
                        sites_shortest_path[df_sites.loc[index, "Site ID"]] = path
                    else:
                        if shortest_path[df_sites.loc[index, "Site ID"]] < sites_shortest_path_length[df_sites.loc[index, "Site ID"]]:
                            sites_shortest_path_length[df_sites.loc[index, "Site ID"]] = shortest_path[df_sites.loc[index, "Site ID"]]
                            sites_shortest_path[df_sites.loc[index, "Site ID"]] = path

        fig = go.Figure(go.Scattermapbox(
            mode = "markers",
            lon = [0],
            lat = [0],
            marker = {'size': 10, 'color': 'red'},
            name = "Core Site",
            legendgroup="Core Site"))


        fig.add_trace(go.Scattermapbox(
            mode = "markers",
            lon = [0],
            lat = [0],
            marker = {'size': 10, 'color': 'blue'},
            name = "Site",
            legendgroup = "Site",))

        fig.add_trace(go.Scattermapbox(
            mode = "lines",
            lon = [0, 0],
            lat = [0, 0],
            marker = {'size': 10, 'color': 'black'},
            name = "Path",
            legendgroup = "Path",))

        existing_paths = []

        for path_list_key in sites_shortest_path:
            path_list = sites_shortest_path[path_list_key]

            if len(path_list) > 1:

                color_index = (core_site_names.index(path_list[0]))%(len(colors))

                for index in range(len(path_list)-1):

                    index2 = index + 1
                    
                    if (path_list[index], path_list[index2]) not in existing_paths and (path_list[index2], path_list[index]) not in existing_paths:

                        name = path_list[index] + " - " + path_list[index2]

                        a_lat = df_sites.loc[df_sites["Site ID"] == path_list[index], "Latitude"].values[0]
                        b_lat = df_sites.loc[df_sites["Site ID"] == path_list[index2], "Latitude"].values[0]
                        a_lon = df_sites.loc[df_sites["Site ID"] == path_list[index], "Longitude"].values[0]
                        b_lon = df_sites.loc[df_sites["Site ID"] == path_list[index2], "Longitude"].values[0]

                        fig.add_trace(go.Scattermapbox(
                                mode = "lines",
                                lon = [a_lon, b_lon],
                                lat = [a_lat, b_lat],
                                marker = {'size': 10, 'color': colors[color_index]},
                                legendgroup="Path",
                                showlegend=False,))

                        existing_paths.append((path_list[index], path_list[index2]))

        fig.add_trace(go.Scattermapbox(
            mode = "markers",
            lon = df_sites[df_sites['Site ID'].isin(core_site_names)]['Longitude'],
            lat = df_sites[df_sites['Site ID'].isin(core_site_names)]['Latitude'],
            marker = {'size': 10, 'color': 'red'},
            #text = df_sites[df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            #name = df_sites[df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            #hoverinfo = 'name',
            hovertext = df_sites[df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            legendgroup="Core Site",
            showlegend=False,))

        fig.add_trace(go.Scattermapbox(
            mode = "markers",
            lon = df_sites[~df_sites['Site ID'].isin(core_site_names)]['Longitude'],
            lat = df_sites[~df_sites['Site ID'].isin(core_site_names)]['Latitude'],
            marker = {'size': 7, 'color': 'blue'},
            #text  = df_sites[~df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            #name = df_sites[~df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            #hoverinfo = 'name',
            hovertext = df_sites[~df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            legendgroup="Site",
            showlegend=False,))

        fig.update_layout(
            margin ={'l':0,'t':0,'b':0,'r':0},
            mapbox = {
                'center': {'lon': -118.267311, 'lat': 33.866938},
                'style': "carto-positron",
                'zoom': 11.6,},
            #autosize=True,
            width=900,
            height=900,
            legend=dict(x = 0, y = 1))

        return(fig)

    def backhaul_capacities(self, df_sites, max_distance, number_core_sites, random_seed):

        np.random.seed(random_seed)

        nodes = df_sites['Site ID'].tolist()

        sites_shortest_path = {}
        sites_shortest_path_length = {}

        core_site_indices = np.random.choice(df_sites.index.max()+1, number_core_sites, replace=False).tolist()

        core_site_names = df_sites.loc[core_site_indices, 'Site ID'].tolist()

        colors = ['FFFFB300','FF803E75','FFFF6800','FFA6BDD7','FFC10020','FFCEA262','FF817066','FF007D34','FFF6768E','FF00538A','FFFF7A5C',
        'FF53377A','FFFF8E00','FFB32851','FFF4C800','FF7F180D','FF93AA00','FF593315','FFF13A13','FF232C16']

        colors = ["#" + x[2:] for x in colors]

        for core_site_index in core_site_indices:
            init_graph = {}

            for node in nodes:
                init_graph[node] = {}

            for index in range(df_sites.index.min(), df_sites.index.max()):
                for index2 in range(index+1, df_sites.index.max()+1):

                    current_distance = self.distance_haversine([df_sites.loc[index, "Latitude"], df_sites.loc[index, "Longitude"]], [df_sites.loc[index2, "Latitude"], df_sites.loc[index2, "Longitude"]])

                    if current_distance <= max_distance:
                        if self.check_terrain(df_sites.loc[index, "Latitude"], df_sites.loc[index, "Longitude"], df_sites.loc[index, "Height (ft)"], df_sites.loc[index2, "Latitude"], df_sites.loc[index2, "Longitude"], df_sites.loc[index2, "Height (ft)"]) == False:
                            
                            init_graph[df_sites.loc[index, "Site ID"]][df_sites.loc[index2, "Site ID"]] = max_distance + current_distance

            graph = Graph(nodes, init_graph)

            previous_nodes, shortest_path = dijkstra_algorithm(graph=graph, start_node=df_sites.loc[core_site_index, "Site ID"])

            for index in range(df_sites.index.min(), df_sites.index.max()+1):
                path = get_path(previous_nodes, shortest_path, start_node=df_sites.loc[core_site_index, "Site ID"], target_node=df_sites.loc[index, "Site ID"])

                if len(path) > 0:
                    if df_sites.loc[index, "Site ID"] not in sites_shortest_path_length:
                        sites_shortest_path_length[df_sites.loc[index, "Site ID"]] = shortest_path[df_sites.loc[index, "Site ID"]]
                        sites_shortest_path[df_sites.loc[index, "Site ID"]] = path
                    else:
                        if shortest_path[df_sites.loc[index, "Site ID"]] < sites_shortest_path_length[df_sites.loc[index, "Site ID"]]:
                            sites_shortest_path_length[df_sites.loc[index, "Site ID"]] = shortest_path[df_sites.loc[index, "Site ID"]]
                            sites_shortest_path[df_sites.loc[index, "Site ID"]] = path


        path_capacities = {}

        for core_site_index in core_site_indices:

            current_core_site = df_sites.loc[core_site_index, "Site ID"]

            core_capacity = 0

            G = nx.DiGraph()

            existing_paths = []
            existing_nodes = []

            for path_list_key in sites_shortest_path:
                path_list = sites_shortest_path[path_list_key]

                if len(path_list) > 1:

                    if current_core_site == path_list[0]:

                        core_capacity += 100

                        for index in range(len(path_list)-1):

                            index2 = index + 1
                            
                            if path_list[index2] not in existing_nodes:
                                G.add_node(path_list[index2], demand = -100)
                                existing_nodes.append(path_list[index2])

            G.add_node(current_core_site, demand = core_capacity)


            for path_list_key in sites_shortest_path:
                path_list = sites_shortest_path[path_list_key]

                if len(path_list) > 1:

                    if current_core_site == path_list[0]:

                        for index in range(len(path_list)-1):

                            index2 = index + 1
                            
                            if (path_list[index2], path_list[index]) not in existing_paths:
                                
                                G.add_edge(path_list[index2], path_list[index], weight=1, capacity=math.inf)
                                
                                existing_paths.append((path_list[index2], path_list[index]))



            flowCost, flowDict = nx.network_simplex(G)


            for site_a in flowDict:
                for site_b in flowDict[site_a]:
                    path_capacities[site_b + " - " + site_a] = flowDict[site_a][site_b]

        df_sites["Core"] = "Site"
        df_sites.loc[core_site_indices, "Core"] = "Core Site"

        color_discrete_map = {"Core Site":"#f54242", "Site":"#4245f5"}

        fig = go.Figure(go.Scatter(
            mode = "markers",
            x = [0],
            y = [0],
            marker = {'size': 10, 'color': 'red'},
            name = "Core Site",
            legendgroup="Core Site"))


        fig.add_trace(go.Scatter(
            mode = "markers",
            x = [0],
            y = [0],
            marker = {'size': 10, 'color': 'blue'},
            name = "Site",
            legendgroup = "Site",))

        fig.add_trace(go.Scatter(
            mode = "lines",
            x = [0, 0],
            y = [0, 0],
            marker = {'size': 10, 'color': 'black'},
            name = "Path",
            legendgroup = "Path",))

        existing_paths = []

        for path_list_key in sites_shortest_path:
            path_list = sites_shortest_path[path_list_key]

            if len(path_list) > 1:

                color_index = (core_site_names.index(path_list[0]))%(len(colors))

                for index in range(len(path_list)-1):

                    index2 = index + 1
                    
                    if (path_list[index], path_list[index2]) not in existing_paths and (path_list[index2], path_list[index]) not in existing_paths:

                        name = path_list[index] + " - " + path_list[index2]

                        a_lat = df_sites.loc[df_sites["Site ID"] == path_list[index], "Latitude"].values[0]
                        b_lat = df_sites.loc[df_sites["Site ID"] == path_list[index2], "Latitude"].values[0]
                        a_lon = df_sites.loc[df_sites["Site ID"] == path_list[index], "Longitude"].values[0]
                        b_lon = df_sites.loc[df_sites["Site ID"] == path_list[index2], "Longitude"].values[0]

                        fig.add_trace(go.Scatter(
                                    mode = "lines",
                                    x = [a_lon, b_lon],
                                    y = [a_lat, b_lat],
                                    marker = {'color': colors[color_index]},
                                    #marker = {'size': 10, 'color': colors[color_index]},
                                    legendgroup="Path",
                                    showlegend=False,))

                        fig.add_trace(go.Scatter(
                            x=[(a_lon + b_lon)/2],
                            y=[(a_lat + b_lat)/2],
                            mode="text",
                            text=[str(path_capacities[name]) + "M"],
                            textposition="middle center",
                            legendgroup="Path",
                            showlegend=False,
                        ))

                        existing_paths.append((path_list[index], path_list[index2]))

        fig.add_trace(go.Scatter(
            mode = "markers",
            x = df_sites[df_sites['Site ID'].isin(core_site_names)]['Longitude'],
            y = df_sites[df_sites['Site ID'].isin(core_site_names)]['Latitude'],
            marker = {'size': 10, 'color': 'red'},
            #text = df_sites[df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            #name = df_sites[df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            #hoverinfo = 'name',
            hovertext = df_sites[df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            legendgroup="Core Site",
            showlegend=False,))

        fig.add_trace(go.Scatter(
            mode = "markers",
            x = df_sites[~df_sites['Site ID'].isin(core_site_names)]['Longitude'],
            y = df_sites[~df_sites['Site ID'].isin(core_site_names)]['Latitude'],
            marker = {'size': 10, 'color': 'blue'},
            #text = df_sites[df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            #name = df_sites[df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            #hoverinfo = 'name',
            hovertext = df_sites[df_sites['Site ID'].isin(core_site_names)]['Site ID'],
            legendgroup="Site",
            showlegend=False,))

        fig.update_layout(
            margin ={'l':0,'t':0,'b':0,'r':0},
            #autosize=True,
            width=900,
            height=900,
            legend=dict(x = 0, y = 1),
            yaxis_range=[33.789683,33.947535],
            xaxis_range=[-118.366900,-118.167000])

        return(fig)
        




    





            