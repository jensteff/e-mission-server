__author__ = 'Yin'
# Standard imports
import logging

# Our imports
from emission.analysis.result.carbon import getModeCarbonFootprint, carbonFootprintForMode
from emission.core.common import Inside_polygon,berkeley_area,getConfirmationModeQuery
from emission.core.get_database import get_section_db,get_profile_db
import geojson as gj
import emission.analysis.plotting.geojson.geojson_feature_converter as gfc
import emission.core.wrapper.motionactivity as ecwm
import emission.storage.decorations.timeline as esdt
import emission.storage.decorations.local_date_queries as esdl
import emission.core.wrapper.trip as ecwt
import emission.core.wrapper.section as ecws
import emission.storage.timeseries.geoquery as estg

# Note that all the points here are returned in (lng, lat) format, which is the
# GeoJSON format.

def carbon_by_zip(start,end):
    Profiles=get_profile_db()
    carbon_list=[]
    for zip in Profiles.distinct('zip'):
        # print(zip)
        if zip!='N/A':
            tempdict={}
            tempdict['weight']=Profiles.find({'zip':zip}).count()
            # print(Profiles.find({'zip':zip}).count())
            tempdict['carbon']=0
            for profile in Profiles.find({'zip':zip}):
                tempdict['loc']=profile['zip_centroid']
                user=profile['user_id']
                user_carbon=getModeCarbonFootprint(user,carbonFootprintForMode,start,end)
                tempdict['carbon']+=sum(list(user_carbon.values()))
            tempdict['carbon']=tempdict['carbon']/tempdict['weight']
            carbon_list.append(tempdict)
    return {"weightedLoc": carbon_list}

def Berkeley_pop_route(start_ts, end_ts):
    berkeley_json  = {"geometry": {
      "type": "Polygon",
      "coordinates": [[
        [-122.267443, 37.864693], [-122.267443, 37.880687], [-122.250985, 37.880687], [-10.0, 10.0]
        ]]
      }
    }
    # box = [ [-122.267443, 37.864693], [-122.250985, 37.880687] ]
    start_dt = esdl.get_local_date(start_ts, "UTC")
    end_dt = esdl.get_local_date(end_ts, "UTC")
    tl = esdt.get_aggregate_timeline_from_dt(start_dt, end_dt, berkeley_json)
    gj_list = gfc.get_geojson_for_timeline(None, tl)
    list_of_points=[]
    for gj in gj_list:
        for feature in gj:
            if feature['type'] == 'FeatureCollection':
                for feat in feature['features']:
                    if "properties" not in feat:
                        continue
                if feat['properties']['feature_type'] == "section":
                    points = feat.geometry.coordinates
                    list_of_points.extend(points)
    return {"latlng": list_of_points}

def range_mode_heatmap(mode, start_ts, end_ts):
    start_dt = esdl.get_local_date(start_ts, "UTC")
    end_dt = esdl.get_local_date(end_ts, "UTC")
    tl = esdt.get_aggregate_timeline_from_dt(start_dt, end_dt)
    gj_list = gfc.get_geojson_for_timeline(None, tl)
 
    logging.debug("len gl list is %d" % len(gj_list))
    list_of_point=[]
 
    for gj in gj_list:
        for feature in gj:
            if feature['type'] == 'FeatureCollection':
                for feat in feature['features']:
                    if "properties" not in feat:
                        continue
                    if feat['properties']['feature_type'] == "section":
                        if mode == 'all' or feat.properties["sensed_mode"] == mode:
                            points = feat.geometry.coordinates
                            list_of_point.extend(points)
    logging.debug("Returning list of size %s" % len(list_of_point))
    return {"latlng": list_of_point}



