import re

def get_premade_filter (str:str):
    config = {}
    #special case fog with ajustable aplha
    if "mod" in str:
        if not re.search("mod_fog\d+\.\d+",str) == None:
            modifier = re.search("\d+\.\d+",str)
            config = {'octaves':4, 'persistence':0.2, 'lacunarity': 3, 'alpha': 1-0.6*(float(modifier.group(0))), 'darkness':0.5}
            result =  ({'fog_set':config})

        if not re.search("mod_night\d+\.\d+", str) == None:
            modifier = re.search("\d+\.\d+",str)
            config = {'factor': 1-0.8*(float(modifier.group(0))-0.001)}
            homo = {'a':1,'b':0.5,'cutoff':800}
            result =  ({'day_set':config,'homo_set':homo})
            # result =  ({'day_set':config})

        if not re.search("mod_rain\d+\.\d+",str) == None:
            modifier = re.search("\d+\.\d+",str)
            config ={'density':(0.1*float(modifier.group(0)), 0.15*float(modifier.group(0))),'density_uniformity':(0.9,1.0),'drop_size':(0.5,0.65),'drop_size_uniformity':(0.1,0.5),'angle':(-20,20),'speed':(0.1,0.2),'blur':(0.001,0.001)}
            result =  ({'wh_set':config})

        if not re.search("mod_snow\d+\.\d+", str) == None:
            modifier = re.search("\d+\.\d+", str)
            config = {'density':(0.11*float(modifier.group(0)),0.16*float(modifier.group(0))),'density_uniformity':(0.95,1.0),'drop_size':(0.7,0.8),'drop_size_uniformity':(0.2,0.3),'angle':(-30,30),'speed':(0.04,0.1),'blur':(0.004,0.01),'mode':'snow'}
            result =  ({'wh_set':config})

    if str == 'fog':
        config = {'octaves':4, 'persistence':0.2, 'lacunarity': 3, 'alpha': 0.4, 'darkness':0.5}
        result =  ({'fog_set':config})
    if str == 'fog_mild':
        config = {'octaves':1, 'persistence':0.2, 'lacunarity': 3, 'alpha': 0.5, 'darkness':0.5}
        result =  ({'fog_set':config})
    if str == 'fog_medium':
        config = {'octaves':4, 'persistence':0.2, 'lacunarity': 3, 'alpha': 0.4, 'darkness':0.5}
        result =  ({'fog_set':config})
    if str == 'fog_heavy':
        config = {'octaves':4, 'persistence':0.2, 'lacunarity': 3, 'alpha': 0.1, 'darkness':0.5}
        result =  ({'fog_set':config})
    if str == 'rain':
        config =  {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001)}
        result =  ({'wh_set':config})
    if str == 'rain_mild':
        config = {'density':(0.01,0.02),'density_uniformity':(0.7,1.0),'drop_size':(0.25,0.3),'drop_size_uniformity':(0.1,0.5),'angle':(-20,20),'speed':(0.1,0.15),'blur':(0.004,0.01)}
        result =  ({'wh_set':config})
    if str == 'rain_medium':
        config ={'density':(0.06,0.08),'density_uniformity':(0.7,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-20,20),'speed':(0.1,0.15),'blur':(0.001,0.001)}
        result =  ({'wh_set':config})
    if str == 'rain_heavy':
        config ={'density':(0.1,0.15),'density_uniformity':(0.9,1.0),'drop_size':(0.5,0.65),'drop_size_uniformity':(0.1,0.5),'angle':(-20,20),'speed':(0.1,0.2),'blur':(0.001,0.001)}
        result =  ({'wh_set':config})
    if str == 'snow':
        config = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.7,0.8),'drop_size_uniformity':(0.2,0.3),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001),'mode':'snow'}
        result =  ({'wh_set':config})
    if str == 'snow_mild':
        config ={'density':(0.07,0.07),'density_uniformity':(0.95,1.0),'drop_size':(0.7,0.8),'drop_size_uniformity':(0.2,0.3),'angle':(-30,30),'speed':(0.04,0.1),'blur':(0.004,0.01),'mode':'snow'}
        result =  ({'wh_set':config})
    if str == 'snow_medium':
        config = {'density':(0.09,0.1),'density_uniformity':(0.95,1.0),'drop_size':(0.7,0.8),'drop_size_uniformity':(0.2,0.3),'angle':(-30,30),'speed':(0.04,0.1),'blur':(0.004,0.01),'mode':'snow'}
        result =  ({'wh_set':config})
    if str == 'snow_heavy':
        config = {'density':(0.11,0.16),'density_uniformity':(0.95,1.0),'drop_size':(0.7,0.8),'drop_size_uniformity':(0.2,0.3),'angle':(-30,30),'speed':(0.04,0.1),'blur':(0.004,0.01),'mode':'snow'}
        result =  ({'wh_set':config})
    if str == 'day':
        config = {'factor':1.0}
        result =  ({'day_set':config})
    if str == 'night':
        config = {'factor':0.3}
        result =  ({'day_set':config})
    if str == 'night_mild':
        config = {'factor':0.5}
        result =  ({'day_set':config})
    if str == 'night_medium':
        config = {'factor':0.3}
        result =  ({'day_set':config})
    if str == 'night_heavy':
        config = {'factor':0.15}
        result =  ({'day_set':config})
    if str == 'std_homo':
        config = {'a':1,'b':0.5,'cutoff':800}
        result =  ({'homo_set':config})
    if str == 'foghomo':
        config_h = {'a':1,'b':0.5,'cutoff':800}
        config_f = {'octaves':8, 'persistence':0.3, 'lacunarity': 5, 'alpha': 0.4}
        result =  ({'fog_set':config_f,'homo_set':config_h})
    if str == 'rainhomo':
        config_h = {'a':1,'b':0.5,'cutoff':800}
        config_r= {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001)}
        result =  ({'wh_set':config_r,'homo_set':config_h})
    if str == 'snowhomo':
        config_h = {'a':1,'b':0.5,'cutoff':800}
        config_s = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.7,0.8),'drop_size_uniformity':(0.2,0.3),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001),'mode':'snow'}
        result =  ({'wh_set':config_s,'homo_set':config_h})
    if str == 'dayhomo':
        config_h = {'a':1,'b':0.5,'cutoff':800}
        config_d = {'factor':1.0}
        result =  ({'day_set':config_d,'homo_set':config_h})
    if str == 'nighthomo':
        config_h = {'a':1,'b':0.5,'cutoff':800}
        config_n = {'factor':0.3}
        result =  ({'day_set':config_n,'homo_set':config_h})
    if str == 'fog_night':
        config_f = {'octaves':8, 'persistence':0.3, 'lacunarity': 5, 'alpha': 0.4}
        config_n = {'factor':0.3}
        result =  ({'day_set':config_n,'fog_set':config_f})
    if str == 'fog_day':
        config_f = {'octaves':8, 'persistence':0.3, 'lacunarity': 5, 'alpha': 0.4}
        config_n = {'factor':1.0}
        result =  ({'day_set':config_n,'fog_set':config_f})
    if str == 'fog_rain':
        config_f = {'octaves':8, 'persistence':0.3, 'lacunarity': 5, 'alpha': 0.4}
        config_w = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001),'mode':'snow'}
        result =  ({'wh_set':config_w,'fog_set':config_f})
    if str == 'fog_snow':
        config_f = {'octaves':8, 'persistence':0.3, 'lacunarity': 5, 'alpha': 0.4}
        config_w = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.7,0.8),'drop_size_uniformity':(0.2,0.3),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001),'mode':'snow'}
        result =  ({'wh_set':config_w,'fog_set':config_f})
    if str == 'rain_night':
        config_w = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001),'mode':'snow'}
        config_n = {'factor':0.3}
        result =  ({'wh_set':config_w,'day_set':config_n})
    if str == 'snow_night':
        config_w = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.7,0.8),'drop_size_uniformity':(0.2,0.3),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001),'mode':'snow'}
        config_n = {'factor':0.3}
        result =  ({'wh_set':config_w,'day_set':config_n})
    if str == 'de_fog15':
        config = {'kernel':15}
        result =  ({'defog_set':config})
    if str == 'de_fog10':
        config = {'kernel':10}
        result =  ({'defog_set':config}) 
    if str == 'de_fog5':
        config = {'kernel':5}
        result =  ({'defog_set':config})
    if str == 'fog_dehaze':
        config_f = {'octaves':4, 'persistence':0.2, 'lacunarity': 3, 'alpha': 0.5, 'darkness':0.5}
        config_d = {'kernel':15}
        result =  ({'fog_set':config_f,'defog_set':config_d})
    if str == 'rain_dehaze':
        config_r = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001)}
        config_d = {'kernel':15}
        result =  ({'wh_set':config_r,'defog_set':config_d})
    if str == 'snow_dehaze':
        config_s = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001)}
        config_d = {'kernel':15}
        result =  ({'wh_set':config_s,'defog_set':config_d})
    if str == 'night_dehaze':
        config_n = {'factor':0.5}
        config_d = {'kernel':15}
        result =  ({'day_set':config_n,'defog_set':config_d})
    if str == 'day_dehaze':
        config_da = {'factor':1.0}
        config_d = {'kernel':15}
        result =  ({'day_set':config_da,'defog_set':config_d})
    if str == 'dehaze_homo':
        config_d = {'kernel':15}
        config = {'a':1,'b':0.5,'cutoff':800}
        result =  ({'homo_set':config,'defog_set':config_d})
        
    if result == None:
        raise KeyError(f"(noise main) No such key exists: {str}")
    
    return result