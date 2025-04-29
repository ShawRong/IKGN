import pickle
import pandas as pd

file_names = ['nyc_category.pkl', 'tky_category.pkl']
file_name = file_names[0]

dataset = None
with open(file_name, 'rb') as file:
    dataset = pickle.load(file)

print(f"type: {type(dataset)}")
print("Keys:", dataset.keys())
print("Number of items:", len(dataset))
n = 10
for key in dataset.keys():
    print(f"type: {type(dataset[key])}")


dict_key = ['data_neural', 'vid_list', 'uid_list', 'parameters', 'data_filter', 'vid_lookup', 'KG']
list_key = ['category_name']
for key1 in dict_key:
    print(f"KEY:{key1}: ")
    for key2 in list(dataset[key1].keys())[:10]:
        print(f"\t key:{key2}: value type: {type(dataset[key1][key2])}") 

print(dataset[list_key[0]][:10])

print(dataset['data_neural'][0].keys())
for key in dataset['data_neural'][0].keys():
    print(f"key{key}, val type: {type(dataset['data_neural'][0][key])}")
#print("session data:", dataset['data_neural'][0]['sessions'])
#print("session data trans:", dataset['data_neural'][0]['sessions_trans'])
print("session data trans:", dataset['data_neural'][0]['train'])
print("session data trans:", dataset['data_neural'][0]['test'])
print("session data trans:", dataset['data_neural'][0]['vaild'])
#print('category name', dataset['category_name'])
#print('vid list', dataset['vid_list'])
#print('uid list', dataset['uid_list'])
print('parameters', dataset['parameters'])
#print('data filters', dataset['data_filter']['1'])
#print('vid lookup', dataset['vid_lookup'])
print('kg', dataset['KG'].keys())
#keys = ['utp', 'ptp', 'ptp_dict', 'poi_trans', 'timining_rel', 'tim_rel', 'dis_rel', 'train_kg', 'max_dis_tim']
#keys_todo = ['utp', 'ptp', 'ptp_dict', 'poi_trans', 'timining_rel', 'tim_rel', 'dis_rel', 'train_kg', 'train_kg_dict', 'max_dis_tim']
#for key in keys:
    #print(dataset['KG'][key])

""" Note:
dataset:
Keys: ['data_neural', 'vid_list', 'uid_list', 'category_name', 'parameters', 'data_filter', 'vid_lookup', 'KG']
is dict: ['data_neural', 'vid_list', 'uid_list', 'parameters', 'data_filter', 'vid_lookup', 'KG']
is list: ['category_name']


data_neural: data used to train the network.
    containing keys: [sessions, train, test, (vaild)(ATTENTION: CODER TYPO ERROR), sessions_trans]
    dict: [sessions, sessions_trans]
        sessions:       key: 0, 1, ... 136, val: [11, 2], [3, 4], ...
        sessions_trans: key: 0, 1, ... 136, val: [49, (0, 2), 46], ...
    list: [train, test, vaild]
        session data trans: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ...]
        session data trans: [123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136]
        session data trans: [109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]
        
vid_list: venue list. what the coder used is just len(vid_list)
    {'4b5f42f1f964a520a2b029e3': [13727, 3, '4bf58dd8d48988d147941735'], ...}
uid_list: user list. what the coder used is just len(uid_list)
    {'426': [1018, 10], '1081': [1019, 14], ...}
parameters: not used in code, at least I didn't see it. I suppose it's some parameter checkpoint of network.
    parameters {'TWITTER_PATH': 'D:\\STKGRec-main\\data\\dataset_TSMC2014_NYC.txt',
    'SAVE_PATH': 'D:\\STKGRec-main\\data\\', 'trace_len_min': 10,
    'location_global_visit_min': 10,
    'hour_gap': 24, 'min_gap': 10,
    'session_max': 10, 'filter_short_session': 3,
    'sessions_min': 5, 'train_split': 0.8}
data_filter: not used, too.
    {{'sessions_count': 7, 'topk_count': 81, 'topk': [('49d2b43ef964a520cb5b1fe3', 7), ('4d4ac10da0ef54814b6ffff6', 4), ('42accc80f964a52047251fe3', 4), ('4fbf92b7e4b08821682bf100', 4), ('42586c80f964a520db201fe3', 3),}...}
vid_lookup: used in main.py line 297 & line 307
    {..., 14084: [-73.79681911801289, 40.72192402665647]}
KG: 
    No idea what this is.

category_name: used only to get the number of category.
    ['4bf58dd8d48988d1bc941735', '4bf58dd8d48988d101941735', ...]

KEY:data_neural:
         key:0: value type: <class 'dict'>
         key:1: value type: <class 'dict'>
         etc ...

KEY:vid_list:
         key:unk: value type: <class 'list'>
         key:4ae8fd76f964a520e1b321e3: value type: <class 'list'>
         key:4b679336f964a520d9552be3: value type: <class 'list'>
         etc ...

KEY:uid_list:
         key:293: value type: <class 'list'>
         key:445: value type: <class 'list'>
         etc ...

KEY:parameters: omit

KEY:data_filter:
         key:768: value type: <class 'dict'>
         key:445: value type: <class 'dict'>
         etc ...

KEY:vid_lookup:
         key:9: value type: <class 'list'>
         key:10: value type: <class 'list'>
         etc ...

KEY:KG:
         key:utp: value type: <class 'dict'>
         key:ptp: value type: <class 'dict'>
         key:ptp_dict: value type: <class 'dict'>
         key:poi_trans: value type: <class 'dict'>
         key:timining_rel: value type: <class 'list'>
         key:tim_rel: value type: <class 'list'>
         key:dis_rel: value type: <class 'list'>
         key:train_kg: value type: <class 'numpy.ndarray'>
         key:train_kg_dict: value type: <class 'collections.defaultdict'>
         key:max_dis_tim: value type: <class 'list'>
         etc ...


'category_name' (list): omit
"""