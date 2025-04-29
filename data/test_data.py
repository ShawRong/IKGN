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

""" Note:
dataset:
Keys: ['data_neural', 'vid_list', 'uid_list', 'category_name', 'parameters', 'data_filter', 'vid_lookup', 'KG']
is dict: ['data_neural', 'vid_list', 'uid_list', 'parameters', 'data_filter', 'vid_lookup', 'KG']
is list: ['category_name']


data_neural: data used to train the network.
vid_list: venue list. what the coder used is just len(vid_list)
uid_list: user list. what the coder used is just len(uid_list)
parameters: not used in code, at least I didn't see it. I suppose it's some parameter checkpoint of network.
data_filter: not used, too.
vid_lookup: used in main.py line 297 & line 307
KG: I suppose it's some generated labels from knowledge graph, but the coder didn't publish their KG building process.
category_name: used only to get the number of category.

KEY:data_neural:
         key:0: value type: <class 'dict'>
         key:1: value type: <class 'dict'>
         key:2: value type: <class 'dict'>
         key:3: value type: <class 'dict'>
         key:4: value type: <class 'dict'>
         key:5: value type: <class 'dict'>
         key:6: value type: <class 'dict'>
         key:7: value type: <class 'dict'>
         key:8: value type: <class 'dict'>
         key:9: value type: <class 'dict'>
         etc ...

KEY:vid_list:
         key:unk: value type: <class 'list'>
         key:4ae8fd76f964a520e1b321e3: value type: <class 'list'>
         key:4d5bedbb5d153704613a6ce7: value type: <class 'list'>
         key:4c97edbdf419a09395806b88: value type: <class 'list'>
         key:4e6eab51d1647b1137a3a2d0: value type: <class 'list'>
         key:4d4b5ceb8e948cfa35fcef48: value type: <class 'list'>
         key:4e5577f445dd0a4826e6ab3e: value type: <class 'list'>
         key:4b1e78c2f964a520461a24e3: value type: <class 'list'>
         key:4db12e95a86e63d21166092b: value type: <class 'list'>
         key:4b679336f964a520d9552be3: value type: <class 'list'>
         etc ...

KEY:uid_list:
         key:293: value type: <class 'list'>
         key:185: value type: <class 'list'>
         key:354: value type: <class 'list'>
         key:315: value type: <class 'list'>
         key:84: value type: <class 'list'>
         key:349: value type: <class 'list'>
         key:384: value type: <class 'list'>
         key:974: value type: <class 'list'>
         key:768: value type: <class 'list'>
         key:445: value type: <class 'list'>
         etc ...

KEY:parameters:
         key:TWITTER_PATH: value type: <class 'str'>
         key:SAVE_PATH: value type: <class 'str'>
         key:trace_len_min: value type: <class 'int'>
         key:location_global_visit_min: value type: <class 'int'>
         key:hour_gap: value type: <class 'int'>
         key:min_gap: value type: <class 'int'>
         key:session_max: value type: <class 'int'>
         key:filter_short_session: value type: <class 'int'>
         key:sessions_min: value type: <class 'int'>
         key:train_split: value type: <class 'float'>
         etc ...

KEY:data_filter:
         key:293: value type: <class 'dict'>
         key:185: value type: <class 'dict'>
         key:354: value type: <class 'dict'>
         key:315: value type: <class 'dict'>
         key:84: value type: <class 'dict'>
         key:349: value type: <class 'dict'>
         key:384: value type: <class 'dict'>
         key:974: value type: <class 'dict'>
         key:768: value type: <class 'dict'>
         key:445: value type: <class 'dict'>
         etc ...

KEY:vid_lookup:
         key:1: value type: <class 'list'>
         key:2: value type: <class 'list'>
         key:3: value type: <class 'list'>
         key:4: value type: <class 'list'>
         key:5: value type: <class 'list'>
         key:6: value type: <class 'list'>
         key:7: value type: <class 'list'>
         key:8: value type: <class 'list'>
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


'category_name' (list): ['4bf58dd8d48988d1bc941735', '4bf58dd8d48988d101941735', '4bf58dd8d48988d124951735', '4bf58dd8d48988d1be941735', '4f4531504b9074f6e4fb0102', '4bf58dd8d48988d154941735', '4bf58dd8d48988d1f8931735', '4bf58dd8d48988d1f3941735', '4bf58dd8d48988d1ea941735', '4bf58dd8d48988d17f941735']
"""