import json
import difflib
import struct
from lgp.env.alfred.segmentation_definitions import OBJECT_CLASSES, object_string_to_intid

number = 0
mode = 'full_try19'

#### making file name list json

data_list = []
same = False
repeat_idx = 0
task = ''
f = open(f"/home/snubi/Downloads/file_name_decompose_{mode}.txt", 'r')
while True:
    line = f.readline()
    if not line: 
        break
    data_dict = {}
    last_task = task
    task = line.replace('/data/syshin/alfred_data_all/json_2.1.0/valid_seen/', '').replace('/traj_data.json\n', '')
    data_dict['task'] = task
    if last_task == task:
        repeat_idx += 1
    else:
        repeat_idx = 0
    data_dict['repeat_idx'] = repeat_idx
    data_list.append(data_dict)
f.close()

final_data_dict = {}
final_data_dict['valid_seen'] = data_list

with open(f'/home/snubi/ECCV2024/hlsm/alfred_src/alfred/data/splits/decompose_{mode}.json', 'w', encoding='utf-8') as file:
    json.dump(final_data_dict, file)




# ### making policy json 


entire_policy_list = []
policy_list = []
count = 0
f = open(f"/home/snubi/Downloads/{number}.policy_decompose_{mode}.txt", 'r')
while True:
    line = f.readline()
    # if len(line) < 10 and len(line) != 1:
    #     print(len(line), line)

    policy_dict = {}
    if '(' in line and ')' in line:
        is_two = False
        idx_1 = line.find('(')
        idx_2 = line.find(')')
        line = line[idx_1 + 1: idx_2]
        split_arguments = line.split(',')
        if len(split_arguments) == 2:
            policy_dict['action'] = split_arguments[0].strip()
            policy_dict['object'] = split_arguments[1].strip()
        elif len(split_arguments) == 1:
            policy_dict['action'] = split_arguments[0].strip()
            policy_dict['object'] = ''
        else:
            policy_dict['action'] = split_arguments[0].strip()
            if 'put' in split_arguments[0].strip().lower():
                policy_dict['object'] = split_arguments[2].strip()
            else:
                policy_dict['object'] = split_arguments[1].strip()
        policy_list.append(policy_dict)
    elif '(' in line or ')' in line:
        continue
    elif len(line) == 1:
        continue
    else:
        if len(policy_list) != 0:
            entire_policy_list.append(policy_list)
        if len(entire_policy_list) != 0 and len(policy_list) == 0:
            print(line)
        policy_list = []
            
    if not line:
        break
f.close()

data_dict = {}
for i, pl in enumerate(entire_policy_list):
    data_dict[i] = pl

with open(f'/home/snubi/ECCV2024/hlsm/data/policy_data/decompose_{mode}.json', 'w', encoding='utf-8') as file:
    json.dump(data_dict, file)




### modify policy json


action_list = ['pick up', 'put', 'toggle on', 'toggle off', 'slice', 'open', 'close', 'stop']
alfred_action_list = ['PickupObject', 'PutObject', 'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject', 'OpenObject', 'CloseObject', 'Stop']

new_dict = {}

with open(f'/home/snubi/ECCV2024/hlsm/data/policy_data/decompose_{mode}.json', 'r') as file:
    data = json.load(file)
    for i in list(data.keys()):
        new_policy_list = []
        for d in data[i]:
            new_policy_dict = {}
            if d['action'] in action_list or d['action'] in alfred_action_list:
                if d['action'] in action_list:
                    new_policy_dict['action'] = alfred_action_list[action_list.index(d['action'])]
                else:
                    new_policy_dict['action'] = d['action']
                new_policy_dict['object'] = ''
                max_len = 0
                final_intersection = ''
                object_name = ''
                is_contained = False
                for o in OBJECT_CLASSES:
                    if o.lower() == d['object'].lower().replace(' ', ''):
                        if len(d['object']) != 0:
                            new_policy_dict['object'] = o
                            is_contained = True
                            break
                if not is_contained:            
                    for o in OBJECT_CLASSES:
                        if o.lower() in d['object'].lower().replace(' ', '') or d['object'].lower().replace(' ', '') in o.lower():
                            if len(d['object']) != 0:
                                new_policy_dict['object'] = o
                                is_contained = True
                                break
                if is_contained:
                    new_policy_list.append(new_policy_dict)
                if not is_contained:
                    # for o in OBJECT_CLASSES:
                    #     match = difflib.SequenceMatcher(a=o.lower(), b=d['object'].lower().replace(' ', '')).find_longest_match(
                    #     0,
                    #     len(o),
                    #     0,
                    #     len(d['object'].lower().replace(' ', '')),
                    #     )  
                    #     object_intersection = o[match.a:match.a+match.size]
                    #     len_intersection = len(object_intersection)
                    #     if len_intersection > max_len:
                    #         final_intersection = object_intersection
                    #         max_len = len_intersection
                    #         object_name = o
                    # if object_name.lower() in final_intersection.lower() or final_intersection.lower() in object_name.lower():
                    #     if len(d['object']) != 0:
                    #         new_policy_dict['object'] = object_name
                    new_policy_dict['object'] = d['object']
                    new_policy_list.append(new_policy_dict)
                
        
        new_dict[i] = new_policy_list
    print(len(new_dict))

with open(f'/home/snubi/ECCV2024/hlsm/data/policy_data/decompose_{mode}_modify.json', 'w', encoding='utf-8') as file:
    json.dump(new_dict, file)




### delete malformed data


# with open(f'/home/snubi/ECCV2024/hlsm/data/policy_data/decompose_{mode}_modify.json', 'r') as policy_file:
#     with open(f'/home/snubi/ECCV2024/hlsm/alfred_src/alfred/data/splits/decompose_{mode}.json', 'r') as file_name_file:
#         policy_data = json.load(policy_file)
#         file_name_data = json.load(file_name_file)

#         print(len(policy_data.keys()))
#         print(len(file_name_data['valid_seen']))

#         new_policy_data = {}
#         idx = 0

#         for i, k in enumerate(list(policy_data.keys())):
#             if len(policy_data[k]) == 0:
#                 del file_name_data['valid_seen'][i]
#             else:
#                 new_policy_data[idx] = policy_data[k]
#                 idx += 1

#         print(len(new_policy_data.keys()))
#         print(len(file_name_data['valid_seen']))

#         with open(f'/home/snubi/ECCV2024/hlsm/data/policy_data/decompose_{mode}_deleted.json', 'w', encoding='utf-8') as file:
#             json.dump(new_policy_data, file)
        
#         with open(f'/home/snubi/ECCV2024/hlsm/alfred_src/alfred/data/splits/decompose_{mode}_deleted.json', 'w', encoding='utf-8') as file:
#             json.dump(file_name_data, file)
        

# # print(object_string_to_intid("Lettuce"))





# tasks = []
# indexes = []
# f = open(f"/home/snubi/Downloads/policy_decompose_try10_all.txt", 'r')
# line = ''
# while True:
#     last_line = line

#     line = f.readline()

#     if not line:
#         break

#     if '(' in line or ')' in line or len(line) == 1:
#         continue

#     if len(last_line) <= 1:
#         tasks.append(line)

# new_tasks = []
# f2 = open(f"/home/snubi/Downloads/{number}.policy_decompose_{mode}.txt", 'r')
# while True:
#     line = f2.readline()

#     if not line:
#         break

#     policy_dict = {}
#     if '(' in line or ')' in line or len(line) == 1:
#         continue
#     new_tasks.append(line)

# for nt in new_tasks:
#     for i, t in enumerate(tasks):
#         if nt == t or ('Put a pan with spoon in it, in the sink.' in t and 'Put a pan with spoon in it, in the sink.' in nt):
#             indexes.append(i)

# new_policy = {}
# print(len(indexes))
# print(indexes)

# for i, idx in enumerate(indexes):
#     if idx == 408:
#         indexes[i] = 405
#     elif idx == 521:
#         indexes[i] = 518
#     elif idx == 540:
#         indexes[i] = 546
#     elif idx == 635:
#         indexes[i] = 632
#     elif idx == 679:
#         indexes[i] = 676

# j = 0
# with open(f'/home/snubi/ECCV2024/hlsm/data/policy_data/decompose_{mode}_modify.json', 'r') as policy_file:
#     policy_data = json.load(policy_file)
#     for nt in new_tasks:
#         new_policy[indexes[j]] = policy_data[str(j)]
#         j += 1
        
# print(indexes)
# with open(f'/home/snubi/ECCV2024/hlsm/data/policy_data/decompose_{mode}_sorted.json', 'w', encoding='utf-8') as file:
#     json.dump(new_policy, file)




