import json
import os

from dl2019.utils.possibles import arg_list

def load_data(json_file_name):
    with open(os.path.abspath(json_file_name), 'r') as f:
        data = json.load(f)
    return data

class JobSpecGenerator(object):
    ''' Generates job specifications from a json file. '''
    def __init__(self, json_file_name):
        self.data = load_data(json_file_name)

    def __iter__(self):
        for item in self.data:
            for arg_item in arg_list:
                if arg_item not in item:
                    item[arg_item] = arg_list[arg_item]
            yield item
        raise StopIteration
