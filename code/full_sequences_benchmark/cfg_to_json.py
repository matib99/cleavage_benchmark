import json
import os
import sys

config = {}

# path to cfg from args
cfg_path = sys.argv[1]

# read cfg
with open(cfg_path, 'r') as f:
    for line in f:
        line = line.strip()
        line = line.replace('--', '')
        key, val = line.split(' ')
        config[key] = val

# save to json
json_path = sys.argv[2]
with open(json_path, 'w') as f:
    json.dump(config, f, indent=4)