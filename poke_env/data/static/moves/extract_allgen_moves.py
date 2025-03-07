import json
import re
import string
import sys

move_name_list = []
move_effect_list = []
with open("allgen_raw.txt", "r") as f:
    data = f.readlines()

for line in data:
    line = line.split('\t')
    s = line[0].lower().replace(' ', '')
    s = s.translate(str.maketrans('', '', string.punctuation))
    move_name_list.append(s)
    move_effect_list.append(line[6])

move2effect = dict(zip(move_name_list, move_effect_list))


with open("moves_effect.json", "w+") as f:
    json.dump(move2effect, f, indent=4)
