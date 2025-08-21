import json
import string

def parse_name_percent(line):
    # Remove the leading and trailing "|"
    cleaned_line = line.strip('| ')
    # Split the line by spaces to separate the percentage
    name, percentage = cleaned_line.rsplit(' ', 1)
    # Convert percentage to a float and remove the "%" symbol
    percent_value = float(percentage.strip('%'))
    
    return name, percent_value

def parse_sets(file, elo):
    with open(file, 'r') as f:
        text = f.read()
    delimiter = ' +----------------------------------------+ \n +----------------------------------------+ '
    sets = text.split(delimiter)
    json_out = {}
    for _set in sets:
        set = _set.split('\n')
        # ignore first
        i = 1
        # parse pokemon name
        species = set[i].strip('| ').lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '')
        # get abilities
        while 'Abilities' not in set[i]:
            i+=1
        i+=1
        abilities = []
        while '+-' not in set[i]:
            if 'Other' not in set[i]:
                name, percentage = parse_name_percent(set[i])
                abilities.append({'name': name, 'percentage': percentage})
            i+=1
            
        # get items
        while 'Items' not in set[i]:
            i+=1
        i+=1
        items = []
        while '+-' not in set[i]:
            if 'Other' not in set[i]:
                name, percentage = parse_name_percent(set[i])
                items.append({'name': name, 'percentage': percentage})
            i+=1
        
        while 'Spreads' not in set[i]:
            i+=1
        i+=1
        # get spread
        spreads = []
        while '+-' not in set[i]:
            if 'Other' not in set[i]:
                # Split the line by spaces to separate the name and percentage
                nature_and_stats, percentage = parse_name_percent(set[i])

                # Split by ":" to separate the nature and stats
                nature, stats = nature_and_stats.split(':')

                # Convert stats into a list of integers
                stats_list = list(map(int, stats.split('/')))

                # Create the desired output
                spreads.append({'nature': nature, 'stats': stats_list, 'percentage': percentage})
            i+=1
            
        # get moves
        while 'Moves' not in set[i]:
            i+=1
        i+=1
        moves = []
        while '+-' not in set[i]:
            if 'Other' not in set[i]:
                name, percentage = parse_name_percent(set[i])
                moves.append({'name': name, 'percentage': percentage})
            i+=1
            
        # get tera_types
        if 'gen9' in file:
            while 'Tera Types' not in set[i]:
                i+=1
            i+=1
            tera_types = []
            while '+-' not in set[i]:
                if 'Other' not in set[i]:
                    name, percentage = parse_name_percent(set[i])
                    tera_types.append({'name': name, 'percentage': percentage})
                i+=1
            
            
        # get teammates
        while 'Teammates' not in set[i]:
            i+=1
        i+=1
        teammates = []
        while '+-' not in set[i]:
            if 'Other' not in set[i]:
                name, percentage = parse_name_percent(set[i])
                teammates.append({'name': name, 'percentage': percentage})
            i+=1
        
        json_out[species] = {
            'abilities': abilities,
            'items': items,
            'spreads': spreads,
            'moves': moves,
            'tera': tera_types,
            'teammates': teammates
        }
    with open(f'poke_env/data/static/gen9/vgc/sets_{elo}.json', 'w') as f:
        f.write(json.dumps(json_out, indent=4))
                    
        
if __name__ == '__main__':
    elo = 0
    file = f'poke_env/data/static/gen9/vgc/gen9vgc2025regi-{elo}.txt'
    parse_sets(file, elo)