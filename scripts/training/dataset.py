from datasets import load_dataset
import argparse
from datetime import datetime
import re

def parse_month_year(month_year_str):
    """Parse a string like 'March2025' or 'March-2025' into a datetime object."""
    # Define month name mapping
    month_map = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    # Extract month and year using regex
    match = re.match(r'([a-zA-Z]+)[- ]?(\d{4})', month_year_str)
    if match:
        month_name, year_str = match.groups()
        month_num = month_map.get(month_name.lower())
        if month_num:
            return datetime(int(year_str), month_num, 1)
    
    # Fallback for other formats
    raise ValueError(f"Could not parse month_year: {month_year_str}")

def parse_elo_range(elo_range):
    """Parse an Elo range string like '1200-1399' or '1800+' into min and max values."""
    if '+' in elo_range:
        lower_bound = int(elo_range[:-1])
        upper_bound = float('inf')  # No upper bound
    else:
        lower_bound, upper_bound = map(int, elo_range.split('-'))
    return lower_bound, upper_bound

def do_elo_ranges_overlap(range1, range2):
    """Check if two Elo ranges overlap."""
    min1, max1 = parse_elo_range(range1)
    min2, max2 = parse_elo_range(range2)
    
    # Check if ranges overlap
    return min1 <= max2 and min2 <= max1

def is_in_elo_range(elo_value, target_elo_ranges):
    """Check if an Elo value or range is within any of the target Elo ranges."""
    # If elo_value is already a range like "1200-1399"
    if isinstance(elo_value, str) and ('-' in elo_value or '+' in elo_value):
        # Check if this range overlaps with any of the target ranges
        for target_range in target_elo_ranges:
            if do_elo_ranges_overlap(elo_value, target_range):
                return True
        return False
    
    # If elo_value is a single number
    try:
        elo_int = int(elo_value)
        for target_range in target_elo_ranges:
            min_val, max_val = parse_elo_range(target_range)
            if min_val <= elo_int <= max_val:
                return True
        return False
    except (ValueError, TypeError):
        # If elo_value can't be converted to int and isn't a range
        return False

def load_filtered_dataset(min_month=None, max_month=None, elo_ranges=None, split="train", gamemode=None):
    """
    Load and filter the pokechamp dataset based on specified criteria.
    
    Args:
        min_month: Minimum month/year to include (e.g., 'March2025')
        max_month: Maximum month/year to include (e.g., 'March2025')
        elo_ranges: List of Elo ranges to include (e.g., ['1000-1199', '1800+'])
        split: Dataset split ('train' or 'test')
        gamemode: Specific gamemode to filter by (e.g., 'gen9ou')
        
    Returns:
        Filtered dataset
    """
    # Load the dataset
    dataset = load_dataset("milkkarten/pokechamp", split=split)
    
    # Filter by month_year if specified
    if min_month is not None or max_month is not None:
        min_date = parse_month_year(min_month) if min_month else None
        max_date = parse_month_year(max_month) if max_month else None
        
        def month_filter(example):
            try:
                example_date = parse_month_year(example["month_year"])
                if min_date and example_date < min_date:
                    return False
                if max_date and example_date > max_date:
                    return False
                return True
            except (KeyError, ValueError):
                return False
        
        dataset = dataset.filter(month_filter)
    
    # Filter by Elo if specified
    if elo_ranges is not None:
        dataset = dataset.filter(lambda example: is_in_elo_range(example["elo"], elo_ranges))
    
    # Filter by gamemode if specified
    if gamemode is not None:
        dataset = dataset.filter(lambda example: example["gamemode"] == gamemode)
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Load and filter the pokechamp dataset")
    parser.add_argument("--min_month", default="March2024", type=str, help="Minimum month_year to include (e.g., 'March2025')")
    parser.add_argument("--max_month", default="March2025", type=str, help="Maximum month_year to include (e.g., 'March2025')")
    parser.add_argument("--elo_ranges", default=["1000-1199", "1200-1399", "1400-1599", "1600-1799", "1800+"], type=str, nargs="+", 
                        help="List of Elo ranges to include (e.g., '1000-1199' '1800+')")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train or test)")
    gamemodes = [
            "gen1ou", "gen1randombattle", "gen2ou", "gen2randombattle", "gen3ou", "gen3randombattle",
            "gen4ou", "gen4randombattle", "gen5ou", "gen5randombattle", "gen6ou", "gen6randombattle",
            "gen7ou", "gen7randombattle", "gen8nationaldex", "gen8ou", "gen8randombattle",
            "gen9anythinggoes", "gen9doublesou", "gen9doublesubers", "gen9doublesuu", "gen9lc",
            "gen9monotype", "gen9nationaldex", "gen9nationaldexdoubles", "gen9nationaldexlc",
            "gen9nationaldexmonotype", "gen9nationaldexubers", "gen9nationaldexuu", "gen9nu",
            "gen9ou", "gen9pu", "gen9randombattle", "gen9randomdoublesbattle", "gen9ru",
            "gen9ubers", "gen9uu", "gen9vgc2024regg"
    ]

    parser.add_argument("--gamemode", default="gen9ou", type=str, choices=gamemodes,
                        help="Specific gamemode to filter by")
    args = parser.parse_args()
    
    # Load and filter the dataset
    filtered_dataset = load_filtered_dataset(
        min_month=args.min_month,
        max_month=args.max_month,
        elo_ranges=args.elo_ranges,
        split=args.split,
        gamemode=args.gamemode
    )
    
    # Get and print an example
    try:
        example = next(iter(filtered_dataset))
        print("Example from filtered dataset:")
        print(f"Month/Year: {example['month_year']}")
        print(f"Gamemode: {example['gamemode']}")
        print(f"Elo: {example['elo']}")
        print(f"Battle ID: {example['battle_id']}")
        print(f"Split: {args.split}")
        print("\nBattle Text:")
        print(example['text'][:500] + "..." if len(example['text']) > 500 else example['text'])
    except StopIteration:
        print("No examples found with the specified filters.")

if __name__ == "__main__":
    main()
