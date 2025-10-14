import sys
import os
import requests
import csv
from io import StringIO
import base64

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)


from poke_env.teambuilder.teambuilder import Teambuilder

def fetch_pokepaste(url: str) -> str:
    """Fetch raw team data from a PokéPaste URL."""
    if not url.endswith("/raw"):
        if url.endswith("/"):
            url += "raw"
        else:
            url += "/raw"

    response = requests.get(url)
    response.raise_for_status()
    return response.text


def extract_pokepaste_links(sheet_id: str, gid: str) -> list:
    """
    Extract all PokéPaste links from a given Google Sheet tab.
    
    Args:
        sheet_id: The Google Sheet ID (long string in the URL).
        gid: The tab's gid value (from the sheet URL).
    
    Returns:
        List of PokéPaste URLs found in the tab.
    """
    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
    response = requests.get(export_url)
    response.raise_for_status()
    
    csv_data = response.text
    reader = csv.reader(StringIO(csv_data))
    
    links = []
    for row in reader:
        for cell in row:
            if "pokepast.es" in cell:
                links.append(cell.strip())
    return links

# deprecated 
def create_teambuilder(url: str):
    raw = fetch_pokepaste(url).replace('\r\n', '\n')
    mons = Teambuilder.parse_showdown_team(raw)
    team = Teambuilder.join_team(mons)

    return mons, team

def export_team_to_file(team_data: str, output_dir: str, team_number: int, battle_format: str = "gen9ou"):
    """Export a team to a file in the correct format for the Bayesian predictor."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"team_{team_number:06d}.{battle_format}_team"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(team_data)
    
    return filepath


if __name__ == "__main__":
    if __name__ == "__main__":
        SHEET_ID = "1axlwmzPA49rYkqXh7zHvAtSP-TKbM0ijGYBPRflLSWw"
        GID = "418553327"  # specific tab
        
        # Configuration
        OUTPUT_DIR = "bayesian_dataset"  # or wherever you want to save teams
        BATTLE_FORMAT = "gen9vgc2025regi"  # or your target format
        
        paste_links = extract_pokepaste_links(SHEET_ID, GID)
        print(f"Found {len(paste_links)} PokéPaste links")
        
        successful_exports = 0
        failed_exports = 0
        
        for i, url in enumerate(paste_links):
            try:
                # Fetch and validate team
                raw_team = fetch_pokepaste(url)
                
                # Basic validation - check if it looks like a valid team
                if "Ability:" in raw_team and "- " in raw_team:
                    # Export to file
                    filepath = export_team_to_file(raw_team, OUTPUT_DIR, i, BATTLE_FORMAT)
                    successful_exports += 1
                    if(successful_exports % 100 == 0):
                        print(f"Exported {successful_exports} teams")
                else:
                    failed_exports += 1
                    
            except Exception as e:
                failed_exports += 1
                continue
        
        print(f"\nExport complete!")
        print(f"Successful: {successful_exports}")
        print(f"Failed: {failed_exports}")
        print(f"Teams saved to: {os.path.abspath(OUTPUT_DIR)}")
