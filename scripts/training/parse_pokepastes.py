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

def create_teambuilder(url: str):
    raw = fetch_pokepaste(url).replace('\r\n', '\n')
    mons = Teambuilder.parse_showdown_team(raw)
    team = Teambuilder.join_team(mons)

    return mons, team

def encode_team_to_base64(url: str) -> str:
    raw = fetch_pokepaste(url).replace('\r\n', '\n')
    return base64.b64encode(raw.encode("utf-8")).decode("utf-8")

if __name__ == "__main__":
    SHEET_ID = "1axlwmzPA49rYkqXh7zHvAtSP-TKbM0ijGYBPRflLSWw"
    GID = "418553327"  # specific tab
    
    paste_links = extract_pokepaste_links(SHEET_ID, GID)
    print("Found PokéPaste links:", paste_links[:5], "...")
    
    # Example: fetch the first team's data
    if paste_links:
        team_objs, packed = create_teambuilder(paste_links[0])
        print(packed)

    # test encoded version like data in metamon-teams
    print(encode_team_to_base64(paste_links[0]))
