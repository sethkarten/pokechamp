import requests
import csv
from io import StringIO

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


if __name__ == "__main__":
    SHEET_ID = "1axlwmzPA49rYkqXh7zHvAtSP-TKbM0ijGYBPRflLSWw"
    GID = "418553327"  # specific tab
    
    paste_links = extract_pokepaste_links(SHEET_ID, GID)
    print("Found PokéPaste links:", paste_links[:5], "...")
    
    # Example: fetch the first team's data
    if paste_links:
        team_data = fetch_pokepaste(paste_links[0])
        print("=== Example Team Data ===")
        print(team_data)
