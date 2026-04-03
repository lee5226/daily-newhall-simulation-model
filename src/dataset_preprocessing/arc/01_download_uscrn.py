"""
Filename: download_uscrn_daily01.py
Author: Moonyoung Lee
Date: 2025-06-19
Edited: 2025-07-06 연도별 하위 저장폴더
Description:
  - Downloads USCRN daily01 data files from NOAA for a specified year.
  - Automatically parses the list of .txt files from the public directory.
  - Downloads only if the file does not already exist locally.
  - Shows download progress using tqdm.

Data source:
  https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/<YEAR>/

Requirements:
  - Internet connection
  - Python packages: requests, beautifulsoup4, tqdm

Output:
  - Local directory: ./uscrn_daily01_<YEAR>/ containing downloaded .txt files
"""

import requests                     # For sending HTTP requests
import pathlib                     # For working with file paths
import re                          # For using regular expressions
from bs4 import BeautifulSoup      # For parsing HTML
from tqdm import tqdm              # For displaying a download progress bar

# ------------------------- Configuration -------------------------
BASE = "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01"
YEAR = 2008  # Change this value to download data for a different year
DEST = pathlib.Path("uscrn_daily01") / str(YEAR)  # Output folder path
DEST.mkdir(parents= True, exist_ok=True)  # Create the folder if it doesn't exist
# -----------------------------------------------------------------


def list_files(year: int) -> list[str]:
    """
    List all .txt filenames from the USCRN daily01 public directory for the given year.

    Args:
        year (int): Year to download data for

    Returns:
        list[str]: List of .txt filenames for that year
    """
    url = f"{BASE}/{year}/"
    html = requests.get(url, timeout=30).text            # Get directory HTML (timeout in 30 seconds)
    soup = BeautifulSoup(html, "html.parser")            # Parse HTML content
    pat = re.compile(fr"CRND0103-{year}-.*\.txt")        # Pattern for matching USCRN filenames
    return [a["href"] for a in soup.find_all("a") if pat.fullmatch(a["href"])]


def download(fname: str, year: int) -> None:
    """
    Download a single file from the USCRN server (skip if already exists).

    Args:
        fname (str): Filename to download
        year (int): Target year (used to construct the full URL)
    """
    url = f"{BASE}/{year}/{fname}"
    path = DEST / fname
    if path.exists():
        return  # Skip download if file already exists
    r = requests.get(url, timeout=60)      # Download the file (timeout in 60 seconds)
    r.raise_for_status()                   # Raise error if download fails
    path.write_bytes(r.content)           # Save content to file


if __name__ == "__main__":
    files = list_files(YEAR)
    print(f"Starting download of {len(files)} files for year {YEAR}")
    for f in tqdm(files, ncols=80):
        download(f, YEAR)
    print("Download complete.")