
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Space-Track URLS
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
DATA_URL = "https://www.space-track.org/basicspacedata/query/class/gp/decay_date/null-val/period/<225/orderby/norad_cat_id/format/tle"

# Get user credentials from environment variables
username = os.getenv("SPACE_TRACK_USERNAME")
password = os.getenv("SPACE_TRACK_PASSWORD")

if not username or not password:
    print("Please set SPACE_TRACK_USERNAME and SPACE_TRACK_PASSWORD in a .env file.")
    exit()

# Create a session
session = requests.session()

# Login
login_data = {"identity": username, "password": password}
response = session.post(LOGIN_URL, data=login_data)

# Check if login was successful
if response.status_code != 200:
    print("Login failed. Please check your credentials.")
    exit()

# Download data
response = session.get(DATA_URL)

# Check if data download was successful
if response.status_code != 200:
    print("Failed to download data.")
    exit()

# Save data to file
with open("leo_satellites.txt", "w") as f:
    f.write(response.text)

print("Successfully downloaded LEO satellite data to leo_satellites.txt")
