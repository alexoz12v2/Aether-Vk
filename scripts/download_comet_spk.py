import urllib.request
import urllib.parse
import json
import ssl
import os
import re
import base64
from datetime import datetime, timedelta

def sanitize_filename(name):
    # Replace non-alphanumeric characters with underscores
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name).strip('_')

def get_comets():
    print("Fetching list of comets from JPL Small-Body Database Query API...")
    # First call: list all comets. We use the SBDB query API which is the modern approach
    # to query all comets from JPL's database.
    url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api?fields=full_name,pdes&sb-kind=c"
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ctx) as response:
        data = json.loads(response.read().decode('utf-8'))
        return data.get("data", [])

def download_spk(comet_name, designation, start_time, stop_time, command_override=None):
    print(f"Downloading SPK file for comet {comet_name} (Designation: {designation})")
    print(f"Period: {start_time} to {stop_time}")
    
    base_url = "https://ssd.jpl.nasa.gov/api/horizons.api"
    # To reliably fetch an SPK for a small body, format the command as 'DES=<designation>;'
    # If a specific record ID is provided (e.g. from a previous multi-match), use it.
    command = command_override if command_override else f"DES={designation};"
    
    params = {
        "format": "json",
        "COMMAND": f"'{command}'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'SPK'",
        "START_TIME": f"'{start_time}'",
        "STOP_TIME": f"'{stop_time}'",
        "CENTER": "'@0'", # SSB center
    }
    
    query_string = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_string}"
    print(f"Querying Horizons API with COMMAND={command}")
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, context=ctx) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            
            if "spk" in res_data:
                spk_base64 = res_data["spk"]
                spk_binary = base64.b64decode(spk_base64)
                
                safe_name = sanitize_filename(comet_name)
                period = f"{start_time}_to_{stop_time}"
                filename = f"{safe_name}_{period}.bsp"
                
                # save in script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                filepath = os.path.join(script_dir, filename)
                
                with open(filepath, "wb") as f:
                    f.write(spk_binary)
                    
                print(f"Successfully downloaded and saved: {filepath}")
                return filepath
            else:
                result_text = res_data.get("result", "")
                if "multiple matches" in result_text.lower() or "multiple major-bodies" in result_text.lower() or "matches. to select" in result_text.lower():
                    print("Multiple matches found. Attempting to select the most recent record...")
                    # Extract record numbers: match lines like "    90000702    2015    67P"
                    record_numbers = re.findall(r'\n\s*(-?\d+)\s+\d{4}\s+', result_text)
                    if not record_numbers:
                        # Fallback for other formats
                        record_numbers = re.findall(r'\n\s*(-?\d+)\s+[a-zA-Z0-9-]+\s+', result_text)
                        
                    if record_numbers:
                        best_record = record_numbers[-1]
                        print(f"Selected record #{best_record}. Retrying...")
                        return download_spk(comet_name, designation, start_time, stop_time, f"{best_record};")
                    else:
                        print("Error: Could not parse record numbers from multi-match response.")
                        print("Result message:", result_text)
                else:
                    print("Error: No SPK data returned in response.")
                    print("Result message:", result_text)
    except Exception as e:
        print(f"HTTP Request failed: {e}")

if __name__ == "__main__":
    comets = get_comets()
    if not comets:
        print("Could not retrieve comets.")
        exit(1)
        
    print(f"Found {len(comets)} comets.")
    # Pick a well known comet for demonstration (e.g. 67P/Churyumov-Gerasimenko)
    # If not found, fallback to first comet in list.
    target_comet = comets[0]
    for c in comets:
        if "67P" in c[0]:
            target_comet = c
            break
            
    full_name = target_comet[0].strip()
    designation = target_comet[1].strip()
    
    # arbitrary 1 week period
    start_date = datetime.now()
    stop_date = start_date + timedelta(days=7)
    start_str = start_date.strftime("%Y-%m-%d")
    stop_str = stop_date.strftime("%Y-%m-%d")
    
    download_spk(full_name, designation, start_str, stop_str)
