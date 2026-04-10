import re

def is_date(s):
    date_patterns = [
        r'\d{1,2}-\d{1,2}-\d{2,4}',        # 04-09-2026, 4-9-26
        r'\d{2,4}-\d{1,2}-\d{1,2}',        # 2026-04-09
        r'[A-Za-z]{3}\s+\d{1,2}\s+\d{2,4}', # Mar 15 2025, Apr 3 25
        r'\d{1,2}/\d{1,2}/\d{2,4}',        # 04/09/2026
        r'\d{1,2}\.\d{1,2}\.\d{2,4}'       # 04.09.2026
    ]
    for p in date_patterns:
        if re.search(p, s):
            return True
    return False

def parse_date(date_str):
    """Parses date string into a sortable object (datetime) or float."""
    from datetime import datetime
    if not date_str:
        return datetime.min
    
    # Try common formats
    formats = ["%b %d %Y", "%m-%d-%Y", "%Y-%m-%d", "%d-%m-%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    
    # Simple regex fallback if standard parsing fails
    try:
        match = re.search(r'(\d+)', date_str)
        if match:
            return datetime(int(match.group(1)), 1, 1) # Just use year if found
    except:
        pass
        
    return datetime.min

def process_extracted_data(detections):
    """
    Processes the raw detections from inference.py into structured data
    for Excel generation in the frontend.
    """
    result = {
        "project_no": "N/A",
        "drawing_no": "",
        "drawing_description": "",
        "revisions": []
    }

    # If detections is empty or there was an error
    if not isinstance(detections, list):
        return result

    for det in detections:
        label = det.get("label")
        text = det.get("text", "")
        rows = det.get("rows", [])

        if label == "PROJECT_NO":
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            for l in lines:
                if re.match(r'^\d{5}$', l):
                    result["project_no"] = l
                    break

        elif label == "DRAWING_NO":
            lines = [l.strip() for l in text.split('\n') if l.strip() and "NO." not in l and "DWG" not in l]
            if lines:
                result["drawing_no"] = lines[0]
            
        elif label == "DRAWING_DESCRIPTION":
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            # Remove lines that are purely label noise (e.g. "NO. :", "DWG DESCRIPTION :")
            lines = [l for l in lines if not re.match(
                r'^(no\.?\s*:?|dwg\s*description\s*:?|drawing\s*description\s*:?)\s*$',
                l, re.IGNORECASE
            )]
            if lines:
                # Strip inline OCR label prefixes such as "NO. : DWG DESCRIPTION : COLUMN DETAIL"
                raw = lines[0]
                # Remove leading "NO. :" / "No:" variants
                raw = re.sub(r'^no\.?\s*:?\s*', '', raw, flags=re.IGNORECASE).strip()
                # Remove leading "DWG DESCRIPTION :" / "Drawing Description :" variants
                raw = re.sub(r'^(dwg\s*description|drawing\s*description)\s*:?\s*', '', raw, flags=re.IGNORECASE).strip()
                result["drawing_description"] = raw if raw else (lines[1] if len(lines) > 1 else "")
            else:
                result["drawing_description"] = ""
                
        elif label == "REVISION_TABLE" and rows:
            for row in rows:
                if row and isinstance(row[0], str):
                    lines = [l.strip() for l in row[0].split('\n') if l.strip()]
                    if len(lines) >= 1:
                        rev = lines[0] if len(lines) > 0 else '0'
                        
                        # Date extraction
                        date_found = ''
                        if len(lines) > 3 and is_date(lines[3]):
                            date_found = lines[3]
                        else:
                            # Try to find a date in remaining lines (same row only)
                            for l in reversed(lines):
                                if is_date(l):
                                    date_found = l
                                    break
                        
                        # Remarks extraction
                        remarks = ''
                        if len(lines) > 2 and len(lines[2]) >= 4 and not is_date(lines[2]):
                            remarks = lines[2]
                        else:
                            # Check other indices of the same row
                            for i, l in enumerate(lines):
                                if i == 0: continue # Skip rev index
                                if l == date_found: continue
                                if len(l) >= 4 and not is_date(l):
                                    remarks = l
                                    break

                        # Clean up "ISS" OCR misreads
                        if not re.search(r'ISS', remarks, re.IGNORECASE) and ('1SS' in remarks or 'lSS' in remarks):
                            remarks = re.sub(r'1SS', 'ISS', remarks, flags=re.IGNORECASE)
                            remarks = re.sub(r'lSS', 'ISS', remarks, flags=re.IGNORECASE)
                            
                        result["revisions"].append({
                            "rev": rev,
                            "date": date_found,
                            "remarks": remarks
                        })

    # Sort revisions by date (newest first)
    if result["revisions"]:
        result["revisions"].sort(key=lambda x: parse_date(x["date"]), reverse=True)
        result["latest_revision"] = result["revisions"][0]
    else:
        result["latest_revision"] = {
            "rev": "0",
            "date": "",
            "remarks": "ISSUED FOR CONSTRUCTION"
        }

    return result
