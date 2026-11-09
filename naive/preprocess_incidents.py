import pandas as pd
import re
import json
from datetime import datetime

def _str(row, key):
    val = row.get(key, '')
    if pd.isna(val):
        return ''
    return str(val)

def preprocess_incident(row):
    # 1. Extract Primary Text Fields
    short_desc  = _str(row, 'Kurzbeschreibung')
    full_desc   = _str(row, 'Beschreibung')
    work_notes  = _str(row, 'Arbeitsnotizenliste')

    # 2. Extract Metadata
    metadata = {
        'incident_id':       _str(row, 'Nummer'),
        'created_at':        _str(row, 'Erstellt'),
        'status':            _str(row, 'Status'),
        'priority':          _str(row, 'Priorität'),
        'assignment_group':  _str(row, 'Zuweisungsgruppe'),
        'department':        _str(row, 'Abteilung'),
        'reporter':          _str(row, 'Melder'),
    }    
    # 3. Parse Description for Department (if not a separate column)
    # Looks for common patterns like "Abteilung: ..." or "Department: ..."
    dept_match = re.search(r'(?:Abteilung|Department)[\s:]*([^\n]+)', full_desc, re.IGNORECASE)
    if dept_match and not metadata['department']:
        metadata['department'] = dept_match.group(1).strip()
    
    # 4. Clean and Consolidate Text
    # Combine short description, full description, and work notes
    # Weight the full description more heavily than notes
    text_parts = [
        f"SHORT_DESC: {short_desc}",
        f"DESCRIPTION: {full_desc}",
        f"WORK_NOTES: {work_notes}"
    ]
    
    # Clean each part
    cleaned_parts = []
    for part in text_parts:
        if part.strip():
            # Remove extra whitespace
            part = re.sub(r'\s+', ' ', part)
            # Remove special characters but keep German umlauts and apostrophes
            part = re.sub(r"[^\w\säöüÄÖÜß\-\.\,\#\@\!\?\':\'\'\"\\']+", ' ', part)
            # Remove excessive timestamps in work notes if they clutter the text
            # (Optional: remove if you want pure semantic analysis)
            part = re.sub(r'\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}:\d{2}\s+', '', part)
            if part.strip():
                cleaned_parts.append(part)
    
    consolidated_text = "\n\n".join(cleaned_parts)
    
    # 5. Return result
    return {
        'incident_id': metadata['incident_id'],
        'priority': metadata['priority'],
        'status': metadata['status'],
        'assignment_group': metadata['assignment_group'],
        'department': metadata['department'],
        'consolidated_text': consolidated_text,
        'created_at': metadata['created_at']
    }

def load_and_clean_excel(file_path, output_file='cleaned_incidents.csv'):
    """
    Loads Excel file, processes each row, and saves to CSV/JSON.
    """
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    print(f"Loaded {len(df)} incidents.")
    
    # Apply preprocessing function
    processed_data = []
    for index, row in df.iterrows():
        result = preprocess_incident(row)
        if result['consolidated_text'].strip():  # Skip empty rows
            processed_data.append(result)
    
    # Convert to DataFrame for easy export
    df_processed = pd.DataFrame(processed_data)
    
    # Save to CSV and JSON
    df_processed.to_csv(output_file, index=False, encoding='utf-8')
    df_processed.to_json(
        output_file.replace('.csv', '.json'), 
        orient='records', 
        force_ascii=False,
        indent=2
    )
    
    print(f"Processed {len(processed_data)} incidents.")
    print(f"Output saved to: {output_file} and {output_file.replace('.csv', '.json')}")
    
    return df_processed

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="ServiceNow incident Excel export -> cleaned_incidents.csv/.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default="incidents.xlsx", help="Excel export")
    parser.add_argument("--output", default="cleaned_incidents.csv",
                        help="output CSV; a .json with the same name is written too")
    args = parser.parse_args()
    input_file = args.input

    try:
        df = load_and_clean_excel(input_file, args.output)
        
        # Show sample output
        print("\n--- Sample Row ---")
        print(f"Incident ID: {df.iloc[0]['incident_id']}")
        print(f"Priority: {df.iloc[0]['priority']}")
        print(f"Assignment Group: {df.iloc[0]['assignment_group']}")
        print(f"Text Preview: {df.iloc[0]['consolidated_text'][:200]}...")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found. Please update the path.")
    except Exception as e:
        print(f"An error occurred: {e}")
