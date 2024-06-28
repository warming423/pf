import pandas as pd
import json,os

def convert_to_excel(json_path, excel_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        trace_data = json.load(f)
    
    # Extract the trace events
    trace_events = trace_data.get('traceEvents', [])

    # Normalize the trace events
    normalized_data = []
    for event in trace_events:
        normalized_data.append(pd.json_normalize(event))

    # Concatenate all normalized events into a single DataFrame
    if normalized_data:
        trace_df = pd.concat(normalized_data, ignore_index=True)
        trace_df.to_excel(excel_path, index=False)
    else:
        # If there are no trace events, create an empty DataFrame and save
        pd.DataFrame().to_excel(excel_path, index=False)
        
convert_to_excel("/home/wangm/pf/data/aisec-dell-server_3259892.1716260363283.pt.trace.json","/home/wangm/pf/data/fsf.xlsx")