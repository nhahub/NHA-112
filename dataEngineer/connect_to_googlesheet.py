import gspread
from google.oauth2.service_account import Credentials
import re
import time
import pandas as pd

def setup_connection(credentials_file, sheet_id):
    """Setup Google Sheets API connection and return the workbook."""
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(credentials_file, scopes=scopes)
        client = gspread.authorize(creds)
        workbook = client.open_by_key(sheet_id)
        print(f"Successfully connected to Google Sheets API and opened workbook: {workbook.title}")
        return client, workbook
    except Exception as e:
        print(f"Error connecting to Google Sheets: {e}")
        raise

def open_worksheet(workbook, worksheet_name=None):
    """Open a specific worksheet."""
    try:
        if worksheet_name:
            worksheet = workbook.worksheet(worksheet_name)
            print(f"Opened worksheet: {worksheet.title}")
        else:
            worksheet = workbook.sheet1
            print(f"Opened first worksheet: {worksheet.title}")
        return worksheet
    except gspread.WorksheetNotFound:
        print(f"Worksheet '{worksheet_name}' not found.")
        raise
    except Exception as e:
        print(f"Error opening worksheet: {e}")
        raise

def get_worksheet_data_and_indices_safely(worksheet):
    """
    Get worksheet data with retry logic and better error handling.
    Returns data as a list of dictionaries along with original row numbers.
    """
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            print(f"Attempting to fetch data (attempt {attempt + 1}/{max_retries})...")

            all_values = worksheet.get_all_values()

            if len(all_values) < 2:
                print("No data rows found besides header.")
                return []

            headers = all_values[0]
            all_data = []

            # Start from row 2 (index 1) for data
            for i, row in enumerate(all_values[1:], start=2):
                row_dict = {headers[j]: row[j] if j < len(row) else '' for j in range(len(headers))}
                row_dict['original_row_number'] = i # Store the original row number
                all_data.append(row_dict)

            print(f"Successfully retrieved {len(all_data)} rows.")
            return all_data

        except Exception as e:
            print(f"Error fetching data (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("All attempts failed!")
                raise

if __name__ == '__main__':
    sheet_id = "1QN01gl4Irn3QlAWue0P7XdiOhmLFoG7LiU4oi-j7dLI"

    client, workbook = setup_connection("credentials.json", sheet_id)
    worksheet = open_worksheet(workbook, "Sheet1")
    data = get_worksheet_data_and_indices_safely(worksheet)
    df = pd.DataFrame(data)