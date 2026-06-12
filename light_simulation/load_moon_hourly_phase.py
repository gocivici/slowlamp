import pandas as pd


def read_navy_moon_table(csv_name, year):
    # 1. Load the data
    # The image shows 3 header rows before the actual table starts ("Day", "1", "2"...). 
    # We use skiprows=3 to skip the title rows and use the "Day" row as the header.
    # (If you are using an Excel file, use pd.read_excel('filename.xlsx', skiprows=3))
    df = pd.read_csv(csv_name, skiprows=3)

    # 2. Reshape the data from wide to long
    # This converts the month columns (1-12) into a single 'Month' column
    df_long = df.melt(id_vars=['Day'], var_name='Month', value_name='Illumination')

    # Ensure Month is an integer (sometimes it gets read as a string during melt)
    df_long['Month'] = df_long['Month'].astype(int)

    # Add the year based on the header in your image
    df_long['Year'] = year

    # 3. Create Date objects and handle invalid dates
    # Months have different numbers of days. When we melt a 31-day table, we'll get 
    # invalid dates like Feb 30 or April 31. 
    # errors='coerce' safely turns these impossible dates into 'NaT' (Not a Time).
    df_long['Date'] = pd.to_datetime(df_long[['Year', 'Month', 'Day']], errors='coerce')

    # Drop the invalid dates (NaT) and any completely empty cells
    df_clean = df_long.dropna(subset=['Date', 'Illumination'])

    # 4. Sort chronologically
    df_clean = df_clean.sort_values(by='Date').reset_index(drop=True)
    df_clean['Illumination'] = df_clean['Illumination'].astype(float)


    # # 5. Extract into the two requested arrays
    # dates_array = df_clean['Date'].values
    # illumination_array = df_clean['Illumination'].values

    return df_clean

# --- Verification ---
# print("First 5 dates:")
# print(dates_array[:5])
# print("\nFirst 5 illumination fractions:")
# print(illumination_array[:5])