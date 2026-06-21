import pandas as pd    
from datetime import datetime
from zoneinfo import ZoneInfo
from scipy.signal import find_peaks

class MoonHelper:
    def __init__(self, moon_table_path, precise_moon_phase = False):
        df_hourly_moon = pd.read_csv(moon_table_path)
        df_hourly_moon['Illum_Change'] = df_hourly_moon['Illumination'].diff().fillna(0)
        self.df_hourly_moon = df_hourly_moon
        self.precise_moon_phase = precise_moon_phase

    def find_matched_indices(self, current_time, existing_hours):
        # Extract the exact illumination at this hour
        target_illum = self.df_hourly_moon.loc[current_time, 'Illumination']
        target_change = self.df_hourly_moon.loc[current_time, 'Illum_Change']

        # Determine the phase direction: True if getting brighter, False if getting darker
        is_waxing = target_change > 0 

        # 5. Map these times to your specific hourly index starting in Nov 2025
        # Replace this with your exact starting datetime
        cover_start_time = datetime.fromtimestamp(existing_hours[0]*3600, tz=ZoneInfo("America/Vancouver"))
        index_start_time = pd.to_datetime(cover_start_time)
        print("cover start time", index_start_time)
        # 3. Filter for past dates only
        df_valid = self.df_hourly_moon[self.df_hourly_moon.index <= current_time].copy()
        df_search = df_valid[df_valid.index > index_start_time].copy()

        # Calculate the baseline absolute difference
        df_search['Abs_Diff'] = (df_search['Illumination'] - target_illum).abs()

        # 4. The Magic Trick: Penalize the wrong phase
        # If we want a waning moon, we artificially inflate the Abs_Diff of all waxing moons 
        # by a huge number (e.g., 100) so find_peaks will never select them as a minimum.
        if is_waxing:
            wrong_phase_mask = df_search['Illum_Change'] <= 0
        else:
            wrong_phase_mask = df_search['Illum_Change'] > 0

        df_search.loc[wrong_phase_mask, 'Abs_Diff'] += 100

        # 5. Find the peaks (valleys)
        # Because we eliminated the opposite phase (which was ~14 days away), the next 
        # valid match is guaranteed to be a full lunar cycle (~29.5 days) away.
        # A distance of 20 days (20 * 24 hours) is now perfectly safe and won't skip February.
        peaks, _ = find_peaks(-df_search['Abs_Diff'], distance=20*24)

        # Extract the valid matches
        matched_data = df_search.iloc[peaks].copy()

        if not self.precise_moon_phase: #same hour of that day 
            current_hour = current_time.hour
            matched_data['Date_norm'] = matched_data.index.normalize() + pd.to_timedelta(current_hour, unit='h')
            matched_data['Hour_POSIX'] = matched_data["Date_norm"].view('int64') // (10**9*3600)
        else: 
            matched_data['Hour_POSIX'] = matched_data.index.view('int64') // (10**9*3600)

        is_in_list = matched_data['Hour_POSIX'].isin(existing_hours)
        
        filtered_matches = matched_data[is_in_list]

        # --- Verification ---
        print(f"Target Date: {current_time}")
        print(f"Target Illumination: {target_illum:.4f}\n")

        print("Matched Past Points (One per cycle):")
        print(filtered_matches[['Illumination', 'Abs_Diff', 'Hour_POSIX']])

        existing_data_indices = is_in_list[is_in_list].index.tolist()
        return existing_data_indices