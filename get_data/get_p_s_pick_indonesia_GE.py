import os
import pandas as pd
import time
import glob
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from tqdm import tqdm

# Inisialisasi FDSN client (GFZ)
client = Client("GFZ")

# Wilayah Indonesia
minlat, maxlat = -11.0, 6.0
minlon, maxlon = 95.0, 141.0
min_magnitude = 2.0

# Rentang tahun
start_year = 2007
end_year = 2025

# Direktori penyimpanan
output_dir = "./dataset/p_s_pick_metadata_complete"
os.makedirs(output_dir, exist_ok=True)

# File paths
log_file = os.path.join(output_dir, "log_progress.txt")
merged_csv = os.path.join(output_dir, "p_s_pick_metadata_merged.csv")
filtered_csv = os.path.join(output_dir, "p_s_pick_metadata_filtered.csv")

# Daftar kategori arrival P-wave dan S-wave
p_phases = {"p", "pp", "pg", "pn", "pb", "pdiff", "pkp", "pkikp"}
s_phases = {"s", "ss", "sg", "sn", "sb", "sdiff", "sks", "skiks"}

# Daftar 21 stasiun yang valid
valid_stations = {
    "BBJI", "BKB", "BKNI", "BNDI", "FAKI", "GENI", "GSI", "JAGI", "LHMI", "LUWI", "MMRI",
    "MNAI", "PLAI", "PMBI", "SANI", "SAUI", "SMRI", "SOEI", "TNTI", "TOLI2", "UGM"
}

MAX_RETRIES = 3  # Maksimal retries jika gagal

# ==================== MODULARIZED FUNCTIONS ====================

def log_message(message):
    """Menulis log ke file"""
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)

def fetch_catalog(year, start_month, end_month):
    """Mengambil katalog event per 6 bulan dengan retries"""
    starttime = UTCDateTime(f"{year}-{start_month:02d}-01")
    endtime = UTCDateTime(f"{year}-{end_month:02d}-30")

    retries = 0
    while retries < MAX_RETRIES:
        try:
            log_message(f"Fetching earthquake catalog for {year} ({start_month}-{end_month}), attempt {retries+1}...")
            catalog = client.get_events(starttime=starttime, endtime=endtime,
                                        minlatitude=minlat, maxlatitude=maxlat,
                                        minlongitude=minlon, maxlongitude=maxlon,
                                        minmagnitude=min_magnitude, includearrivals=True)
            log_message(f"Total events for {year} ({start_month}-{end_month}): {len(catalog)}")
            return catalog
        except Exception as e:
            log_message(f"Error fetching catalog for {year} ({start_month}-{end_month}): {e}")
            retries += 1
            time.sleep(5)  # Tunggu 5 detik sebelum mencoba ulang

    log_message(f"Failed to fetch catalog for {year} ({start_month}-{end_month}) after {MAX_RETRIES} retries. Skipping batch...")
    return []

def extract_event_metadata(event):
    """Mengambil metadata event"""
    return {
        "event_id": event.resource_id.id.split('/')[-1],
        "origin_time": str(event.origins[0].time),
        "latitude": event.origins[0].latitude,
        "longitude": event.origins[0].longitude,
        "depth_km": event.origins[0].depth / 1000.0,  # Convert depth to km
        "magnitude": event.magnitudes[0].mag if event.magnitudes else None,
        "magnitude_type": event.magnitudes[0].magnitude_type if event.magnitudes else None,
        "event_type": event.event_type if hasattr(event, "event_type") else "earthquake",
        "event_region": event.event_descriptions[0].text if event.event_descriptions else None,
        "arrivals_count": len(event.picks)
    }

def find_p_pick(event, s_pick_time, station_code):
    """Mencari P-pick terdekat sebelum S-pick dalam event yang sama"""
    p_pick = None
    min_time_diff = float("inf")  # Inisialisasi selisih waktu yang sangat besar

    for pick in event.picks:
        if pick.phase_hint.lower() in p_phases and pick.waveform_id.station_code == station_code:
            time_diff = (s_pick_time - pick.time)
            if 0 < time_diff < min_time_diff:  # Cari P-pick sebelum S-pick dengan selisih waktu terkecil
                min_time_diff = time_diff
                p_pick = pick

    return p_pick, min_time_diff if p_pick else None

def extract_p_s_picks(event):
    """Mengambil semua S-pick dan mencari P-pick yang sesuai"""
    picks = []
    for pick in event.picks:
        phase = pick.phase_hint.lower()
        if phase in s_phases and pick.waveform_id.station_code in valid_stations:
            s_pick_time = pick.time
            p_pick, time_diff = find_p_pick(event, s_pick_time, pick.waveform_id.station_code)
            p_pick_time = str(p_pick.time) if p_pick else None
            p_pick_station = p_pick.waveform_id.station_code if p_pick else None
            p_pick_network = p_pick.waveform_id.network_code if p_pick else None
            p_pick_channel = p_pick.waveform_id.channel_code if p_pick else None
            p_pick_phase = p_pick.phase_hint.lower() if p_pick else None
            p_pick_method = p_pick.method_id.id if p_pick and p_pick.method_id else None
            p_pick_time_uncertainty = p_pick.time_errors.uncertainty if p_pick and p_pick.time_errors else None
            
            picks.append({
                # S-pick metadata
                "s_pick_time": str(s_pick_time),
                "s_pick_station": pick.waveform_id.station_code,
                "s_pick_network": pick.waveform_id.network_code,
                "s_pick_channel": pick.waveform_id.channel_code,
                "s_pick_phase": phase,
                "s_pick_method": pick.method_id.id if pick.method_id else None,
                "s_pick_time_uncertainty": pick.time_errors.uncertainty if pick.time_errors else None,

                # P-pick metadata (bisa None jika tidak ditemukan)
                "p_pick_time": p_pick_time,
                "p_pick_station": p_pick_station,
                "p_pick_network": p_pick_network,
                "p_pick_channel": p_pick_channel,
                "p_pick_phase": p_pick_phase,
                "p_pick_method": p_pick_method,
                "p_pick_time_uncertainty": p_pick_time_uncertainty,

                # Interval P-S
                "p_s_interval_sec": time_diff if p_pick else None
            })

    return picks

def merge_csv_files():
    """Menggabungkan semua file CSV di direktori ke dalam satu file CSV"""
    log_message("Merging all CSV files into a single file...")
    
    # Mencari semua file CSV di direktori
    csv_files = glob.glob(os.path.join(output_dir, "p_s_pick_metadata_*.csv"))
    
    # Filter file CSV untuk hanya menyertakan file batch, bukan file merged atau filtered
    batch_csvs = [f for f in csv_files if 'merged' not in f and 'filtered' not in f]
    
    if not batch_csvs:
        log_message("No CSV files found to merge!")
        return False
    
    log_message(f"Found {len(batch_csvs)} CSV files to merge.")
    
    # Menggabungkan semua file CSV
    dfs = []
    for csv_file in tqdm(batch_csvs, desc="Reading CSV files", unit="file"):
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            log_message(f"Error reading {csv_file}: {e}")
    
    if not dfs:
        log_message("No valid CSV data found for merging!")
        return False
    
    # Menggabungkan semua dataframe
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Simpan ke file CSV
    merged_df.to_csv(merged_csv, index=False)
    log_message(f"Merged CSV file saved at: {merged_csv}")
    log_message(f"Total records in merged file: {len(merged_df)}")
    
    return True

def filter_merged_data():
    """Filter data merged untuk hanya menyertakan baris dengan P dan S pick"""
    log_message("Filtering merged data to include only rows with both P and S picks...")
    
    try:
        # Membaca file CSV
        df = pd.read_csv(merged_csv)
        log_message(f"🔍 Total data sebelum filtering: {len(df)}")
        
        # Filter hanya yang memiliki P dan S pick
        if 'p_pick_time' in df.columns and 's_pick_time' in df.columns:
            df_filtered = df.dropna(subset=['p_pick_time', 's_pick_time'])
            log_message(f"✅ Total data setelah filtering (hanya yang punya P dan S pick): {len(df_filtered)}")
            
            # Simpan ke CSV
            df_filtered.to_csv(filtered_csv, index=False)
            log_message(f"📂 File hasil filtering disimpan di: {filtered_csv}")
            return True
        else:
            log_message("❌ Kolom 'p_pick_time' atau 's_pick_time' tidak ditemukan di dataset!")
            return False
            
    except Exception as e:
        log_message(f"Error filtering data: {e}")
        return False

# ==================== MAIN EXECUTION ====================

def fetch_and_process_data():
    """Fetch and process earthquake data"""
    log_message("Starting the P-S pick metadata extraction process...\n")

    for year in range(start_year, end_year + 1):
        for batch_start, batch_end in [(1, 6), (7, 12)]:  # Membagi per 6 bulan
            log_message(f"\nProcessing {year} ({batch_start}-{batch_end})...")

            catalog = fetch_catalog(year, batch_start, batch_end)
            if not catalog:
                continue  # Skip jika tidak ada event

            # Proses setiap event
            metadata_list = []
            for event in tqdm(catalog, desc=f"Processing {year} ({batch_start}-{batch_end})", unit="event"):
                event_metadata = extract_event_metadata(event)
                p_s_picks = extract_p_s_picks(event)

                for pick in p_s_picks:
                    metadata_list.append({**event_metadata, **pick})

            # Simpan CSV per batch (6 bulan)
            csv_output_path = os.path.join(output_dir, f"p_s_pick_metadata_{year}_{batch_start}-{batch_end}.csv")
            df_csv = pd.DataFrame(metadata_list)
            df_csv.to_csv(csv_output_path, index=False)

            log_message(f"Finished processing {year} ({batch_start}-{batch_end}). Saved CSV at {csv_output_path}\n")

    log_message("All years processed successfully.\n")

if __name__ == "__main__":
    # Step 1: Fetch and process data
    fetch_and_process_data()
    
    # Step 2: Merge all CSV files
    if merge_csv_files():
        # Step 3: Filter merged data
        filter_merged_data()
    
    log_message("Process completed!")