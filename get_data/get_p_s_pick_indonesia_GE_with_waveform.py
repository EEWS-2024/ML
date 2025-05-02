import os
import pandas as pd
import numpy as np
import time
import glob
from datetime import datetime
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from tqdm import tqdm
import matplotlib.pyplot as plt
from obspy import read

# ==================== KONFIGURASI ====================

# Inisialisasi FDSN client (GFZ)
client = Client("GFZ")

# Wilayah Indonesia
minlat, maxlat = -11.0, 6.0
minlon, maxlon = 95.0, 141.0
min_magnitude = 2.0

# Rentang tahun
start_year = 2005
end_year = 2025

# Parameter waktu
PRE_P_TIME = 30  # Detik sebelum fase P
POST_S_TIME = 30  # Detik setelah fase S
MAX_DURATION = 300  # Durasi maksimal rekaman (detik)

# Parameter sampling
TARGET_SAMPLING_RATE = 100  # Hz (standar untuk PhaseNet)

# Konfigurasi untuk panjang tetap
MIN_FIXED_LENGTH = 3000  # Panjang minimum dalam sampel (30 detik pada 100 Hz)
MAX_FIXED_LENGTH = 60000  # Panjang maksimum dalam sampel (10 menit pada 100 Hz)

# Direktori penyimpanan
base_dir = "./dataset_phasenet_from_2005"
metadata_dir = os.path.join(base_dir, "metadata")
waveform_dir = os.path.join(base_dir, "waveform")
npz_dir = os.path.join(base_dir, "npz")
figure_dir = os.path.join(base_dir, "figures")

# Buat direktori
for d in [metadata_dir, waveform_dir, npz_dir, figure_dir]:
    os.makedirs(d, exist_ok=True)

# File paths
log_file = os.path.join(metadata_dir, "log_progress.txt")
merged_csv = os.path.join(metadata_dir, "p_s_pick_metadata_merged.csv")
filtered_csv = os.path.join(metadata_dir, "p_s_pick_metadata_filtered.csv")
config_file = os.path.join(base_dir, "dataset_config.json")
data_list_csv = os.path.join(base_dir, "data_list.csv")

# Daftar kategori arrival P-wave dan S-wave
p_phases = {"p", "pp", "pg", "pn", "pb", "pdiff", "pkp", "pkikp"}
s_phases = {"s", "ss", "sg", "sn", "sb", "sdiff", "sks", "skiks"}

# Daftar 21 stasiun yang valid
valid_stations = {
    "BBJI", "BKB", "BKNI", "BNDI", "FAKI", "GENI", "GSI", "JAGI", "LHMI", "LUWI", "MMRI",
    "MNAI", "PLAI", "PMBI", "SANI", "SAUI", "SMRI", "SOEI", "TNTI", "TOLI2", "UGM"
}

MAX_RETRIES = 3  # Maksimal retries jika gagal

# ==================== FUNGSI LOGGING ====================

def log_message(message):
    """Menulis log ke file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"

    with open(log_file, "a") as f:
        f.write(log_entry + "\n")
    print(log_entry)

# ==================== FUNGSI PENGAMBILAN METADATA ====================

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
        "depth_km": event.origins[0].depth / 1000.0 if event.origins[0].depth else None,  # Convert depth to km
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
    csv_files = glob.glob(os.path.join(metadata_dir, "p_s_pick_metadata_*.csv"))

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

            # Filter tambahan untuk memastikan data berkualitas baik
            df_filtered = df_filtered[
                (df_filtered['p_s_interval_sec'] > 0)  # P harus sebelum S
            ]

            log_message(f"✅ Total data setelah filtering (hanya yang punya P dan S pick dengan interval wajar): {len(df_filtered)}")

            # Simpan ke CSV
            df_filtered.to_csv(filtered_csv, index=False)
            log_message(f"📂 File hasil filtering disimpan di: {filtered_csv}")

            # Analisis distribusi jarak P-S
            analyze_ps_interval(df_filtered)

            return True
        else:
            log_message("❌ Kolom 'p_pick_time' atau 's_pick_time' tidak ditemukan di dataset!")
            return False

    except Exception as e:
        log_message(f"Error filtering data: {e}")
        return False

def analyze_ps_interval(df):
    """Analisis distribusi jarak P-S untuk membantu menentukan panjang tetap"""
    log_message("Analyzing P-S interval distribution...")

    intervals = df['p_s_interval_sec'].dropna()

    # Statistik dasar
    min_interval = intervals.min()
    max_interval = intervals.max()
    mean_interval = intervals.mean()
    median_interval = intervals.median()
    p95_interval = intervals.quantile(0.95)

    log_message(f"P-S Interval Statistics:")
    log_message(f"  Min: {min_interval:.2f} seconds")
    log_message(f"  Max: {max_interval:.2f} seconds")
    log_message(f"  Mean: {mean_interval:.2f} seconds")
    log_message(f"  Median: {median_interval:.2f} seconds")
    log_message(f"  95th Percentile: {p95_interval:.2f} seconds")

    # Buat histogram untuk visualisasi
    plt.figure(figsize=(10, 6))
    plt.hist(intervals, bins=50, alpha=0.7)
    plt.axvline(p95_interval, color='r', linestyle='--', label=f'95th Percentile: {p95_interval:.2f}s')
    plt.axvline(max_interval, color='g', linestyle='--', label=f'Max: {max_interval:.2f}s')
    plt.xlabel('P-S Interval (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of P-S Time Intervals')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'ps_interval_distribution.png'))
    plt.close()

    return p95_interval, max_interval

def determine_fixed_length(df):
    """Menentukan panjang tetap berdasarkan analisis dataset"""
    try:
        # Dapatkan statistik interval P-S
        p95_interval, max_interval = analyze_ps_interval(df)

        # Hitung panjang yang dibutuhkan (dalam sampel)
        # Gunakan persentil ke-95 sebagai dasar dengan tambahan margin
        p95_samples = int((p95_interval + PRE_P_TIME + POST_S_TIME) * TARGET_SAMPLING_RATE)
        max_samples = int((max_interval + PRE_P_TIME + POST_S_TIME) * TARGET_SAMPLING_RATE)

        # Pilih panjang yang masuk akal
        if max_samples <= MAX_FIXED_LENGTH:
            # Jika panjang maksimum masih dalam batas wajar, gunakan itu
            fixed_length = max_samples
            log_message(f"Using maximum P-S interval for fixed length: {fixed_length} samples ({fixed_length/TARGET_SAMPLING_RATE:.2f} seconds)")
        else:
            # Jika terlalu panjang, gunakan persentil ke-95 atau batas maksimum
            fixed_length = min(p95_samples, MAX_FIXED_LENGTH)
            log_message(f"Using 95th percentile P-S interval for fixed length: {fixed_length} samples ({fixed_length/TARGET_SAMPLING_RATE:.2f} seconds)")
            log_message(f"Note: {len(df[df['p_s_interval_sec'] > fixed_length/TARGET_SAMPLING_RATE - (PRE_P_TIME + POST_S_TIME)])} events have P-S intervals that exceed this fixed length")

        # Pastikan panjang minimal tercapai
        fixed_length = max(fixed_length, MIN_FIXED_LENGTH)

        # Simpan konfigurasi untuk digunakan dengan PhaseNet
        import json
        config = {
            "fixed_length_samples": int(fixed_length),
            "fixed_length_seconds": float(fixed_length / TARGET_SAMPLING_RATE),
            "sampling_rate": float(TARGET_SAMPLING_RATE),
            "pre_p_time": float(PRE_P_TIME),
            "post_s_time": float(POST_S_TIME)
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        return fixed_length

    except Exception as e:
        log_message(f"Error determining fixed length: {e}")
        # Default jika terjadi error
        return MIN_FIXED_LENGTH

def fetch_and_process_metadata():
    """Fetch and process earthquake metadata"""
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
            csv_output_path = os.path.join(metadata_dir, f"p_s_pick_metadata_{year}_{batch_start}-{batch_end}.csv")
            df_csv = pd.DataFrame(metadata_list)
            df_csv.to_csv(csv_output_path, index=False)

            log_message(f"Finished processing {year} ({batch_start}-{batch_end}). Saved CSV at {csv_output_path}\n")

    log_message("All years processed successfully.\n")

# ==================== FUNGSI PENGAMBILAN WAVEFORM ====================

def download_waveform(row, idx, fixed_length_seconds):
    """Download waveform data for a specific P-S pick pair with fixed length consideration"""
    try:
        # Extract required information
        network = row['p_pick_network']
        station = row['p_pick_station']
        channel_base = row['p_pick_channel'][:-1] if not pd.isna(row.get('p_pick_channel')) else 'BH'
        p_time = UTCDateTime(row['p_pick_time'])
        s_time = UTCDateTime(row['s_pick_time'])
        
        # Format filename
        event_id = row.get('event_id', f"EV_{p_time.strftime('%Y%m%d_%H%M%S')}")
        filename = f"{network}.{station}.{event_id}.mseed"
        full_path = os.path.join(waveform_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(full_path):
            return idx, filename, True, "File already exists"
        
        # Calculate time window - simpler approach
        p_s_interval = s_time - p_time
        margin = min(PRE_P_TIME, fixed_length_seconds * 0.3)  # Use at most 30% of fixed length for pre-P time
        
        starttime = p_time - margin
        endtime = starttime + fixed_length_seconds
        
        # Log the request details for debugging
        log_message(f"Requesting waveform for {filename}: {network}.{station} from {starttime} to {endtime}")
        
        # Initialize empty stream
        stream = Stream()
        
        # Try to get each component separately - more reliable than wildcards
        components = ['Z', 'N', 'E']  # Typical component order
        success = False
        
        # First attempt: Try with the channel base from metadata
        for comp in components:
            try:
                channel = f"{channel_base}{comp}"
                st_comp = client.get_waveforms(network, station, "*", channel, starttime, endtime)
                if st_comp:
                    stream += st_comp
                    success = True
            except Exception as e:
                log_message(f"  Failed to get {channel}: {str(e)}")
        
        # Second attempt: Try with common channel types if first attempt failed
        if not success:
            for base in ['BH', 'HH', 'EH', 'SH']:  # Common instrument types
                if success:
                    break
                    
                try:
                    for comp in components:
                        channel = f"{base}{comp}"
                        st_comp = client.get_waveforms(network, station, "*", channel, starttime, endtime)
                        if st_comp:
                            stream += st_comp
                            success = True
                except Exception as e:
                    log_message(f"  Failed to get {base} channels: {str(e)}")
        
        # Alternative attempt: Try with 12Z components (for some stations)
        if not success:
            try:
                for comp in ['1', '2', 'Z']:
                    channel = f"{channel_base[:-1]}{comp}"
                    st_comp = client.get_waveforms(network, station, "*", channel, starttime, endtime)
                    if st_comp:
                        stream += st_comp
                        success = True
            except Exception as e:
                log_message(f"  Failed to get 12Z components: {str(e)}")
        
        # Check if we have enough components
        if not success or len(stream) < 3:
            return idx, None, False, f"Could not get 3 components for {station}"
            
        # Merge traces by ID (in case we got multiple traces per component)
        stream.merge()
        
        # Resample all traces to target sampling rate if needed
        for tr in stream:
            if abs(tr.stats.sampling_rate - TARGET_SAMPLING_RATE) > 0.1:
                tr.interpolate(TARGET_SAMPLING_RATE)
        
        # Save to file
        stream.write(full_path, format="MSEED")
        
        # Calculate timing info for preprocessing
        timing_info = {
            'starttime': starttime.timestamp,
            'endtime': endtime.timestamp,
            'p_time': p_time.timestamp,
            's_time': s_time.timestamp
        }
        
        log_message(f"Successfully downloaded waveform for {filename}")
        return idx, filename, True, timing_info
        
    except Exception as e:
        log_message(f"Error downloading waveform: {str(e)}")
        return idx, None, False, f"Error: {str(e)}"

def download_all_waveforms(df, fixed_length_samples):
    """Download all waveforms for filtered data sequentially with better error handling"""
    fixed_length_seconds = fixed_length_samples / TARGET_SAMPLING_RATE
    log_message(f"Downloading waveforms for {len(df)} P-S pick pairs with fixed length of {fixed_length_seconds:.2f} seconds...")

    # Add columns for tracking results
    df['waveform_file'] = None
    df['download_error'] = None
    df['timing_info'] = None

    success_count = 0
    failure_count = 0
    retry_count = 0
    max_retries = 3  # For the entire batch
    
    # Create a list to track items that need retry
    retry_list = list(range(len(df)))
    
    while retry_list and retry_count < max_retries:
        if retry_count > 0:
            log_message(f"Retry attempt {retry_count}/{max_retries} for {len(retry_list)} remaining items")
        
        # Create a new list for next retry cycle
        next_retry = []
        
        for idx in tqdm(retry_list, desc=f"Download batch {retry_count+1}", unit="file"):
            row = df.iloc[idx]
            
            # Skip if already downloaded
            if not pd.isna(df.at[idx, 'waveform_file']):
                continue
                
            idx_num, filename, success, result = download_waveform(row, idx, fixed_length_seconds)

            if success and filename:
                df.at[idx, 'waveform_file'] = filename

                # Save timing info for preprocessing
                if isinstance(result, dict):
                    df.at[idx, 'timing_info'] = str(result)

                success_count += 1
                
                # Log progress periodically
                if success_count % 10 == 0:
                    log_message(f"Download progress: {success_count} successful, {failure_count} failed")
            else:
                df.at[idx, 'download_error'] = result
                next_retry.append(idx)  # Add to retry list
            
            # Brief pause to avoid overwhelming the server
            time.sleep(1)
            
        # Update retry list and increment counter
        retry_list = next_retry
        retry_count += 1
        
        # Save progress after each batch
        df.to_csv(os.path.join(metadata_dir, f"p_s_pick_metadata_with_waveforms_batch{retry_count}.csv"), index=False)
        
        # If we still have failures, wait longer before retrying
        if retry_list and retry_count < max_retries:
            wait_time = 60 * retry_count  # Progressive backoff
            log_message(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    # Final count of failures
    failure_count = len(df) - success_count
    log_message(f"Waveform download complete: {success_count} successful, {failure_count} failed")

    # Save final results
    df.to_csv(os.path.join(metadata_dir, "p_s_pick_metadata_with_waveforms.csv"), index=False)

    return df

# ==================== FUNGSI PREPROCESSING ====================

def preprocess_waveform(mseed_file, row, fixed_length_samples):
    """
    Preprocess waveform untuk format PhaseNet dengan panjang tetap
    """
    try:
        # Debug info
        log_message(f"Processing {mseed_file}")
        
        # Extract timing info jika tersedia
        timing_info = None
        if not pd.isna(row.get('timing_info')):
            import ast
            try:
                timing_info = ast.literal_eval(row['timing_info'])
                log_message(f"Found timing info: P at {timing_info['p_time']}, S at {timing_info['s_time']}")
            except Exception as e:
                log_message(f"Failed to parse timing info: {e}")

        # Correct way to read MSEED files in ObsPy
        try:
            from obspy import read
            stream = read(mseed_file)
            log_message(f"Successfully read MSEED file with {len(stream)} traces")
            # Log trace details
            for i, tr in enumerate(stream):
                log_message(f"Trace {i}: {tr.stats.network}.{tr.stats.station}.{tr.stats.channel}, " +
                          f"length: {len(tr.data)}, sampling_rate: {tr.stats.sampling_rate}")
        except Exception as e:
            log_message(f"Failed to read MSEED file: {e}")
            return None, False, f"Failed to read MSEED file: {e}"

        # Pastikan kita punya 3 komponen
        if len(stream) < 3:
            log_message(f"Warning: Less than 3 components found ({len(stream)})")
            return None, False, f"Less than 3 components: {len(stream)}"

        # Resampling ke 100 Hz
        for tr in stream:
            if abs(tr.stats.sampling_rate - TARGET_SAMPLING_RATE) > 0.1:
                tr.interpolate(TARGET_SAMPLING_RATE)
                log_message(f"Resampled trace {tr.stats.channel} to {TARGET_SAMPLING_RATE} Hz")

        # Dapatkan waktu P dan S
        p_time = UTCDateTime(row['p_pick_time'])
        s_time = UTCDateTime(row['s_pick_time'])
        log_message(f"P arrival at {p_time}, S arrival at {s_time}")

        # Trim data ke rentang yang sama untuk semua komponen
        stream_starttime = max([tr.stats.starttime for tr in stream])
        stream_endtime = min([tr.stats.endtime for tr in stream])

        if stream_endtime <= stream_starttime:
            log_message(f"Invalid time range after trim: {stream_starttime} to {stream_endtime}")
            return None, False, "Invalid time range after trim"

        stream = stream.trim(stream_starttime, stream_endtime)
        log_message(f"Trimmed stream to {stream_starttime} - {stream_endtime}")

        # Detrend dan normalisasi
        stream.detrend('demean')

        # Urutkan komponen berdasarkan nama channel
        components = [tr.stats.channel[-1] for tr in stream]
        log_message(f"Available components: {components}")
        
        # Store channel information for output
        channel_info = ""
        
        # Coba deteksi format komponen
        if set(components).intersection({'E', 'N', 'Z'}) == {'E', 'N', 'Z'}:
            # Format ENZ
            stream_sorted = Stream()
            for comp in ['E', 'N', 'Z']:
                for tr in stream:
                    if tr.stats.channel[-1] == comp:
                        stream_sorted.append(tr)
                        channel_info += tr.stats.channel + "_"
                        break
            stream = stream_sorted
            log_message("Using ENZ component set")
        elif set(components).intersection({'1', '2', 'Z'}) == {'1', '2', 'Z'}:
            # Format 12Z
            stream_sorted = Stream()
            for comp in ['1', '2', 'Z']:
                for tr in stream:
                    if tr.stats.channel[-1] == comp:
                        stream_sorted.append(tr)
                        channel_info += tr.stats.channel + "_"
                        break
            stream = stream_sorted
            log_message("Using 12Z component set")
        else:
            # Custom sorting jika tidak sesuai format standar
            from obspy import trace
            # Sort komponen yang tersedia
            sorted_comps = sorted(components)
            if len(sorted_comps) < 3:
                # Isi dengan dummy jika kurang dari 3
                while len(sorted_comps) < 3:
                    sorted_comps.append(f"X{len(sorted_comps)}")
            
            stream_sorted = Stream()
            for i, comp in enumerate(sorted_comps[:3]):  # Ambil 3 komponen pertama
                comp_found = False
                for tr in stream:
                    if tr.stats.channel[-1] == comp:
                        stream_sorted.append(tr)
                        channel_info += tr.stats.channel + "_"
                        comp_found = True
                        break
                
                if not comp_found:
                    # Buat dummy trace jika komponen tidak ada
                    log_message(f"Creating dummy trace for component {comp}")
                    template_tr = stream[0]
                    dummy_data = np.zeros_like(template_tr.data)
                    header = template_tr.stats.copy()
                    header.channel = header.channel[:-1] + comp
                    dummy_tr = trace.Trace(data=dummy_data, header=header)
                    stream_sorted.append(dummy_tr)
                    channel_info += header.channel + "_"
            
            stream = stream_sorted
            log_message(f"Using custom component set: {[tr.stats.channel[-1] for tr in stream]}")

        # Remove trailing underscore from channel_info
        channel_info = channel_info.rstrip("_")
        log_message(f"Channel info: {channel_info}")

        # Normalisasi amplitudo untuk setiap trace
        for tr in stream:
            if np.any(tr.data):  # Skip jika semua nol
                tr.data = tr.data / np.max(np.abs(tr.data))

        # Hitung indeks P dan S dalam jendela trimmed
        if p_time < stream[0].stats.starttime or p_time > stream[0].stats.endtime:
            log_message(f"P time outside data range: {p_time} not in [{stream[0].stats.starttime} - {stream[0].stats.endtime}]")
            return None, False, "P time outside data range"

        if s_time < stream[0].stats.starttime or s_time > stream[0].stats.endtime:
            # Jika S di luar rentang, gunakan timing info
            if timing_info and 'p_time' in timing_info and 's_time' in timing_info:
                p_s_interval_samples = int((timing_info['s_time'] - timing_info['p_time']) * TARGET_SAMPLING_RATE)
                p_sample = int((p_time - stream[0].stats.starttime) * TARGET_SAMPLING_RATE)
                s_sample = p_sample + p_s_interval_samples
                
                if s_sample >= len(stream[0].data):
                    s_sample = len(stream[0].data) - 1
                    log_message(f"Estimated S phase at end of trace (sample {s_sample})")
            else:
                log_message(f"S time outside data range and no timing info available")
                return None, False, "S time outside data range and no timing info available"
        else:
            p_sample = int((p_time - stream[0].stats.starttime) * TARGET_SAMPLING_RATE)
            s_sample = int((s_time - stream[0].stats.starttime) * TARGET_SAMPLING_RATE)
            log_message(f"P at sample {p_sample}, S at sample {s_sample}")

        # Buat array dengan panjang tetap - FIXED: Using (n, 3) shape instead of (n, 1, 3)
        data = np.zeros((fixed_length_samples, 3))

        # Menyesuaikan data ke panjang tetap
        actual_length = len(stream[0].data)
        log_message(f"Actual data length: {actual_length}, fixed length: {fixed_length_samples}")

        if actual_length <= fixed_length_samples:
            # Jika data aktual lebih pendek, copy semua dan biarkan sisanya nol
            for i, tr in enumerate(stream[:3]):  # Pastikan hanya 3 komponen
                data[:len(tr.data), i] = tr.data
            log_message("Data padded with zeros")
        else:
            # Strategi window untuk data yang lebih panjang
            p_s_interval_samples = s_sample - p_sample
            
            if p_s_interval_samples < fixed_length_samples and p_s_interval_samples > 0:
                # Bagi jendela di sekitar P dan S
                available_margin = fixed_length_samples - p_s_interval_samples
                pre_p_margin = min(p_sample, available_margin // 2)
                post_s_margin = min(actual_length - s_sample - 1, available_margin - pre_p_margin)
                
                start_idx = p_sample - pre_p_margin
                end_idx = s_sample + post_s_margin
                
                if end_idx > actual_length:
                    end_idx = actual_length
                    start_idx = max(0, end_idx - fixed_length_samples)
                
                log_message(f"Window from {start_idx} to {end_idx} (length: {end_idx-start_idx})")
                
                for i, tr in enumerate(stream[:3]):
                    window_length = min(fixed_length_samples, len(tr.data) - start_idx)
                    data[:window_length, i] = tr.data[start_idx:start_idx+window_length]
                
                # Update P and S indices
                p_sample = p_sample - start_idx
                s_sample = s_sample - start_idx
            else:
                # Jika P-S interval tidak valid atau terlalu besar, ambil dari awal
                for i, tr in enumerate(stream[:3]):
                    data[:fixed_length_samples, i] = tr.data[:fixed_length_samples]
                
                if s_sample >= fixed_length_samples:
                    s_sample = fixed_length_samples - 1
        
        # Pastikan indeks dalam batas valid dan P sebelum S
        p_sample = max(0, min(p_sample, fixed_length_samples - 1))
        s_sample = max(p_sample + 1, min(s_sample, fixed_length_samples - 1))
        log_message(f"Final indices - P: {p_sample}, S: {s_sample}")

        # Buat hasil untuk PhaseNet with corrected format
        result = {
            'data': data,
            'p_idx': [[p_sample]],
            's_idx': [[s_sample]],
            'station_id': stream[0].stats.station,
            't0': stream[0].stats.starttime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            'channel': channel_info  # Added channel information
        }
        
        log_message(f"Successfully preprocessed waveform")
        return result, True, "Success"

    except Exception as e:
        import traceback
        log_message(f"Exception in preprocess_waveform: {str(e)}")
        log_message(traceback.format_exc())
        return None, False, f"Error preprocessing: {str(e)}"


def preprocess_all_waveforms(df, fixed_length_samples):
    """Preprocess all downloaded waveforms dengan panjang tetap"""
    log_message(f"Preprocessing downloaded waveforms to fixed length of {fixed_length_samples} samples...")

    success_count = 0
    failure_count = 0
    
    # Add detailed log file for preprocessing errors
    preprocess_error_log = os.path.join(metadata_dir, "preprocess_errors.txt")
    with open(preprocess_error_log, 'w') as f:
        f.write("Preprocessing Error Log\n")
        f.write("======================\n\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing waveforms"):
        if pd.isna(row.get('waveform_file', None)):
            continue

        try:
            mseed_file = os.path.join(waveform_dir, row['waveform_file'])
            log_message(f"Processing {mseed_file}")

            # Preprocess waveform
            result, success, message = preprocess_waveform(mseed_file, row, fixed_length_samples)

            if success:
                # Simpan hasil sebagai file NPZ
                npz_filename = os.path.splitext(row['waveform_file'])[0] + '.npz'
                np.savez(os.path.join(npz_dir, npz_filename), **result)

                # Update df
                df.at[idx, 'npz_file'] = npz_filename
                success_count += 1

                # Plot contoh untuk validasi visual (batasi jumlah plot)
                if success_count <= 20:  # Hanya plot 20 contoh pertama
                    plot_waveform_with_picks(result, os.path.join(figure_dir, f"{npz_filename.replace('.npz', '')}.png"))
                    
                log_message(f"Successfully created NPZ file: {npz_filename}")
            else:
                df.at[idx, 'preprocess_error'] = message
                failure_count += 1
                
                # Log detailed error
                with open(preprocess_error_log, 'a') as f:
                    f.write(f"File: {mseed_file}\n")
                    f.write(f"Error: {message}\n")
                    f.write("---\n\n")

        except Exception as e:
            df.at[idx, 'preprocess_error'] = str(e)
            failure_count += 1
            
            # Log exception
            with open(preprocess_error_log, 'a') as f:
                f.write(f"File: {mseed_file}\n")
                f.write(f"Exception: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
                f.write("---\n\n")

    log_message(f"Preprocessing complete: {success_count} successful, {failure_count} failed")
    log_message(f"Detailed error log saved to {preprocess_error_log}")

    # Simpan df yang diupdate
    df.to_csv(os.path.join(metadata_dir, "p_s_pick_metadata_preprocessed.csv"), index=False)

    return df

# ==================== FUNGSI VISUALISASI ====================

def plot_waveform_with_picks(data_dict, output_file):
    """Plot waveform dengan picks P dan S untuk validasi visual"""
    try:
        data = data_dict['data']
        p_idx = data_dict['p_idx'][0][0]
        s_idx = data_dict['s_idx'][0][0]
        station = data_dict['station_id']
        t0 = data_dict['t0']

        plt.figure(figsize=(12, 8))

        # Plot komponen E, N, Z
        component_labels = ["E", "N", "Z"]
        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.plot(data[:, 0, i], 'k', linewidth=0.8)

            # Mark P dan S
            plt.axvline(x=p_idx, color='blue', linestyle='--', label='P pick')
            plt.axvline(x=s_idx, color='red', linestyle='--', label='S pick')

            plt.title(f"Component {component_labels[i]}")
            plt.legend()

            if i < 2:  # Tidak perlu xlabel kecuali di subplot terakhir
                plt.xticks([])

        # Tambah info pada plot
        plt.suptitle(f"Station: {station}, Start Time: {t0}", fontsize=12)
        plt.xlabel("Sample")
        plt.tight_layout()

        # Simpan plot
        plt.savefig(output_file, dpi=100)
        plt.close()

        return True
    except Exception as e:
        log_message(f"Error plotting waveform: {str(e)}")
        return False

# ==================== FUNGSI MEMBUAT DATA LIST UNTUK PHASENET ====================

def create_data_list():
    """Buat file data_list.csv yang dibutuhkan oleh PhaseNet"""
    log_message("Creating data list for PhaseNet...")

    try:
        # Cari semua file NPZ
        npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))

        if not npz_files:
            log_message("No NPZ files found!")
            return False

        # Buat DataFrame dengan nama file
        df = pd.DataFrame({"fname": npz_files})

        # Simpan ke CSV
        df.to_csv(data_list_csv, index=False)

        log_message(f"Data list created with {len(npz_files)} files at {data_list_csv}")
        return True

    except Exception as e:
        log_message(f"Error creating data list: {str(e)}")
        return False

# ==================== FUNGSI UTAMA ====================

def main():
    """Fungsi utama yang mengkoordinasikan seluruh proses"""
    log_message("STARTING PHASENET DATA PREPARATION PIPELINE")
    log_message("===========================================")

    # Step 1: Ambil metadata gempa dan P-S picks
    log_message("\nSTEP 1: FETCHING EARTHQUAKE METADATA")
    fetch_and_process_metadata()

    # Step 2: Gabungkan dan filter metadata
    log_message("\nSTEP 2: MERGING AND FILTERING METADATA")
    if merge_csv_files():
        filter_merged_data()

    # Step 3: Tentukan panjang tetap yang akan digunakan
    log_message("\nSTEP 3: DETERMINING FIXED LENGTH FOR ALL WAVEFORMS")
    df_filtered = pd.read_csv(filtered_csv)
    fixed_length_samples = determine_fixed_length(df_filtered)

    # Step 4: Download bentuk gelombang (waveform)
    log_message(f"\nSTEP 4: DOWNLOADING WAVEFORMS WITH FIXED LENGTH OF {fixed_length_samples} SAMPLES")
    df_with_waveforms = download_all_waveforms(df_filtered, fixed_length_samples)

    # Step 5: Preprocess waveform ke format PhaseNet
    log_message("\nSTEP 5: PREPROCESSING WAVEFORMS")
    df_preprocessed = preprocess_all_waveforms(df_with_waveforms, fixed_length_samples)

    # Step 6: Buat data list untuk PhaseNet
    log_message("\nSTEP 6: CREATING DATA LIST FOR PHASENET")
    create_data_list()

    # Statistik akhir
    npz_count = len(glob.glob(os.path.join(npz_dir, "*.npz")))

    log_message("\nPROCESS COMPLETED")
    log_message("===================")
    log_message(f"Total metadata records: {len(df_filtered)}")
    log_message(f"Total waveforms downloaded: {len(df_preprocessed[~pd.isna(df_preprocessed['waveform_file'])])}")
    log_message(f"Total NPZ files created: {npz_count}")
    log_message(f"Fixed length used: {fixed_length_samples} samples ({fixed_length_samples/TARGET_SAMPLING_RATE:.2f} seconds)")
    log_message(f"\nData is ready for PhaseNet at: {base_dir}")
    log_message(f"Use this command to run PhaseNet prediction:")
    log_message(f"python phasenet/predict.py --model=model/190703-214543 --data_list={data_list_csv} --format=numpy")

if __name__ == "__main__":
    main()
