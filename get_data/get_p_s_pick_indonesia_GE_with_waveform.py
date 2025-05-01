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

# ==================== KONFIGURASI ====================

# Inisialisasi FDSN client (GFZ)
client = Client("GFZ")

# Wilayah Indonesia
minlat, maxlat = -11.0, 6.0
minlon, maxlon = 95.0, 141.0
min_magnitude = 2.0

# Rentang tahun
start_year = 2007
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
base_dir = "./dataset_phasenet"
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
        # Ekstrak informasi yang diperlukan
        network = row['p_pick_network']
        station = row['p_pick_station']
        p_time = UTCDateTime(row['p_pick_time'])
        s_time = UTCDateTime(row['s_pick_time'])

        # Hitung interval P-S dalam detik
        p_s_interval = s_time - p_time

        # Format nama file
        event_id = row.get('event_id', f"EV_{p_time.strftime('%Y%m%d_%H%M%S')}")
        filename = f"{network}.{station}.{event_id}.mseed"
        full_path = os.path.join(waveform_dir, filename)

        # Skip jika file sudah ada
        if os.path.exists(full_path):
            return idx, filename, True, "File already exists"

        # Kalkulasi berapa banyak waktu kita miliki untuk pre-P dan post-S
        available_margin = fixed_length_seconds - p_s_interval

        if available_margin <= 0:
            # Jika interval P-S lebih besar dari panjang tetap, fokuskan pada P dan sebanyak mungkin setelahnya
            starttime = p_time
            endtime = p_time + fixed_length_seconds
            log_message(f"Warning: P-S interval ({p_s_interval:.2f}s) exceeds fixed length ({fixed_length_seconds:.2f}s) for {filename}. S phase may be cut off.")
        else:
            # Bagi margin yang tersedia antara pre-P dan post-S secara proporsional
            # Standarnya: 30% sebelum P, 70% setelah S
            pre_p_margin = min(PRE_P_TIME, available_margin * 0.3)
            post_s_margin = min(POST_S_TIME, available_margin * 0.7)

            # Jika masih ada margin tersisa, alokasikan ke yang lain
            remaining_margin = available_margin - pre_p_margin - post_s_margin
            if remaining_margin > 0:
                if pre_p_margin < PRE_P_TIME:
                    additional_pre_p = min(PRE_P_TIME - pre_p_margin, remaining_margin)
                    pre_p_margin += additional_pre_p
                    remaining_margin -= additional_pre_p

                if remaining_margin > 0 and post_s_margin < POST_S_TIME:
                    post_s_margin += min(POST_S_TIME - post_s_margin, remaining_margin)

            # Tetapkan waktu awal dan akhir
            starttime = p_time - pre_p_margin
            endtime = s_time + post_s_margin

        # Pastikan panjang data tidak melebihi fixed_length_seconds
        if (endtime - starttime) > fixed_length_seconds:
            endtime = starttime + fixed_length_seconds

        # Request waveform data
        try:
            stream = client.get_waveforms(network, station, "*", "*[ENZ]", starttime, endtime)
        except Exception as e:
            # Jika ENZ tidak berhasil, coba alternatif 12Z
            try:
                stream = client.get_waveforms(network, station, "*", "*[12Z]", starttime, endtime)
            except Exception as e2:
                return idx, None, False, f"Error getting waveforms: {str(e)}, {str(e2)}"

        # Periksa apakah kita punya semua komponen yang diperlukan
        components = [tr.stats.channel[-1] for tr in stream]

        if not all(c in components for c in ["E", "N", "Z"]) and not all(c in components for c in ["1", "2", "Z"]):
            return idx, None, False, f"Missing components, found only: {components}"

        # Simpan ke file
        stream.write(full_path, format="MSEED")

        # Catat timing info untuk preprocessing
        timing_info = {
            'starttime': starttime.timestamp,
            'endtime': endtime.timestamp,
            'p_time': p_time.timestamp,
            's_time': s_time.timestamp
        }

        return idx, filename, True, timing_info

    except Exception as e:
        return idx, None, False, f"Error: {str(e)}"

def download_all_waveforms(df, fixed_length_samples):
    """Download all waveforms for filtered data sequentially"""
    fixed_length_seconds = fixed_length_samples / TARGET_SAMPLING_RATE
    log_message(f"Downloading waveforms for {len(df)} P-S pick pairs with fixed length of {fixed_length_seconds:.2f} seconds...")

    # Download secara sekuensial
    success_count = 0
    failure_count = 0

    # Tambah kolom untuk menyimpan timing info
    df['timing_info'] = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading waveforms"):
        idx_num, filename, success, result = download_waveform(row, idx, fixed_length_seconds)

        if success and filename:
            df.at[idx, 'waveform_file'] = filename

            # Jika result adalah timing info, simpan untuk preprocessing
            if isinstance(result, dict):
                df.at[idx, 'timing_info'] = str(result)

            success_count += 1

            # Log progress setiap 10 unduhan berhasil
            if success_count % 10 == 0:
                log_message(f"Download progress: {success_count} successful downloads so far")
        else:
            df.at[idx, 'download_error'] = result
            failure_count += 1

        # Tambahkan delay untuk menghindari pembatasan server
        time.sleep(2)  # Delay 2 detik antar request

    log_message(f"Waveform download complete: {success_count} successful, {failure_count} failed")

    # Simpan df yang diupdate
    df.to_csv(os.path.join(metadata_dir, "p_s_pick_metadata_with_waveforms.csv"), index=False)

    return df

# ==================== FUNGSI PREPROCESSING ====================

def preprocess_waveform(mseed_file, row, fixed_length_samples):
    """
    Preprocess waveform untuk format PhaseNet dengan panjang tetap

    Args:
        mseed_file: Path ke file MiniSEED
        row: Row dari dataframe dengan timing info
        fixed_length_samples: Panjang tetap dalam sampel

    Returns:
        Dictionary data preprocessed untuk PhaseNet
    """
    try:
        # Extract timing info jika tersedia
        timing_info = None
        if not pd.isna(row.get('timing_info')):
            import ast
            try:
                timing_info = ast.literal_eval(row['timing_info'])
            except:
                pass

        # Baca file MiniSEED
        stream = Stream()
        stream = stream.read(mseed_file)

        # Pastikan kita punya 3 komponen
        if len(stream) < 3:
            return None, False, "Less than 3 components"

        # Resampling ke 100 Hz
        for tr in stream:
            if abs(tr.stats.sampling_rate - TARGET_SAMPLING_RATE) > 0.1:
                tr.interpolate(TARGET_SAMPLING_RATE)

        # Dapatkan waktu P dan S
        p_time = UTCDateTime(row['p_pick_time'])
        s_time = UTCDateTime(row['s_pick_time'])

        # Trim data ke rentang yang sama untuk semua komponen
        stream_starttime = max([tr.stats.starttime for tr in stream])
        stream_endtime = min([tr.stats.endtime for tr in stream])

        if stream_endtime <= stream_starttime:
            return None, False, "Invalid time range after trim"

        stream = stream.trim(stream_starttime, stream_endtime)

        # Detrend dan normalisasi
        stream.detrend('demean')

        # Urutkan komponen
        components = sorted([tr.stats.channel[-1] for tr in stream])

        # Penanganan untuk berbagai format komponen
        if set(components) == {'1', '2', 'Z'}:
            stream_sorted = Stream()
            for comp in ['1', '2', 'Z']:
                for tr in stream:
                    if tr.stats.channel[-1] == comp:
                        stream_sorted.append(tr)
                        break
            stream = stream_sorted
        elif set(components) == {'E', 'N', 'Z'}:
            stream_sorted = Stream()
            for comp in ['E', 'N', 'Z']:
                for tr in stream:
                    if tr.stats.channel[-1] == comp:
                        stream_sorted.append(tr)
                        break
            stream = stream_sorted
        else:
            return None, False, f"Cannot sort components: {components}"

        # Hitung indeks P dan S dalam jendela trimmed
        if p_time < stream[0].stats.starttime or p_time > stream[0].stats.endtime:
            return None, False, "P time outside data range"

        if s_time < stream[0].stats.starttime or s_time > stream[0].stats.endtime:
            # Jika S di luar rentang, gunakan timing info dari download jika tersedia
            if timing_info and timing_info['s_time'] > timing_info['p_time']:
                # Kita tahu S seharusnya ada namun terpotong
                # Hitung indeks S berdasarkan jarak P-S yang diketahui
                p_s_interval_samples = int((timing_info['s_time'] - timing_info['p_time']) * TARGET_SAMPLING_RATE)
                p_sample = int((p_time - stream[0].stats.starttime) * TARGET_SAMPLING_RATE)
                s_sample = p_sample + p_s_interval_samples

                # Jika masih di luar rentang, gunakan indeks terakhir
                if s_sample >= len(stream[0].data):
                    s_sample = len(stream[0].data) - 1
                    log_message(f"Warning: S-phase outside data range for {row.get('waveform_file')}. Using estimated S index.")
            else:
                return None, False, "S time outside data range and no timing info available"
        else:
            # Hitung indeks P dan S normal
            p_sample = int((p_time - stream[0].stats.starttime) * TARGET_SAMPLING_RATE)
            s_sample = int((s_time - stream[0].stats.starttime) * TARGET_SAMPLING_RATE)

        # Buat array dengan panjang tetap
        data = np.zeros((fixed_length_samples, 1, 3))

        # Tentukan strategi untuk menyesuaikan data ke panjang tetap
        actual_length = len(stream[0].data)

        if actual_length <= fixed_length_samples:
            # Jika data aktual lebih pendek, copy semua dan biarkan sisanya nol
            for i, tr in enumerate(stream):
                data[:len(tr.data), 0, i] = tr.data
        else:
            # Jika data lebih panjang, kita perlu memotongnya
            # Strategi: Pastikan P dan S tetap dalam rentang data
            # Jika jarak P-S lebih kecil dari fixed_length_samples, bisa centang di sekitar keduanya
            # Jika tidak, prioritaskan P dan potong dari awal

            p_s_interval_samples = s_sample - p_sample

            if p_s_interval_samples < fixed_length_samples:
                # Bisa mengambil kedua fase
                # Alokasikan margin sebelum P dan setelah S secara proporsional
                available_margin = fixed_length_samples - p_s_interval_samples
                pre_p_margin = min(p_sample, available_margin // 2)
                post_s_margin = min(actual_length - s_sample - 1, available_margin - pre_p_margin)

                # Jika masih ada margin tersisa, tambahkan ke pre_p
                if pre_p_margin + post_s_margin < available_margin:
                    additional_pre_p = min(p_sample, available_margin - pre_p_margin - post_s_margin)
                    pre_p_margin += additional_pre_p

                # Tentukan indeks awal dan akhir
                start_idx = p_sample - pre_p_margin
                end_idx = s_sample + post_s_margin

                # Pastikan rentang tidak melebihi data
                if end_idx > actual_length:
                    end_idx = actual_length
                    start_idx = max(0, end_idx - fixed_length_samples)

                # Copy data dan update indeks P dan S
                for i, tr in enumerate(stream):
                    data_length = min(fixed_length_samples, len(tr.data) - start_idx)
                    data[:data_length, 0, i] = tr.data[start_idx:start_idx + data_length]

                # Update indeks P dan S relatif terhadap jendela baru
                p_sample = p_sample - start_idx
                s_sample = s_sample - start_idx
            else:
                # Jarak P-S terlalu besar, prioritaskan P dan potong dari awal
                for i, tr in enumerate(stream):
                    data[:fixed_length_samples, 0, i] = tr.data[:fixed_length_samples]

                # S mungkin terpotong, periksa
                if s_sample >= fixed_length_samples:
                    log_message(f"Warning: S-phase at sample {s_sample} beyond fixed length {fixed_length_samples} for {row.get('waveform_file')}. Using estimated S index.")
                    s_sample = fixed_length_samples - 1  # Gunakan indeks terakhir sebagai estimasi

        # Pastikan indeks dalam rentang yang valid
        p_sample = max(0, min(p_sample, fixed_length_samples - 1))
        s_sample = max(0, min(s_sample, fixed_length_samples - 1))

        # Pastikan P selalu sebelum S
        if p_sample >= s_sample:
            s_sample = min(p_sample + 1, fixed_length_samples - 1)

        return {
            'data': data,
            'p_idx': [[p_sample]],
            's_idx': [[s_sample]],
            'station_id': stream[0].stats.station,
            't0': stream[0].stats.starttime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        }, True, "Success"

    except Exception as e:
        return None, False, f"Error preprocessing: {str(e)}"

def preprocess_all_waveforms(df, fixed_length_samples):
    """Preprocess all downloaded waveforms dengan panjang tetap"""
    log_message(f"Preprocessing downloaded waveforms to fixed length of {fixed_length_samples} samples...")

    success_count = 0
    failure_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing waveforms"):
        if pd.isna(row.get('waveform_file', None)):
            continue

        try:
            mseed_file = os.path.join(waveform_dir, row['waveform_file'])

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
            else:
                df.at[idx, 'preprocess_error'] = message
                failure_count += 1

        except Exception as e:
            df.at[idx, 'preprocess_error'] = str(e)
            failure_count += 1

    log_message(f"Preprocessing complete: {success_count} successful, {failure_count} failed")

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