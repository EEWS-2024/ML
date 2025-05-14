import pandas as pd
from obspy.clients.fdsn import Client
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')  # Mute warnings

def check_station_channels():
    """
    Check which stations have complete E, N, Z components for different channel types
    """
    print("Checking station channels using GFZ FDSN client...")

    # Inisialisasi FDSN client (GFZ)
    client = Client("GFZ")

    # Daftar 21 stasiun yang ingin dicek
    valid_stations = {
        "BBJI", "BKB", "BKNI", "BNDI", "FAKI", "GENI", "GSI", "JAGI", "LHMI", "LUWI", "MMRI",
        "MNAI", "PLAI", "PMBI", "SANI", "SAUI", "SMRI", "SOEI", "TNTI", "TOLI2", "UGM"
    }

    # Daftar waktu referensi untuk dicoba (untuk memastikan data tersedia)
    reference_years = [2020, 2018, 2015]

    # List untuk menyimpan seluruh informasi
    all_channels_info = []

    # Proses setiap stasiun
    for station in tqdm(valid_stations, desc="Processing stations"):
        channels_info = []

        # Coba beberapa tahun referensi
        for year in reference_years:
            if channels_info:  # Jika sudah dapat data, tidak perlu coba tahun lain
                break

            try:
                print(f"Checking {station} for {year}...")

                # Tentukan rentang waktu (1 bulan)
                start_date = datetime(year, 1, 1)
                end_date = datetime(year, 1, 31)

                # Ambil inventori stasiun
                inventory = client.get_stations(
                    network="*",
                    station=station,
                    level="channel",
                    starttime=start_date,
                    endtime=end_date
                )

                # Proses informasi channel
                for network in inventory:
                    for station_info in network:
                        for channel in station_info:
                            ch_code = channel.code
                            ch_type = ch_code[:2]  # Band + instrument code
                            ch_orientation = ch_code[-1]  # Orientation code

                            channels_info.append({
                                'network': network.code,
                                'station': station_info.code,
                                'channel': ch_code,
                                'channel_type': ch_type,
                                'orientation': ch_orientation,
                                'sample_rate': float(channel.sample_rate),
                                'reference_year': year
                            })

            except Exception as e:
                print(f"Error for {station} ({year}): {str(e)}")
                time.sleep(2)  # Tunggu sebelum mencoba lagi

        # Tambahkan ke daftar utama
        if channels_info:
            all_channels_info.extend(channels_info)
            print(f"Found {len(channels_info)} channels for {station}")
        else:
            print(f"No channels found for {station}")

        # Jeda untuk menghindari rate limiting
        time.sleep(2)

    # Konversi ke DataFrame untuk analisis
    df = pd.DataFrame(all_channels_info)

    if len(df) == 0:
        print("No channel data found!")
        return

    # Simpan semua informasi channel
    df.to_csv('all_station_channels.csv', index=False)
    print(f"Saved channel information ({len(df)} records) to all_station_channels.csv")

    # Analisis komponen lengkap
    complete_channels = []

    # Group berdasarkan stasiun dan tipe channel
    for (station, ch_type), group in df.groupby(['station', 'channel_type']):
        # Skip BH channels (sudah diketahui lengkap)
        if ch_type == 'BH':
            continue

        # Cek apakah ada ketiga komponen (E, N, Z)
        orientations = set(group['orientation'])
        has_e = 'E' in orientations
        has_n = 'N' in orientations
        has_z = 'Z' in orientations

        # Jika memiliki ketiga komponen
        if has_e and has_n and has_z:
            # Dapatkan sample rate
            sample_rates = group['sample_rate'].unique()

            complete_channels.append({
                'station': station,
                'channel_type': ch_type,
                'has_E': has_e,
                'has_N': has_n,
                'has_Z': has_z,
                'sample_rates': ', '.join([f"{rate:.1f}" for rate in sample_rates])
            })

    # Konversi ke DataFrame dan sort
    complete_df = pd.DataFrame(complete_channels)
    if len(complete_df) > 0:
        complete_df = complete_df.sort_values(['station', 'channel_type'])

        # Simpan ke CSV
        complete_df.to_csv('stasiun_channel_lengkap.csv', index=False)

        # Tampilkan hasil
        print("\nStasiun dengan komponen lengkap (E, N, Z) selain BH:")
        print(complete_df[['station', 'channel_type', 'sample_rates']])

        # Stasiun yang tidak memiliki komponen lengkap
        missing_stations = set(valid_stations) - set(complete_df['station'])
        if missing_stations:
            print("\nStasiun yang tidak memiliki komponen lengkap selain BH:")
            print(sorted(list(missing_stations)))
    else:
        print("\nTidak ada stasiun dengan komponen lengkap selain BH!")

if __name__ == "__main__":
    check_station_channels()