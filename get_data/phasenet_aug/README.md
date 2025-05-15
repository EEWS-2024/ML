# README - Pipeline Persiapan Dataset PhaseNet dengan Augmentasi Channel

## Ringkasan

Pipeline ini dirancang untuk mengumpulkan, memproses, dan augmentasi data seismik dari jaringan seismik Indonesia untuk pelatihan dan pengujian model PhaseNet. PhaseNet adalah model deep neural network yang digunakan untuk deteksi otomatis waktu kedatangan fase P dan S pada data seismik. Keunikan pipeline ini adalah kemampuannya untuk melakukan augmentasi data dengan memanfaatkan berbagai tipe channel seismik selain BH* yang standar.

## Fitur Utama

- Pengambilan metadata gempa dan fase P-S dari FDSN server (GFZ)
- Filtering data untuk memastikan kualitas dan konsistensi
- Augmentasi data dengan berbagai tipe channel (HH, BL, HL, SH, dll)
- Download dan preprocessing bentuk gelombang seismik
- Standardisasi data ke format yang dibutuhkan PhaseNet
- Pembuatan dataset terpisah untuk data original dan data hasil augmentasi
- Padding P-index untuk kompatibilitas dengan model PhaseNet

## Struktur Direktori dan File-file Penting

Hasil dari pipeline ini akan membuat struktur direktori sebagai berikut:

```
./dataset_phasenet_aug/
├── metadata/                              # Direktori metadata
│   ├── log_progress.txt                   # Log proses pipeline
│   ├── p_s_pick_metadata_*.csv            # File metadata per batch
│   ├── p_s_pick_metadata_merged.csv       # Data gabungan semua batch
│   ├── p_s_pick_metadata_filtered.csv     # Data terfilter
│   ├── p_s_pick_metadata_original.csv     # Data original (tanpa augmentasi)
│   ├── p_s_pick_metadata_augmented.csv    # Data dengan augmentasi
│   ├── p_s_pick_metadata_with_waveforms.csv   # Data setelah download waveform
│   ├── p_s_pick_metadata_with_waveforms_batch*.csv   # Checkpoint download per batch
│   ├── p_s_pick_metadata_preprocessed.csv    # Data setelah preprocessing
│   └── preprocess_errors.txt              # Log error preprocessing
├── waveform/                              # File mentah MSEED
│   └── [network].[station].[channel].[event_id].mseed   # Format file waveform
├── npz/                                   # Data bentuk gelombang dalam format NPZ
│   └── [network].[station].[channel].[event_id].npz     # Format file NPZ
├── npz_padded/                            # Data NPZ dengan padding P-index
│   └── [network].[station].[channel].[event_id].npz     # Format file NPZ padded
├── figures/                               # Visualisasi waveform dan statistik
│   ├── ps_interval_distribution.png       # Histogram distribusi interval P-S
│   └── [network].[station].[channel].[event_id].png     # Plot waveform dengan P-S pick
├── dataset_config.json                    # Konfigurasi dataset (fixed length, dll)
├── data_list.csv                          # Daftar file NPZ original
├── padded_data_list.csv                   # Daftar file NPZ dengan padding
├── padded_train_list.csv                  # Daftar file untuk training (95%)
├── padded_valid_list.csv                  # Daftar file untuk validasi (5%)
├── original_data_list.csv                 # Daftar file data original saja
├── augmented_data_list.csv                # Daftar file data augmentasi saja
├── original_train_list.csv                # Daftar file training data original
├── original_valid_list.csv                # Daftar file validasi data original
├── augmented_train_list.csv               # Daftar file training data augmentasi
└── augmented_valid_list.csv               # Daftar file validasi data augmentasi
```

## Lokasi dan Penjelasan File-file Final

### 1. File CSV Metadata (di direktori `./dataset_phasenet_aug/metadata/`)

| Nama File | Deskripsi |
|-----------|-----------|
| `p_s_pick_metadata_merged.csv` | Data gabungan dari semua batch metadata yang diambil dari server |
| `p_s_pick_metadata_filtered.csv` | Data yang sudah difilter (hanya entri dengan fase P dan S) |
| `p_s_pick_metadata_original.csv` | Data original sebelum augmentasi |
| `p_s_pick_metadata_augmented.csv` | Data lengkap setelah augmentasi (original + augmented) |
| `p_s_pick_metadata_with_waveforms.csv` | Data dengan informasi file waveform hasil download |
| `p_s_pick_metadata_preprocessed.csv` | Data dengan informasi file NPZ hasil preprocessing |

### 2. File Data Waveform (di direktori `./dataset_phasenet_aug/waveform/`)

File MSEED dengan format: `[network].[station].[channel].[event_id].mseed`

Contoh: `GE.MNAI.BH.gfz2022gtxj.mseed`

File-file ini berisi data mentah dari stasiun seismik dalam format miniSEED, yang merupakan format standar untuk pertukaran data seismik. Setiap file berisi 3 komponen (Z, N, E atau 1, 2, Z) untuk satu rekaman gempa.

### 3. File NPZ (di direktori `./dataset_phasenet_aug/npz/` dan `./dataset_phasenet_aug/npz_padded/`)

NPZ adalah format terkompresi NumPy yang berisi array dari data seismik dan informasi terkait:

- Di direktori `npz/`: Hasil preprocessing tanpa padding
- Di direktori `npz_padded/`: Hasil preprocessing dengan padding P-index minimal 3001

Setiap file NPZ berisi:
- `data`: Array 2D (time, channel) berisi waveform 3 komponen
- `p_idx`: Indeks sampel kedatangan fase P
- `s_idx`: Indeks sampel kedatangan fase S
- `station_id`: ID stasiun
- `t0`: Timestamp awal dari rekaman
- `channel`: Informasi nama channel yang digunakan
- `channel_type`: Tipe channel (BH, HH, dll)
- `is_augmented`: Flag yang menandakan data asli atau hasil augmentasi
- `original_channel`: Informasi channel asli jika data hasil augmentasi

### 4. File Data List untuk PhaseNet (di direktori root `./dataset_phasenet_aug/`)

| Nama File | Deskripsi |
|-----------|-----------|
| `data_list.csv` | Daftar file NPZ asli untuk input ke PhaseNet |
| `padded_data_list.csv` | Daftar file NPZ dengan padding untuk input ke PhaseNet |
| `padded_train_list.csv` | Subset file untuk pelatihan (95% dari total) |
| `padded_valid_list.csv` | Subset file untuk validasi (5% dari total) |
| `original_data_list.csv` | Subset file yang hanya berisi data original (bukan augmentasi) |
| `augmented_data_list.csv` | Subset file yang hanya berisi data hasil augmentasi |
| `original_train_list.csv` | File training dari data original |
| `original_valid_list.csv` | File validasi dari data original |
| `augmented_train_list.csv` | File training dari data augmentasi |
| `augmented_valid_list.csv` | File validasi dari data augmentasi |

### 5. File Konfigurasi (di direktori root `./dataset_phasenet_aug/`)

`dataset_config.json` berisi parameter penting untuk dataset:
- `fixed_length_samples`: Panjang tetap dalam sampel
- `fixed_length_seconds`: Panjang tetap dalam detik
- `sampling_rate`: Frekuensi sampling
- `pre_p_time`: Waktu sebelum fase P (detik)
- `post_s_time`: Waktu setelah fase S (detik)

### 6. File Visualisasi (di direktori `./dataset_phasenet_aug/figures/`)

- `ps_interval_distribution.png`: Histogram distribusi interval antara fase P dan S
- File-file PNG per waveform: Visualisasi bentuk gelombang dengan fase P dan S yang ditandai

## Persyaratan

- Python 3.6+
- ObsPy
- Pandas, NumPy, Matplotlib
- tqdm
- scikit-learn
- Akses internet untuk mengunduh data dari server FDSN

## Cara Penggunaan

### 1. Persiapan

Pastikan semua dependensi terinstal dan file `stasiun_channel_lengkap.csv` yang berisi informasi channel stasiun tersedia di direktori yang sama dengan script.

### 2. Menjalankan Pipeline

Pipeline lengkap dapat dijalankan dengan perintah:

```bash
python phasenet_multi_channel_dataset_preparation_aug.py
```

### 3. Langkah-langkah Proses

Pipeline ini terdiri dari beberapa tahap berurutan:

1. **Pengambilan Metadata Gempa** - Mengambil data gempa dan fase P-S dari server GFZ
2. **Penggabungan dan Filtering Metadata** - Memastikan hanya data berkualitas baik yang digunakan
3. **Augmentasi Metadata** - Menambahkan data dengan tipe channel alternatif dari stasiun yang sama
4. **Penentuan Panjang Tetap** - Menghitung panjang yang optimal untuk semua bentuk gelombang
5. **Download Waveform** - Mengunduh data bentuk gelombang seismik dari server
6. **Preprocessing Waveform** - Normalisasi, resampling, dan standardisasi data
7. **Pembuatan Data List** - Mempersiapkan file daftar untuk PhaseNet
8. **Padding P-index** - Memastikan kompatibilitas dengan model PhaseNet
9. **Pembuatan Training/Validation Set** - Membagi data untuk pelatihan dan validasi

### 4. Cara Mengakses dan Menggunakan File-file Final

- **Untuk melihat metadata**: Buka file CSV di `./dataset_phasenet_aug/metadata/` menggunakan program spreadsheet atau pandas
- **Untuk melihat bentuk gelombang mentah**: Gunakan ObsPy untuk membaca file MSEED di direktori `waveform/`
- **Untuk mengakses data preprocessed**: Gunakan numpy untuk membaca file NPZ di direktori `npz_padded/`

Contoh membaca file NPZ:
```python
import numpy as np
data = np.load('./dataset_phasenet_aug/npz_padded/GE.MNAI.BH.gfz2022gtxj.npz')
waveform = data['data']
p_idx = data['p_idx']
s_idx = data['s_idx']
```

### 5. Menggunakan Data dengan PhaseNet

Untuk melatih model PhaseNet menggunakan dataset yang dihasilkan:

```bash
python phasenet/train.py --model=model/your_model_name --data_dir=./dataset_phasenet_aug/npz_padded --data_list=./dataset_phasenet_aug/padded_train_list.csv --data_list_val=./dataset_phasenet_aug/padded_valid_list.csv
```

Untuk prediksi menggunakan model pretrained:

```bash
python phasenet/predict.py --model=model/190703-214543 --data_list=./dataset_phasenet_aug/padded_data_list.csv --format=numpy
```

Untuk eksperimen hanya dengan data original:
```bash
python phasenet/train.py --model=model/original_only --data_dir=./dataset_phasenet_aug/npz_padded --data_list=./dataset_phasenet_aug/original_train_list.csv --data_list_val=./dataset_phasenet_aug/original_valid_list.csv
```

Untuk eksperimen hanya dengan data augmentasi:
```bash
python phasenet/train.py --model=model/augmented_only --data_dir=./dataset_phasenet_aug/npz_padded --data_list=./dataset_phasenet_aug/augmented_train_list.csv --data_list_val=./dataset_phasenet_aug/augmented_valid_list.csv
```

## Fitur Augmentasi Data

Pipeline ini meningkatkan jumlah data dengan memanfaatkan channel selain BH* yang standar. Proses augmentasi bekerja sebagai berikut:

1. Mengidentifikasi stasiun dan waktu kejadian gempa dari data BH* yang memiliki fase P dan S terdeteksi
2. Mencari tipe channel lain (seperti HH, BL, HL, SH) yang tersedia di stasiun yang sama
3. Menggunakan waktu fase P dan S dari channel BH* sebagai referensi untuk channel alternatif
4. Mengunduh dan memproses data dari channel alternatif tersebut

Pendekatan ini meningkatkan jumlah data pelatihan secara signifikan tanpa perlu fase P dan S baru, karena memanfaatkan fase yang sudah ada pada metadata original.

## Fitur Ketahanan

Pipeline dirancang dengan ketahanan tinggi:

- Retry otomatis saat download gagal (hingga 3 kali dengan peningkatan waktu tunggu)
- Checkpoint reguler untuk memudahkan pemulihan jika proses terganggu
- Log detail untuk debugging
- Penanganan kesalahan yang komprehensif

## Visualisasi 

Pipeline menghasilkan visualisasi untuk:
- Distribusi interval P-S (lihat `figures/ps_interval_distribution.png`)
- Contoh bentuk gelombang dengan fase P dan S yang ditandai
- Statistik hasil augmentasi data

## Catatan

- Proses ini mungkin memerlukan waktu yang cukup lama tergantung pada ukuran dataset dan kecepatan koneksi
- Hasil augmentasi sangat tergantung pada ketersediaan channel tambahan di stasiun yang dipilih
- Kualitas hasil augmentasi bergantung pada kualitas labelisasi fase P dan S original

---

Dibuat oleh: Syahrul
Tanggal: 15 Mei 2025