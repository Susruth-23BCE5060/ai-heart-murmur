import os
import glob
import pandas as pd
from tqdm import tqdm

# Define file paths
CSV_PATH = "./training_data.csv"
AUDIO_DIR = "./data/raw"
OUTPUT_MANIFEST = "./dataset_manifest.csv"

def generate_manifest_from_csv():
    # 1. Load the dataset CSV metadata
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find '{CSV_PATH}'. Make sure the CSV file is in the project root.")
        return

    print(f"Loading metadata from {CSV_PATH}...")
    df_raw = pd.read_csv(CSV_PATH)
    
    # Standardize column names
    df_raw.rename(columns={
        'Patient ID': 'Subject_ID',
        'Recording locations:': 'Recording_Locations'
    }, inplace=True)

    records = []
    print(f"Matching audio files in '{AUDIO_DIR}' for {len(df_raw)} subjects...")

    # 2. Iterate through each patient and match audio/segmentation files
    for _, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
        subject_id = str(int(row['Subject_ID']))
        
        # Search for all .wav files belonging to this patient ID
        pattern = os.path.join(AUDIO_DIR, f"{subject_id}_*.wav")
        matching_wavs = glob.glob(pattern)

        # Fallback if no files matched (check for lowercase extensions)
        if not matching_wavs:
            pattern = os.path.join(AUDIO_DIR, f"{subject_id}_*.WAV")
            matching_wavs = glob.glob(pattern)

        for wav_path in matching_wavs:
            wav_file = os.path.basename(wav_path)
            base_name = os.path.splitext(wav_file)[0]
            
            # Derive corresponding .tsv file path
            tsv_file = f"{base_name}.tsv"
            tsv_path = os.path.join(AUDIO_DIR, tsv_file)
            
            # Extract auscultation location (e.g. 'AV', 'PV', 'TV', 'MV', 'Phc')
            location_part = base_name[len(subject_id) + 1:]  # strip '1234_'
            location = location_part.split('_')[0]          # handles 'MV_1' -> 'MV'
            
            # Copy patient row metadata and append file details
            rec_entry = row.to_dict()
            rec_entry.update({
                'Subject_ID': subject_id,
                'Location': location,
                'Wav_File': wav_file,
                'Wav_Path': os.path.abspath(wav_path),
                'Tsv_File': tsv_file,
                'Tsv_Path': os.path.abspath(tsv_path),
                'Sampling_Rate': 4000,
                'Wav_Exists': os.path.exists(wav_path),
                'Tsv_Exists': os.path.exists(tsv_path)
            })
            records.append(rec_entry)

    if not records:
        print(f"\nWarning: No matching .wav files found in '{AUDIO_DIR}'.")
        print("Please ensure your unzipped .wav/.tsv files are placed inside the 'AUDIO_DIR' path.")
        return

    # 3. Create DataFrame and apply cleaning filters
    df_manifest = pd.DataFrame(records)

    # Filter out missing audio or segmentation files
    df_clean = df_manifest[(df_manifest['Wav_Exists'] == True) & (df_manifest['Tsv_Exists'] == True)].copy()

    # Filter out 'Unknown' murmur classifications for clean training labels
    df_clean = df_clean[df_clean['Murmur'].isin(['Present', 'Absent'])].copy()

    # Generate binary label (1 = Present, 0 = Absent)
    df_clean['Label'] = df_clean['Murmur'].apply(lambda x: 1 if x == 'Present' else 0)

    # 4. Save manifest
    df_clean.to_csv(OUTPUT_MANIFEST, index=False)

    print(f"\nSuccessfully generated: {OUTPUT_MANIFEST}")
    print(f"Total Matched Audio Recordings: {len(df_clean)}")
    print(f"Unique Subjects: {df_clean['Subject_ID'].nunique()}")
    print(f"Murmur Present (1): {len(df_clean[df_clean['Label'] == 1])}")
    print(f"Murmur Absent  (0): {len(df_clean[df_clean['Label'] == 0])}")

if __name__ == "__main__":
    generate_manifest_from_csv()

"""import os
import glob
import pandas as pd
from tqdm import tqdm

DATA_DIR = "./data/raw"
OUTPUT_MANIFEST = "./dataset_manifest.csv"

def parse_subject_file(txt_path, data_dir):
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    if not lines:
        return []

    # Line 1: [Subject_ID, Number_of_recordings, Sampling_frequency]
    header_parts = lines[0].split()
    subject_id = header_parts[0]
    num_recordings = int(header_parts[1])
    sampling_rate = int(header_parts[2])

    # Extract Key-Value Metadata Tags (lines starting with '#')
    metadata = {
        'Subject_ID': subject_id,
        'Sampling_Rate': sampling_rate,
        'Num_Recordings': num_recordings
    }
    
    file_lines = []
    for line in lines[1:]:
        if line.startswith('#'):
            # Parse key-value tags (e.g., #Murmur: Present)
            tag_data = line[1:].split(':', 1)
            if len(tag_data) == 2:
                key = tag_data[0].strip()
                val = tag_data[1].strip()
                metadata[key] = val
        else:
            # Lines listing recording files (e.g., AV 1234_AV.hea 1234_AV.wav 1234_AV.tsv)
            file_lines.append(line)

    # Parse each auscultation location recording file entry
    records = []
    for line in file_lines:
        parts = line.split()
        if len(parts) >= 4:
            location = parts[0]
            hea_file = parts[1]
            wav_file = parts[2]
            tsv_file = parts[3]
            
            # Form absolute paths
            wav_path = os.path.join(data_dir, wav_file)
            tsv_path = os.path.join(data_dir, tsv_file)
            
            # Combine subject metadata with this specific recording's location
            rec_entry = metadata.copy()
            rec_entry.update({
                'Location': location,
                'Wav_File': wav_file,
                'Wav_Path': wav_path,
                'Tsv_File': tsv_file,
                'Tsv_Path': tsv_path,
                'Wav_Exists': os.path.exists(wav_path),
                'Tsv_Exists': os.path.exists(tsv_path)
            })
            records.append(rec_entry)
            
    return records

def generate_manifest():
    txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    
    if not txt_files:
        print(f"Error: No .txt files found in '{DATA_DIR}'. Check your directory path.")
        return

    all_records = []
    print(f"Processing {len(txt_files)} subject metadata files...")
    
    for txt_path in tqdm(txt_files):
        # Exclude aggregate training CSVs if present in directory
        if "training_data" in os.path.basename(txt_path):
            continue
        records = parse_subject_file(txt_path, DATA_DIR)
        all_records.extend(records)

    df = pd.DataFrame(all_records)

    # Clean & Filter Data
    # 1. Ensure audio and segmentation files exist
    df = df[(df['Wav_Exists'] == True) & (df['Tsv_Exists'] == True)]
    
    # 2. Filter out 'Unknown' Murmur labels for clean model training
    df_clean = df[df['Murmur'].isin(['Present', 'Absent'])].copy()
    
    # 3. Create a binary Target Label (1 for Present, 0 for Absent)
    df_clean['Label'] = df_clean['Murmur'].apply(lambda x: 1 if x == 'Present' else 0)

    # Export
    df_clean.to_csv(OUTPUT_MANIFEST, index=False)
    print(f"\nManifest successfully created: {OUTPUT_MANIFEST}")
    print(f"Total Recordings: {len(df_clean)}")
    print(f"Present (Murmur): {len(df_clean[df_clean['Label'] == 1])}")
    print(f"Absent (Normal):  {len(df_clean[df_clean['Label'] == 0])}")

if __name__ == "__main__":
    generate_manifest()"""