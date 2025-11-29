import os
import soundfile as sf

working_dir = r"G:\T1\MLADDC_test_flac"

for root, dirs, files in os.walk(working_dir):
    for file in files:
        if file.lower().endswith('.wav'):
            wav_path = os.path.join(root, file)
            flac_path = wav_path.replace('.wav', '.flac')

            print("Converting:", wav_path)

            # Read WAV
            audio, sr = sf.read(wav_path)

            # Write FLAC
            sf.write(flac_path, audio, sr, format='FLAC')

print("DONE — everything converted safely!")
