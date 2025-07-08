import os

import pandas as pd
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS


def convert_to_degrees(value):
    """Accepts either tuples or IFDRational values."""
    try:
        if isinstance(value[0], tuple):  # format ((num, den), ...)
            d = value[0][0] / value[0][1]
            m = value[1][0] / value[1][1]
            s = value[2][0] / value[2][1]
        else:  # format (IFDRational, ...)
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    except Exception as e:
        print(f"❌ Błąd konwersji GPS: {e}")
        return None


def get_image_metadata(filepath):
    try:
        image = Image.open(filepath)
        exif_data = image._getexif()
        if not exif_data:
            return None, None, None

        # Get date
        date_taken = exif_data.get(36867)  # DateTimeOriginal

        # Get GPS
        gps_info_raw = exif_data.get(34853)  # GPSInfo
        lat = lon = None

        if gps_info_raw:
            gps_info = {GPSTAGS.get(k, k): v for k, v in gps_info_raw.items()}
            try:
                lat = convert_to_degrees(gps_info['GPSLatitude'])
                if gps_info.get('GPSLatitudeRef') != 'N':
                    lat = -lat
                lon = convert_to_degrees(gps_info['GPSLongitude'])
                if gps_info.get('GPSLongitudeRef') != 'E':
                    lon = -lon
            except KeyError:
                lat = lon = None

        return date_taken, lat, lon

    except Exception as e:
        print(f"Błąd przy pliku {filepath}: {e}")
        return None, None, None

def gather_metadata(folder_path):
    data = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(folder_path, filename)
            date_taken, lat, lon = get_image_metadata(filepath)
            data.append({
                'filename': filename,
                'date_taken': date_taken,
                'latitude': lat,
                'longitude': lon
            })

    df = pd.DataFrame(data)
    df = df.sort_values(by='date_taken', na_position='last')
    df.to_csv('metadata_output_with_gps.csv', index=False)
    print("✅ Zapisano do metadata_output_with_gps.csv")

# 🔧 Ścieżka do folderu
folder_path = 'data/EPG_photos'  # <--- podaj swój folder!
gather_metadata(folder_path)
