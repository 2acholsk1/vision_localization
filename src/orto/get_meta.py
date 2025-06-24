from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def convert_to_degrees(value):
    d, m, s = [float(x) for x in value]
    return d + m / 60 + s / 3600

def get_gps_position(jpg_path):
    img = Image.open(jpg_path)
    exif_data = img._getexif()

    if not exif_data:
        return None

    gps_data = exif_data.get(34853)
    if not gps_data:
        return None

    gps_info = {GPSTAGS.get(k, k): v for k, v in gps_data.items()}

    try:
        lat = convert_to_degrees(gps_info["GPSLatitude"])
        lon = convert_to_degrees(gps_info["GPSLongitude"])

        if gps_info.get("GPSLatitudeRef") != "N":
            lat = -lat
        if gps_info.get("GPSLongitudeRef") != "E":
            lon = -lon

        print(f"POS GPS: {lat:.6f}, {lon:.6f}")
        return lat, lon

    except KeyError as e:
        return None


get_gps_position("data/DJI_20250523143646_0009.JPG")
