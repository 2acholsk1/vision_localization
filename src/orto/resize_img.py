import os
import csv
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pathlib import Path

def convert_to_degrees(value):
    d, m, s = [float(x) for x in value]
    return d + m / 60 + s / 3600

def extract_gps_from_exif(exif_data):
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

        return lat, lon
    except KeyError:
        return None

def resize_images_and_extract_gps(
    input_dir, 
    output_dir, 
    csv_path,
    target_size=(512, 512)
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.glob("*.JPG")) + list(input_dir.glob("*.jpeg"))

    print(f"Found {len(image_files)}")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "latitude", "longitude"])

        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    exif_data = img._getexif()
                    gps = extract_gps_from_exif(exif_data) if exif_data else None

                    img = img.convert("RGB")
                    img = img.resize(target_size, Image.LANCZOS)

                    output_path = output_dir / img_path.name
                    img.save(output_path, quality=95)

                    if gps:
                        writer.writerow([img_path.name, gps[0], gps[1]])
                        print(f"✔️ {img_path.name}: zapisano + GPS {gps[0]:.6f}, {gps[1]:.6f}")
                    else:
                        print(f"⚠️ {img_path.name}: brak GPS")

            except Exception as e:
                print(f"ERR {img_path.name}: {e}")

    print(f"\nsaved to {csv_path}")

resize_images_and_extract_gps(
    input_dir="data/EPG_photos",
    output_dir="data/EPG_photos_resized_proper",
    csv_path="data/EPG_photos_gps.csv",
    target_size=(512, 512)
)
