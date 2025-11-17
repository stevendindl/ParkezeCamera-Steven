import csv
from pathlib import Path
from collections import defaultdict
import shutil

def load_slot_bounding_boxes(csv_dir):
    """Load bounding boxes for each camera's parking slots"""
    slot_boxes = {}  # {camera_id: {slot_id: (x, y, w, h)}}
    
    csv_files = list(Path(csv_dir).glob('camera*.csv'))
    print(f"Found {len(csv_files)} camera CSV files")
    
    for csv_file in csv_files:
        # Extract camera number from filename (e.g., camera1.csv -> 1)
        camera_id = csv_file.stem.replace('camera', '')
        slot_boxes[camera_id] = {}
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                slot_id = row['SlotId']
                # Coordinates are in 2592x1944, need to scale to 1000x750
                x = int(row['X']) * 1000 / 2592
                y = int(row['Y']) * 750 / 1944
                w = int(row['W']) * 1000 / 2592
                h = int(row['H']) * 750 / 1944
                slot_boxes[camera_id][slot_id] = (x, y, w, h)
        
        print(f"  Camera {camera_id}: {len(slot_boxes[camera_id])} slots")
    
    return slot_boxes

def load_metadata(metadata_csv):
    """Load CNRPark+EXT.csv metadata with occupancy info"""
    # Structure: {image_filename: {slot_id: occupancy}}
    metadata = defaultdict(dict)
    
    print(f"\nLoading metadata from {metadata_csv}")
    
    with open(metadata_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_url = row['image_url']
            
            # Skip CNRPark entries (we only want CNR-EXT)
            if 'CNRPark/' in image_url:
                continue
            
            # Extract filename from image_url
            # Example: PATCHES/SUNNY/2015-11-12/camera1/S_2015-11-12_08.47_C01_191.jpg
            filename_parts = image_url.split('/')[-1]  # Get last part
            # Extract: date_time from S_2015-11-12_08.47_C01_191.jpg
            parts = filename_parts.split('_')
            if len(parts) >= 4:
                date = parts[1]  # 2015-11-12
                time = parts[2]  # 08.47
                slot_id = parts[4].replace('.jpg', '')  # 191
                
                # Reconstruct the full image filename
                full_filename = f"{date}_{time.replace('.', '')}.jpg"
                
                occupancy = int(row['occupancy'])  # 0=free, 1=busy
                metadata[full_filename][slot_id] = occupancy
    
    print(f"Loaded metadata for {len(metadata)} images")
    return metadata

def convert_bbox_to_yolo(x, y, w, h, img_width=1000, img_height=750):
    """Convert bounding box to YOLO format (normalized)"""
    # Convert from top-left corner to center
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    
    return x_center, y_center, width, height

def process_txt_file(txt_file, slot_boxes, metadata, images_dir, output_dir):
    """Process a single .txt split file"""
    print(f"\nProcessing {txt_file.name}")
    
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    processed = 0
    skipped = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Parse line: "RAINY/2016-02-12/camera1/2016-02-12_0910.jpg 1"
        parts = line.split()
        image_path = parts[0]
        
        # Extract components
        path_parts = image_path.split('/')
        weather = path_parts[0]
        date = path_parts[1]
        camera_folder = path_parts[2]
        filename_dummy = path_parts[3]

        filename_dummy = filename_dummy[2:].replace(".", "").split("_C0")
        image_name = filename_dummy[0] + ".jpg"
        camera_id, slot_id = filename_dummy[1].replace("jpg", "").split("_")
        
        # Full path to source image
        src_image = images_dir / weather / date / camera_folder / image_name
        
        if not src_image.exists():
            skipped += 1
            continue
        
        # Get slot boxes for this camera
        if camera_id not in slot_boxes:
            print(f"  Warning: No bounding boxes for camera {camera_id}")
            skipped += 1
            continue
        
        camera_slots = slot_boxes[camera_id]
        
        # Get occupancy metadata for this image
        image_metadata = metadata.get(image_name, {})
        
        if not image_metadata:
            # If no metadata, we can't determine occupancy
            skipped += 1
            continue
        
        # Create YOLO label file
        label_lines = []
        for slot_id, (x, y, w, h) in camera_slots.items():
            if slot_id not in image_metadata:
                continue  # Skip slots without occupancy info for this image
            
            occupancy = image_metadata[slot_id]  # 0=free, 1=busy
            x_center, y_center, width, height = convert_bbox_to_yolo(x, y, w, h)
            
            # YOLO format: class x_center y_center width height
            label_lines.append(f"{occupancy} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        if label_lines:
            # Create output paths
            split_name = txt_file.stem  # train, val, or test
            out_images = output_dir / split_name / 'images'
            out_labels = output_dir / split_name / 'labels'
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)
            
            # Copy image
            dst_image = out_images / image_name
            shutil.copy2(src_image, dst_image)
            
            # Write label file
            label_file = out_labels / image_name.replace('.jpg', '.txt')
            with open(label_file, 'w') as f:
                f.writelines(label_lines)
            
            processed += 1
        else:
            skipped += 1
    
    print(f"  Processed: {processed}, Skipped: {skipped}")

def main():
    # Paths
    cnr_dir = Path("../data/CNR-EXT_FULL_IMAGE_1000x750/")
    images_dir = cnr_dir / "images"
    txts_dir = cnr_dir / "txt"
    csvs_dir = cnr_dir / "csv"
    metadata_csv = cnr_dir / "CNRPark+EXT.csv"
    
    output_dir = Path("../data/CNRPark-EXT-YOLO")
    
    print("="*80)
    print("CNRPark-EXT to YOLO Format Converter")
    print("="*80)
    
    # Load bounding boxes for each camera
    slot_boxes = load_slot_bounding_boxes(csvs_dir)
    
    # Load metadata with occupancy information
    metadata = load_metadata(metadata_csv)
    
    # Process each split file
    txt_files = list(txts_dir.glob('*.txt'))
    print(f"\nFound {len(txt_files)} split files")
    
    for txt_file in txt_files:
        process_txt_file(txt_file, slot_boxes, metadata, images_dir, output_dir)
    
    # Create data.yaml
    yaml_content = f"""train: ../train/images
val: ../valid/images
test: ../test/images

nc: 2
names: ['space-empty', 'space-occupied']
"""
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nConversion complete! Output saved to: {output_dir}")
    print(f"data.yaml created at: {yaml_path}")

if __name__ == '__main__':
    main()