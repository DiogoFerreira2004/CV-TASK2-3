import json
import os

def merge_coco_jsons(json_files, output_file):
    print("Starting the unified merge process...")
    
    # Master dictionary with ONLY ONE category: "ball"
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "ball", "supercategory": "none"}],
        "info": {"description": "Merged 8-Ball Pool Dataset - Single Class"}
    }

    image_id_offset = 0
    annotation_id_offset = 0

    for file_path in json_files:
        print(f"Processing: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)

        local_cat_id_to_name = {}
        for cat in data.get('categories', []):
            local_cat_id_to_name[cat['id']] = cat['name'].lower().strip()

        # 1. Merge Images with ID offset
        local_img_to_merged_img = {}
        for img in data.get('images', []):
            old_img_id = img['id']
            new_img_id = old_img_id + image_id_offset
            local_img_to_merged_img[old_img_id] = new_img_id
            
            img['id'] = new_img_id
            merged_data['images'].append(img)
        
        # 2. Merge Annotations and FORCE category_id to 1 ("ball")
        for ann in data.get('annotations', []):
            cat_name = local_cat_id_to_name.get(ann['category_id'], "")
            
            # --- THE FIX: Identify and destroy the dots ---
            if cat_name == 'dot':
                continue # Skip this annotation entirely!
            
            old_ann_id = ann['id']
            ann['id'] = old_ann_id + annotation_id_offset
            ann['image_id'] = local_img_to_merged_img[ann['image_id']]
            
            # Force valid annotations to be our unified "ball" class
            ann['category_id'] = 1
            
            merged_data['annotations'].append(ann)

        # Update offsets
        if data.get('images'):
            image_id_offset = max([img['id'] for img in merged_data['images']]) + 1
        if data.get('annotations'):
            annotation_id_offset = max([ann['id'] for ann in merged_data['annotations']]) + 1

    # Save out the master file
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)
        
    print(f"\nSuccess! Merged into --> {output_file}")
    print(f"Total Images: {len(merged_data['images'])}")
    print(f"Total Annotations: {len(merged_data['annotations'])}")
    print(f"Unique Categories: {[c['name'] for c in merged_data['categories']]}")

# ---------------------------------------------------------
# Run the function on your specific files
# ---------------------------------------------------------
files_to_merge = [
    "images/test/_annotations.coco (2).json",
    "images/test/_annotations.coco (3).json",
    "images/test/_annotations.coco (4).json",
    "images/test/_annotations.coco.json"
]

output_filename = "images/test/test_annotations.json"

merge_coco_jsons(files_to_merge, output_filename)