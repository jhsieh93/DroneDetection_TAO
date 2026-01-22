"""
Convert COCO JSON annotations to KITTI format for NVIDIA TAO DetectNet_v2
"""

import json
import os
from pathlib import Path
import shutil


def convert_coco_to_kitti(coco_json_path, images_dir, output_dir):
    """
    Convert COCO format annotations to KITTI format

    KITTI format per line:
    <class_name> 0 0 0 <xmin> <ymin> <xmax> <ymax> 0 0 0 0 0 0 0
    """

    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create category mapping (category_id -> category_name)
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Create image mapping (image_id -> image_info)
    images = {img['id']: img for img in coco_data['images']}

    # Create output directories
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Convert each image's annotations
    converted_count = 0
    for image_id, image_info in images.items():
        image_filename = image_info['file_name']
        image_path = os.path.join(images_dir, image_filename)

        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        # Copy image to output directory
        output_image_path = os.path.join(output_images_dir, image_filename)
        shutil.copy(image_path, output_image_path)

        # Create KITTI label file
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(output_labels_dir, label_filename)

        with open(label_path, 'w') as f:
            # Get annotations for this image
            if image_id in annotations_by_image:
                for ann in annotations_by_image[image_id]:
                    # COCO bbox format: [x, y, width, height]
                    x, y, w, h = ann['bbox']

                    # Convert to KITTI format: [xmin, ymin, xmax, ymax]
                    xmin = x
                    ymin = y
                    xmax = x + w
                    ymax = y + h

                    # Get category name
                    category_id = ann['category_id']
                    class_name = categories[category_id].replace(' ', '_')

                    # KITTI format line
                    # <class_name> truncated occluded alpha xmin ymin xmax ymax height width length x y z rotation
                    # For 2D detection, we only need: class_name 0 0 0 xmin ymin xmax ymax 0 0 0 0 0 0 0
                    kitti_line = f"{class_name} 0 0 0 {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} 0 0 0 0 0 0 0\n"
                    f.write(kitti_line)

        converted_count += 1
        if converted_count % 100 == 0:
            print(f"Converted {converted_count} images...")

    print(f"\nConversion complete!")
    print(f"Total images converted: {converted_count}")
    print(f"Output directory: {output_dir}")
    print(f"Images saved to: {output_images_dir}")
    print(f"Labels saved to: {output_labels_dir}")


def main():
    # Define paths - UPDATE THESE FOR YOUR SETUP
    base_dataset_dir = r"C:\Users\Jordan\PycharmProjects\DroneDetection\Dataset"

    # Convert train set
    print("Converting training set...")
    convert_coco_to_kitti(
        coco_json_path=os.path.join(base_dataset_dir, 'train', '_annotations.coco.json'),
        images_dir=os.path.join(base_dataset_dir, 'train'),
        output_dir=os.path.join(base_dataset_dir, 'kitti', 'train')
    )

    # Convert validation set
    print("\nConverting validation set...")
    convert_coco_to_kitti(
        coco_json_path=os.path.join(base_dataset_dir, 'valid', '_annotations.coco.json'),
        images_dir=os.path.join(base_dataset_dir, 'valid'),
        output_dir=os.path.join(base_dataset_dir, 'kitti', 'valid')
    )

    # Convert test set if it exists
    test_json = os.path.join(base_dataset_dir, 'test', '_annotations.coco.json')
    if os.path.exists(test_json):
        print("\nConverting test set...")
        convert_coco_to_kitti(
            coco_json_path=test_json,
            images_dir=os.path.join(base_dataset_dir, 'test'),
            output_dir=os.path.join(base_dataset_dir, 'kitti', 'test')
        )

    print("\n" + "=" * 50)
    print("All conversions complete!")
    print("Your KITTI-format dataset is ready at:")
    print(os.path.join(base_dataset_dir, 'kitti'))
    print("=" * 50)


if __name__ == "__main__":
    main()