"""
Test XML Parser - Verify your XML annotation format

This script helps you test if your XML annotations can be parsed correctly
before running the full pipeline.

Usage:
    python test_xml_parser.py --xml /path/to/sample.xml
    python test_xml_parser.py --xml /path/to/sample.xml --image /path/to/sample.png
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def parse_xml_annotation(xml_path):
    """Parse XML annotation file"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = []
    
    # Try Pascal VOC format
    for obj in root.findall('object'):
        label = obj.find('name').text if obj.find('name') is not None else 'unknown'
        
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            
            annotations.append({
                'label': label,
                'bbox': [xmin, ymin, xmax, ymax],
                'format': 'Pascal VOC'
            })
    
    # Try CVAT format
    if not annotations:
        for image in root.findall('.//image'):
            for box in image.findall('box'):
                label = box.get('label', 'unknown')
                xtl = int(float(box.get('xtl', 0)))
                ytl = int(float(box.get('ytl', 0)))
                xbr = int(float(box.get('xbr', 0)))
                ybr = int(float(box.get('ybr', 0)))
                
                annotations.append({
                    'label': label,
                    'bbox': [xtl, ytl, xbr, ybr],
                    'format': 'CVAT'
                })
    
    return annotations

def normalize_label(label):
    """
    Normalize chromosome label to standard format
    
    Supports Denver classification: A1, A2, A3, B4, B5, C6-C12, D13-D15, E16-E18, F19-F20, G21-G22, X, Y
    """
    label = str(label).strip().upper()
    
    # Denver classification mapping
    denver_mapping = {
        'A1': '1', 'A2': '2', 'A3': '3',
        'B4': '4', 'B5': '5',
        'C6': '6', 'C7': '7', 'C8': '8', 'C9': '9', 'C10': '10', 'C11': '11', 'C12': '12',
        'D13': '13', 'D14': '14', 'D15': '15',
        'E16': '16', 'E17': '17', 'E18': '18',
        'F19': '19', 'F20': '20',
        'G21': '21', 'G22': '22',
        'X': 'X', 'Y': 'Y'
    }
    
    if label in denver_mapping:
        label = denver_mapping[label]
    else:
        # Handle other formats
        label = label.lower()
        label = label.replace('chromosome', '').replace('chr', '').strip()
        label = label.replace('_', '').replace('-', '').replace(' ', '')
        
        if label in ['x', 'y']:
            label = label.upper()
    
    return f'chromosome_{label}'

def visualize_annotations(image_path, xml_path, output_path='xml_test_visualization.png'):
    """Visualize bounding boxes on image"""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Error: Could not load image: {image_path}")
        return False
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Parse XML
    annotations = parse_xml_annotation(xml_path)
    
    if not annotations:
        print("❌ Error: No annotations found in XML")
        return False
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Draw bounding boxes
    colors = plt.cm.tab20(np.linspace(0, 1, 24))
    label_counts = {}
    
    for ann in annotations:
        label = normalize_label(ann['label'])
        bbox = ann['bbox']
        xmin, ymin, xmax, ymax = bbox
        
        # Count labels
        label_counts[label] = label_counts.get(label, 0) + 1
        
        # Get color based on label
        try:
            chr_num = int(label.split('_')[1]) if label.split('_')[1].isdigit() else 23
        except:
            chr_num = 23
        color = colors[chr_num % 24]
        
        # Draw rectangle
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(xmin, ymin - 5, label, 
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
               fontsize=8, color='white')
    
    ax.set_title(f'XML Annotations Visualization\n{len(annotations)} chromosomes detected', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    plt.close()
    
    return True, label_counts

def main():
    parser = argparse.ArgumentParser(description='Test XML annotation parser')
    parser.add_argument('--xml', required=True, help='Path to XML annotation file')
    parser.add_argument('--image', help='Path to corresponding image (optional, for visualization)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("XML ANNOTATION PARSER TEST")
    print("="*70)
    print(f"\nXML file: {args.xml}")
    
    # Check if file exists
    if not Path(args.xml).exists():
        print(f"\n❌ Error: XML file not found: {args.xml}")
        return
    
    # Parse XML
    print("\nParsing XML...")
    try:
        annotations = parse_xml_annotation(args.xml)
    except Exception as e:
        print(f"\n❌ Error parsing XML: {e}")
        print("\nYour XML format may not be supported.")
        print("Please check KAGGLE_DATASET_GUIDE.md for supported formats.")
        return
    
    if not annotations:
        print("\n❌ No annotations found in XML")
        print("\nSupported formats:")
        print("  • Pascal VOC (with <object> and <bndbox> tags)")
        print("  • CVAT (with <image> and <box> tags)")
        print("\nPlease check your XML structure.")
        return
    
    # Print results
    print(f"\n✓ Successfully parsed {len(annotations)} annotations")
    print(f"\nFormat detected: {annotations[0]['format']}")
    
    print("\n" + "="*70)
    print("ANNOTATIONS FOUND:")
    print("="*70)
    print(f"{'#':<5} {'Original Label':<20} {'Normalized Label':<20} {'BBox (xmin,ymin,xmax,ymax)'}")
    print("-"*70)
    
    label_counts = {}
    for i, ann in enumerate(annotations, 1):
        original_label = ann['label']
        normalized_label = normalize_label(original_label)
        bbox = ann['bbox']
        
        label_counts[normalized_label] = label_counts.get(normalized_label, 0) + 1
        
        print(f"{i:<5} {original_label:<20} {normalized_label:<20} {bbox}")
    
    print("="*70)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print(f"Total chromosomes: {len(annotations)}")
    print(f"Unique classes: {len(label_counts)}")
    print("\nPer-class distribution:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        print(f"  {label}: {count}")
    
    # Check for expected 46 chromosomes (23 pairs)
    if len(annotations) == 46:
        print("\n✓ Expected number of chromosomes (46 = 23 pairs)")
    elif len(annotations) < 46:
        print(f"\n⚠️  Warning: Only {len(annotations)} chromosomes found (expected 46)")
    else:
        print(f"\n⚠️  Warning: {len(annotations)} chromosomes found (expected 46)")
    
    # Visualize if image provided
    if args.image:
        print("\n" + "="*70)
        print("VISUALIZATION:")
        print("="*70)
        
        if not Path(args.image).exists():
            print(f"❌ Error: Image file not found: {args.image}")
        else:
            print(f"Image file: {args.image}")
            success, vis_counts = visualize_annotations(args.image, args.xml)
            if success:
                print("\n✓ Visualization complete!")
                print("  Check: xml_test_visualization.png")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    if len(annotations) > 0:
        print("✓ Your XML format is supported!")
        print("\nYou can now run the full pipeline:")
        print("  python kaggle_pipeline.py --input /path/to/data --output kaggle_processed")
    else:
        print("❌ Your XML format is not recognized")
        print("\nPlease:")
        print("  1. Check KAGGLE_DATASET_GUIDE.md for supported formats")
        print("  2. Modify parse_xml_annotation() in crop_from_xml.py")
        print("  3. Or convert your XMLs to Pascal VOC format")

if __name__ == "__main__":
    main()
