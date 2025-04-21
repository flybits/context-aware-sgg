import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def parse_annotations(xml_file):
    """Parses the XML file and returns a list of image data with annotations."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    images = []
    for img in root.findall('image'):
        name = img.get('name')
        id_in_name = int(name.split('_')[-1].split('.')[0])
        boxes = [{
            'label': box.get('label'),
            'xtl': float(box.get('xtl')),
            'ytl': float(box.get('ytl')),
            'xbr': float(box.get('xbr')),
            'ybr': float(box.get('ybr'))
        } for box in img.findall('box')]
        images.append({'id': id_in_name, 'name': name, 'boxes': boxes})
    return images

def annotate_image(image_path, boxes):
    """Draws bounding boxes and labels on the image."""
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box in boxes:
        draw.rectangle([box['xtl'], box['ytl'], box['xbr'], box['ybr']], outline='red', width=3)
        text_width, text_height = font.getsize(box['label'])
        # Ensure text background stays within image boundaries
        text_y = max(box['ytl'] - text_height, 0)
        draw.rectangle([box['xtl'], text_y, box['xtl'] + text_width, box['ytl']], fill='red')
        draw.text((box['xtl'], text_y), box['label'], fill='white', font=font)
    return image

def display_and_save_image(image, save_path):
    """Displays the image using matplotlib and saves it to disk."""
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    image.save(save_path)
    print(f"Annotated image saved to '{save_path}'")

def filter_nobox_annotations(annotation_data):
    filtered_data = []
    # Loop through the data and check for bounding boxes
    for entry in annotation_data:
        if entry['boxes']:  # Check if there are boxes in the current entry
            filtered_data.append(entry)
    
    return filtered_data


def main():
    xml_file = '../datasets/custom-data/pai-toy-data/pai-toy-data-annotations.xml'
    images_folder = '../datasets/custom-data/pai-toy-data/images'

    images_data = parse_annotations(xml_file)
    print(f"Total images parsed: {len(images_data)}")
    
    # Select the first image (change the index for different images)
    first_image = images_data[2]
    image_path = os.path.join(images_folder, first_image['name'])
    
    annotated_image = annotate_image(image_path, first_image['boxes'])
    save_path = f"../datasets/custom-data/pai-toy-data/test-annotation-image/annotated_{first_image['name']}"

    # displaying not working so just save and view in './'
    display_and_save_image(annotated_image, save_path)

if __name__ == "__main__":
    main()
