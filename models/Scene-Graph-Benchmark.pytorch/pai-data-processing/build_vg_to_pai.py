import os
import re
import csv
import json
from PIL import Image
from h5py_utils import process_h5, inspect_splits, inspect_objects, inspect_relationships
from vg_to_pai_utils import convert_annotations_to_objects_json, convert_annotations_to_relationships_json
from cvat_annotations_utils import parse_annotations, filter_nobox_annotations, display_and_save_image, annotate_image

def create_image_data_json(labels_csv_path, image_dir_path, output_json_path):
    """
    Converts custom labels and images into the Visual Genome image_data.json format,
    setting the 'url' field to the absolute path of each image.
    
    Args:
        labels_csv_path (str): Path to the labels CSV file.
        image_dir_path (str): Path to the directory containing images.
        output_json_path (str): Path where the output image_data.json will be saved.


    Note: Unsure of the meaning of the anti_prop field
    """
    image_data_list = []
    image_id_counter = 1

    # Open the labels CSV file
    with open(labels_csv_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row['filename (date_time_id)']
            image_filename = filename + '.jpg'  # Adjust if images have a different extension
            image_path = os.path.join(image_dir_path, image_filename)
            
            # Check if the image file exists
            if not os.path.isfile(image_path):
                print(f"Warning: Image file {image_path} not found. Skipping.")
                continue
            
            # Open the image to get width and height
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                continue

            # Construct the URL as the absolute path to the image
            url = os.path.abspath(image_path)

            # Construct the image data entry
            image_data = {
                "width": width,
                "url": url,  # Set URL to the absolute image path
                "height": height,
                "image_id": image_id_counter,
                "coco_id": None,
                "flickr_id": None,
                "anti_prop": 0.0  # Set to 0.0 as the meaning is unknown
            }

            image_data_list.append(image_data)
            image_id_counter += 1

    # Save the image data list to a JSON file
    with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(image_data_list, jsonfile, indent=4)
    
    print(f"image_data.json has been created at {output_json_path} with {len(image_data_list)} entries.")

def create_sgg_dicts_with_attri_json(annotations, labels, obj_pred_info, json_output_path):
    # Step 1: Read obj-pred-list.csv and build mappings
    label_to_idx = {}
    idx_to_label = {}
    predicate_to_idx = {}
    idx_to_predicate = {}
    object_count = {}     # Initialize counts
    predicate_count = {}  # Initialize counts

    with open(obj_pred_info, 'r', encoding='utf-8') as csvfile:
        # Skip initial empty lines and find the header
        while True:
            line = csvfile.readline()
            if not line:
                break  # End of file
            if 'Object list' in line and 'Predicate list' in line and 'Index' in line:
                break  # Found the header
        reader = csv.DictReader(csvfile, fieldnames=['Object list', 'Predicate list', 'Index'])
        for row in reader:
            label = row['Object list'].strip()
            predicate = row['Predicate list'].strip()
            index_str = row['Index'].strip()
            if not index_str:
                continue  # Skip if index is missing
            index = int(index_str)
            if label:
                label_to_idx[label] = index
                idx_to_label[str(index)] = label
                object_count[label] = 0  # Initialize count to zero
            if predicate:
                predicate_to_idx[predicate] = index
                idx_to_predicate[str(index)] = predicate
                predicate_count[predicate] = 0  # Initialize count to zero

    # Step 2: Process annotations to calculate object counts
    for annotation in annotations:
        for box in annotation['boxes']:
            label = box['label']
            if label in label_to_idx:
                object_count[label] += 1
            else:
                # Handle labels not found in obj-pred-list.csv (optional)
                pass

    # Step 3: Process toy-labels.csv to calculate predicate counts
    with open(labels, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            labels_str = row['label']
            # Use regex to find all predicates
            predicates = re.findall(r"\('.*?',\s*'(.*?)',\s*'.*?'\)", labels_str)
            for predicate in predicates:
                if predicate in predicate_to_idx:
                    predicate_count[predicate] += 1
                else:
                    # Handle predicates not found in obj-pred-list.csv (optional)
                    pass

    # Step 4: Prepare the final JSON structure
    output_data = {
        "object_count": object_count,
        "predicate_to_idx": predicate_to_idx,
        "predicate_count": predicate_count,
        "idx_to_predicate": idx_to_predicate,
        "label_to_idx": label_to_idx,
        "attribute_count": {},
        "idx_to_attribute": {},
        "attribute_to_idx": {}
    }

    # Step 5: Write the output to a JSON file
    with open(json_output_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(output_data, jsonfile, indent=4)





def main():

    # labels_path = 'labels-clean.v1.csv'
    # sg_image_dir = '/home/ubuntu/code/context-aware-sgg/models/Scene-Graph-Benchmark.pytorch/datasets/custom-data/sg-images'
    # pai_img_json_path = '../datasets/custom-data/'
    
    # labels_path = '../datasets/custom-data/pai-toy-data/toy-labels.csv'
    # toy_sg_image_dir = '../datasets/custom-data/pai-toy-data/images'

    # create_image_data_json(labels_path, toy_sg_image_dir, output_json_path=pai_img_json_path)

    pai_data_dir = '../datasets/custom-data/pai-toy-data/'
    toy_images_dir = pai_data_dir + 'images/'
    xml_annotations = pai_data_dir + 'pai-toy-data-annotations.xml'

    ### code to view image with boxes to verify boxes are correct
    # test_image = toy_data[3]
    # test_image_path = os.path.join(toy_images_dir, test_image['name'])
    # annotated_image = annotate_image(test_image_path, test_image['boxes'])
    # save_path = f"../datasets/custom-data/pai-toy-data/test-annotation-image/annotated_{test_image['name']}"
    # display_and_save_image(annotated_image, save_path)    


    annotation_data = parse_annotations(xml_annotations)
    toy_data_annotations = filter_nobox_annotations(annotation_data)
    print(toy_data_annotations)

    # toy_labels = '../datasets/custom-data/pai-toy-data/toy-labels.csv'
    # toy_obj_pred_info = '../datasets/custom-data/pai-toy-data/obj-pred-list.csv'
    # json_output_path = './pai-SGG-dicts-with-attri.json'

    # create_sgg_dicts_with_attri_json(toy_data_annotations, toy_labels, toy_obj_pred_info, json_output_path)
    # print('Generated SGG-dicts-with-attri.json')


    # # create pai-objects.json
    # convert_annotations_to_objects_json(toy_data_annotations, 
    #                                     object_synsets_filepath='/home/ubuntu/code/context-aware-sgg/data-helper-scripts/scene-graph-TF-release/data_tools/object_synsets.json', 
    #                                     output_filepath='../datasets/custom-data/pai-toy-data/pai-objects.json')


    # # # create pai-relationships.json
    # labels_csv_filepath = '../datasets/custom-data/pai-toy-data/toy-labels.csv'
    # objects_json_filepath = '/home/ubuntu/code/context-aware-sgg/models/Scene-Graph-Benchmark.pytorch/datasets/custom-data/pai-toy-data/pai-objects.json'
    # relationships_synsets_filepath = '/home/ubuntu/code/context-aware-sgg/models/Scene-Graph-Benchmark.pytorch/datasets/custom-data/pai-toy-data/relationship_synsets.json'
    # output_filepath = '/home/ubuntu/code/context-aware-sgg/models/Scene-Graph-Benchmark.pytorch/datasets/custom-data/pai-toy-data/pai-relationships.json'

    # convert_annotations_to_relationships_json(labels_csv_filepath, objects_json_filepath, relationships_synsets_filepath, output_filepath)

    h5_path = '/home/ubuntu/code/context-aware-sgg/models/Scene-Graph-Benchmark.pytorch/datasets/custom-data/pai-SGG.h5'

    print('process_h5 output')
    process_h5(h5_path)

    print('inspect splits output')
    inspect_splits(h5_path)
    print('inspect_objects output')
    inspect_objects(h5_path, 1)
    print('inspect_relationships output')
    inspect_relationships(h5_path, 1)
    



if __name__ == '__main__':
    main()
