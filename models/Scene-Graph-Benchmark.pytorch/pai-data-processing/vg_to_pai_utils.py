import os
import csv
import ast
import json


def convert_annotations_to_objects_json(toy_data_annotations, object_synsets_filepath, output_filepath='./pai-objects.json'):
    """
    Converts toy_data_annotations to the desired objects.json format and writes to a file.
    Retrieves synsets for each label from the provided object_synsets.json file.

    Args:
        toy_data_annotations (list): List of image annotations.
        object_synsets_filepath (str): Filepath to the object_synsets.json file containing label to synset mappings.
        output_filepath (str): Output JSON filepath.
    """
    # Load the object_synsets.json file into a dictionary
    with open(object_synsets_filepath, 'r') as f:
        object_synset_mapping = json.load(f)

    # Initialize the list to hold all images' object data
    objects_data = []
    current_object_id = 1  # Starting object_id

    for annotation in toy_data_annotations:
        image_id = annotation['id']
        image_name = annotation['name']
        boxes = annotation['boxes']

        image_objects = []

        for box in boxes:
            # Extract box coordinates and calculate width and height
            x = int(round(box['xtl']))
            y = int(round(box['ytl']))
            w = int(round(box['xbr'] - box['xtl']))
            h = int(round(box['ybr'] - box['ytl']))

            # Retrieve the synset for the label from the object_synset_mapping
            label = box['label'].lower().strip()
            synset = object_synset_mapping.get(label, None)
            if synset:
                synsets = [synset]
            else:
                synsets = []

            # Create object dictionary
            obj = {
                "synsets": synsets,
                "h": h,
                "object_id": current_object_id,
                "merged_object_ids": [],  # Empty list as per requirement
                "names": [box['label']],
                "w": w,
                "y": y,
                "x": x
            }

            image_objects.append(obj)
            current_object_id += 1  # Increment object_id for the next object

        # Create the image dictionary
        image_dict = {
            "image_id": image_id,
            "objects": image_objects,
            "image_url": image_name  # Assuming image_name is the filename or URL
        }

        objects_data.append(image_dict)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the data to the output JSON file with indentation for readability
    with open(output_filepath, 'w') as f:
        json.dump(objects_data, f, indent=4)

    print(f"Successfully created {output_filepath} with {len(objects_data)} images.")




def convert_annotations_to_relationships_json(labels_csv_filepath, objects_json_filepath, relationships_synsets_filepath, output_filepath='./relationships.json'):
    """
    Converts annotations from toy-labels.csv and objects.json to the desired relationships.json format.
    Incorporates synsets for predicates using relationships_synsets.json.

    Args:
        labels_csv_filepath (str): Path to the toy-labels.csv file containing relationships.
        objects_json_filepath (str): Path to the objects.json file containing object annotations.
        relationships_synsets_filepath (str): Path to relationships_synsets.json containing predicate to synset mappings.
        output_filepath (str): Output JSON filepath for relationships.json.
    """
    # Step 1: Load relationships_synsets.json
    with open(relationships_synsets_filepath, 'r') as f:
        predicate_synset_mapping = json.load(f)
    
    # Normalize predicate keys in the mapping
    predicate_synset_mapping_normalized = {k.lower().strip(): v for k, v in predicate_synset_mapping.items()}

    # Step 2: Read objects.json and build mappings
    with open(objects_json_filepath, 'r') as f:
        objects_data = json.load(f)
    
    # Build mappings
    image_filename_to_id = {}  # Map from image filename without extension to image_id
    image_id_to_objects = {}  # Map from image_id to list of objects
    image_id_to_object_name_to_objects = {}  # Map from image_id to mapping from object names to object data
    
    for image_entry in objects_data:
        image_id = image_entry['image_id']
        image_url = image_entry['image_url']
        image_filename = os.path.splitext(image_url)[0]  # Remove file extension
        image_filename_to_id[image_filename] = image_id
        objects = image_entry['objects']
        image_id_to_objects[image_id] = objects
        
        # Build mapping from object names to objects
        name_to_objects = {}
        for obj in objects:
            object_id = obj['object_id']
            names = obj['names']
            for name in names:
                normalized_name = name.lower().strip()
                if normalized_name not in name_to_objects:
                    name_to_objects[normalized_name] = []
                name_to_objects[normalized_name].append(obj)
        image_id_to_object_name_to_objects[image_id] = name_to_objects
    
    # Step 3: Read labels CSV and build mapping from image filename to relationships
    image_filename_to_relationships = {}
    with open(labels_csv_filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        for row in reader:
            filename = row[0]
            label_str = row[1]
            # Parse label_str as a list of tuples
            relationships = ast.literal_eval(label_str)
            # Store relationships under the image filename (without extension)
            image_filename = os.path.splitext(filename)[0]
            if image_filename not in image_filename_to_relationships:
                image_filename_to_relationships[image_filename] = []
            image_filename_to_relationships[image_filename].extend(relationships)
    
    # Step 4: Generate relationships.json entries
    relationships_json_data = []
    relationship_id_counter = 1  # Starting relationship ID
    
    for image_filename, relationships in image_filename_to_relationships.items():
        if image_filename not in image_filename_to_id:
            print(f"Warning: Image filename {image_filename} not found in objects.json")
            continue
        image_id = image_filename_to_id[image_filename]
        objects = image_id_to_objects.get(image_id, [])
        name_to_objects = image_id_to_object_name_to_objects.get(image_id, {})
        
        image_relationships = []
        
        for rel in relationships:
            subj_label, predicate, obj_label = rel
            # Normalize labels
            subj_label_norm = subj_label.lower().strip()
            obj_label_norm = obj_label.lower().strip()
            predicate_norm = predicate.lower().strip()
            
            # Find subject object(s)
            subj_objects = name_to_objects.get(subj_label_norm, [])
            if not subj_objects:
                print(f"Warning: Subject '{subj_label}' not found in image {image_filename}")
                continue  # Skip this relationship
            # For simplicity, we take the first matching object
            subj_obj_full = subj_objects[0]
            subj_obj = {
                'name': subj_obj_full['names'][0] if subj_obj_full['names'] else '',
                'h': subj_obj_full['h'],
                'object_id': subj_obj_full['object_id'],
                'synsets': subj_obj_full['synsets'],
                'w': subj_obj_full['w'],
                'y': subj_obj_full['y'],
                'x': subj_obj_full['x']
            }
            
            # Find object object(s)
            obj_objects = name_to_objects.get(obj_label_norm, [])
            if not obj_objects:
                print(f"Warning: Object '{obj_label}' not found in image {image_filename}")
                continue  # Skip this relationship
            obj_obj_full = obj_objects[0]
            obj_obj = {
                'name': obj_obj_full['names'][0] if obj_obj_full['names'] else '',
                'h': obj_obj_full['h'],
                'object_id': obj_obj_full['object_id'],
                'synsets': obj_obj_full['synsets'],
                'w': obj_obj_full['w'],
                'y': obj_obj_full['y'],
                'x': obj_obj_full['x']
            }
            
            # Get synsets for the predicate
            predicate_synsets = []
            if predicate_norm in predicate_synset_mapping_normalized:
                synset = predicate_synset_mapping_normalized[predicate_norm]
                predicate_synsets = [synset] if synset else []
            else:
                print(f"Warning: Predicate '{predicate}' not found in relationships_synsets.json")
            
            # Create relationship entry
            relationship_entry = {
                "predicate": predicate,
                "object": obj_obj,
                "relationship_id": relationship_id_counter,
                "synsets": predicate_synsets,  # Synsets for predicate
                "subject": subj_obj
            }
            relationship_id_counter += 1
            image_relationships.append(relationship_entry)
        
        # Add to relationships.json data
        image_entry = {
            "relationships": image_relationships,
            "image_id": image_id
        }
        relationships_json_data.append(image_entry)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write relationships.json file
    with open(output_filepath, 'w') as f:
        json.dump(relationships_json_data, f, indent=4)
    print(f"Successfully created {output_filepath} with {len(relationships_json_data)} images.")

