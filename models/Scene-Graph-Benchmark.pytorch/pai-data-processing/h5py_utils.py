import h5py
import numpy as np

def process_h5(h5_file_path):
    """
    Inspects the given HDF5 (.h5) file and prints out its contents,
    including datasets and groups with their shapes and data types.

    Args:
        h5_file_path (str): Path to the HDF5 file to inspect.
    """
    with h5py.File(h5_file_path, 'r') as h5_file:
        print(f"Inspecting HDF5 file: {h5_file_path}\n")
        
        def print_attrs(name, obj):
            """
            Callback function for h5py visititems() to print attributes of each object.
            """
            obj_type = 'Group' if isinstance(obj, h5py.Group) else 'Dataset'
            print(f"{obj_type}: {name}")
            if isinstance(obj, h5py.Dataset):
                print(f" - Shape: {obj.shape}")
                print(f" - Data type: {obj.dtype}")
                # Optionally, print sample data
                print(f" - Sample data: {obj[0]}")
            elif isinstance(obj, h5py.Group):
                print(f" - Contains keys: {list(obj.keys())}")
            print()

        # Traverse the HDF5 file and print information
        h5_file.visititems(print_attrs)


def inspect_splits(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5_file:
        splits = h5_file['split'][:]
        unique_splits, counts = np.unique(splits, return_counts=True)
        split_mapping = {0: 'train', 1: 'val', 2: 'test'}
        print("Dataset splits:")
        for split_value, count in zip(unique_splits, counts):
            split_name = split_mapping.get(split_value, 'unknown')
            print(f" - {split_name} ({split_value}): {count} images")


def inspect_objects(h5_file_path, image_index):
    with h5py.File(h5_file_path, 'r') as h5_file:
        img_to_first_box = h5_file['img_to_first_box'][:]
        img_to_last_box = h5_file['img_to_last_box'][:]
        boxes = h5_file['boxes_1024'][:]
        labels = h5_file['labels'][:]

        first_box = img_to_first_box[image_index]
        last_box = img_to_last_box[image_index]
        print(f"Image {image_index} has objects from index {first_box} to {last_box}")

        object_boxes = boxes[first_box:last_box+1]
        object_labels = labels[first_box:last_box+1]

        print("Object boxes:")
        print(object_boxes)
        print("Object labels:")
        print(object_labels)


def inspect_relationships(h5_file_path, image_index):
    with h5py.File(h5_file_path, 'r') as h5_file:
        img_to_first_rel = h5_file['img_to_first_rel'][:]
        img_to_last_rel = h5_file['img_to_last_rel'][:]
        relationships = h5_file['relationships'][:]
        predicates = h5_file['predicates'][:]

        first_rel = img_to_first_rel[image_index]
        last_rel = img_to_last_rel[image_index]
        print(f"Image {image_index} has relationships from index {first_rel} to {last_rel}")

        rels = relationships[first_rel:last_rel+1]
        preds = predicates[first_rel:last_rel+1]

        print("Relationships (object pairs):")
        print(rels)
        print("Predicates:")
        print(preds)





def main():

    labels_path = 'labels-clean.v1.csv'
    sg_image_dir = '../datasets/custom-data/sg-images'
    pai_img_json_path = 'image_data.json'

    vg_sgg_dict_h5 = '/home/ubuntu/code/context-aware-sgg/models/Scene-Graph-Benchmark.pytorch/datasets/vg/VG-SGG-with-attri.h5'

    print('process_h5 output')
    process_h5(vg_sgg_dict_h5)

    print('inspect splits output')
    inspect_splits(vg_sgg_dict_h5)
    print('inspect_objects output')
    inspect_objects(vg_sgg_dict_h5, 0)
    print('inspect_relationships output')
    inspect_relationships(vg_sgg_dict_h5, 0)


    pai_sgg_dict_h5 = '/home/ubuntu/code/context-aware-sgg/models/Scene-Graph-Benchmark.pytorch/datasets/vg/pai-SGG-with-attri.h5'

    print('process_h5 output')
    process_h5(pai_sgg_dict_h5)

    print('inspect splits output')
    inspect_splits(pai_sgg_dict_h5)
    print('inspect_objects output')
    inspect_objects(pai_sgg_dict_h5, 0)
    print('inspect_relationships output')
    inspect_relationships(pai_sgg_dict_h5, 0)



if __name__ == '__main__':
    main()
