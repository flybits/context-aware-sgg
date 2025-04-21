import os
import re
import ast
import pandas as pd
from datetime import datetime

def clean_data(df):
    """
    Removes deleted entries from the DataFrame and evaluates label strings to lists.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        tuple: A tuple containing the cleaned DataFrame and the count of deleted entries.
    """
    # Remove rows where 'filename (date_time_id)' is marked as "deleted"
    count_deleted = df[df['filename (date_time_id)'] == 'deleted'].shape[0]
    df = df[df['filename (date_time_id)'] != 'deleted'].copy()

    # Ensure labels are evaluated from strings to lists if stored as string representations
    df['label'] = df['label'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    return df, count_deleted

def ensure_temp_id(df):
    """
    Ensures that the DataFrame has a 'temp_id' column for indexing.

    Parameters:
        df (pd.DataFrame): The DataFrame to update.

    Returns:
        pd.DataFrame: The updated DataFrame with 'temp_id' column.
    """
    if 'temp_id' not in df.columns:
        df['temp_id'] = range(1, len(df) + 1)
    return df

def update_intermediary_tuples(df):
    """
    Updates the 'filename (date_time_id)' and IDs in 'label' tuples to match 'temp_id'.

    Parameters:
        df (pd.DataFrame): The DataFrame to update.

    Returns:
        pd.DataFrame: The DataFrame with updated filenames and labels.
    """
    def update_filename_id(filename, new_id):
        parts = filename.split('_')
        parts[-1] = str(new_id)  # Update the last part which is the ID
        return '_'.join(parts)

    def update_tuple_ids(tuples, new_id):
        updated_tuples = []
        for t in tuples:
            updated_tuples.append(tuple(
                [re.sub(r'_[0-9]+$', f'_{new_id}', element) if isinstance(element, str) else element for element in t]
            ))
        return updated_tuples

    for index, row in df.iterrows():
        new_id = row['temp_id']
        # Update the 'filename (date_time_id)' with the new ID
        df.at[index, 'filename (date_time_id)'] = update_filename_id(row['filename (date_time_id)'], new_id)
        # Update the tuples in 'label'
        if isinstance(row['label'], str):
            row['label'] = ast.literal_eval(row['label'])
        df.at[index, 'label'] = update_tuple_ids(row['label'], new_id)

    return df

def clean_img_indexes(cleaned_labels_path, sg_img_dir):
    """
    Renames image files in sg_img_dir according to the provided clean labels.

    Parameters:
        cleaned_labels_path (str): Path to the cleaned labels CSV file.
        sg_img_dir (str): Directory containing the images to rename.
    """
    # Load the CSV file containing the labels
    data = pd.read_csv(cleaned_labels_path)
    filename_ids = list(data['filename (date_time_id)'])
    current_files = os.listdir(sg_img_dir)

    # Build mapping from date_time to correct ID
    date_time_to_id = {}
    for fname_id in filename_ids:
        date_time = '_'.join(fname_id.split('_')[:-1])
        id_str = fname_id.split('_')[-1]
        date_time_to_id[date_time] = id_str

    # Process current files
    for fname in current_files:
        if not fname.endswith('.jpg'):
            continue  # Skip non-JPG files
        base_fname = fname[:-4]  # Remove '.jpg'
        date_time = '_'.join(base_fname.split('_')[:-1])
        current_id = base_fname.split('_')[-1]

        correct_id = date_time_to_id.get(date_time)
        if correct_id and current_id != correct_id:
            new_fname = f'{date_time}_{correct_id}.jpg'
            old_path = os.path.join(sg_img_dir, fname)
            new_path = os.path.join(sg_img_dir, new_fname)
            os.rename(old_path, new_path)
            print(f'Renamed {fname} to {new_fname}')
        elif not correct_id:
            print(f'No matching ID found for {fname}')

def update_deleted_entries(labels_with_deleted_entries, sg_img_dir, updated_labels_path):
    """
    Main function to clean data by removing deleted entries, updating indexes,
    and renaming image files accordingly.

    Parameters:
        labels_with_deleted_entries (str): Path to the labels CSV file with deleted entries.
        sg_img_dir (str): Directory containing the images.
        updated_labels_path (str): Path to save the cleaned labels CSV.
    """
    data = pd.read_csv(labels_with_deleted_entries)

    # Remove "deleted" entries and calibrate the indexes to retain chronological order
    intermediary_data, num_deleted = clean_data(data)
    print(f'Removed {num_deleted} deleted entries.')

    intermediary_data = ensure_temp_id(intermediary_data)
    cleaned_data = update_intermediary_tuples(intermediary_data)
    print('Corrected indexing in labels.')

    # Drop temp_id column as it's now unnecessary
    cleaned_data = cleaned_data.drop('temp_id', axis=1)

    # Save the updated data back to a CSV file
    cleaned_data.to_csv(updated_labels_path, index=False)
    print(f'Saved cleaned labels to {updated_labels_path}.')

    # Rename image files based on updated labels
    clean_img_indexes(updated_labels_path, sg_img_dir)
    print('Corrected indexes of images based on deleted entries.')



def main():
    labels_with_deleted_entries = 'labels-raw.v1.csv'
    sg_img_dir = '/home/ubuntu/code/context-aware-sgg/models/Scene-Graph-Benchmark.pytorch/datasets/custom-data/sg-images'
    updated_labels_path = 'labels-clean.v1.csv'

    update_deleted_entries(labels_with_deleted_entries, sg_img_dir, updated_labels_path)

    
if __name__ == '__main__':
    main()
