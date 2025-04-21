import csv
# Your data in Form A

def convert_to_csv(data, csv_path='reltr-top20k-output.csv'):
    # Convert and write to CSV in Form B
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename (date_time_id)','label'])
        for item in data:
            image_name = item[0]
            scene_graph = item[1]
            # Convert the list of tuples to a string representation
            scene_graph_str = str(scene_graph)
            writer.writerow([image_name, scene_graph_str])