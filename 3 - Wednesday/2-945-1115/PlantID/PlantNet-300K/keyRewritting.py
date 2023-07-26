import json

def rewrite_json_keys(json_file_path):
    # Read the JSON file
    with open(json_file_path) as file:
        data = json.load(file)
    
    # Create a new dictionary with updated keys
    new_data = {}
    for index, key in enumerate(sorted(data.keys())):
        new_data[str(index)] = data[key]
    
    # Write the updated JSON back to the file
    with open(json_file_path, 'w') as file:
        json.dump(new_data, file, indent=4)
    
    print("JSON keys have been rewritten successfully.")

# Example usage
json_file_path = 'C:/Users/mspring6/Documents/ML4HST-2023/PlantID/plantnet_300K_images/plantnet300K_species_id_2_name_test.json'
rewrite_json_keys(json_file_path)
