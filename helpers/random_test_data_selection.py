import os
import random
import shutil

# Path to the folder containing the images
folder_path = "Lemonada"

# List all the files in the folder
all_images = os.listdir(folder_path)

# Choose 200 random images
random_images = random.sample(all_images, 200)

# Create a new folder to store the selected images
selected_folder_path = "test_data/Lemonada"
os.makedirs(selected_folder_path, exist_ok=True)

# Move the selected images to the new folder
for image in random_images:
    image_path = os.path.join(folder_path, image)
    shutil.move(image_path, selected_folder_path)

print("Random images moved successfully.")