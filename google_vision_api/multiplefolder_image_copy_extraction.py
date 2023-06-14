import io
import os
import pandas as pd
from google.cloud import vision
from google_vision_ai import VisionAI
from google_vision_ai import prepare_image_local, prepare_image_web, draw_boundary, draw_boundary_normalized, prepare_images_from_directory

# client autho and instances
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"service_owner.json"
client = vision.ImageAnnotatorClient()

# define the root directory
root_dir = r"your_directory_here"

all_text_detections = []

# loop over each folder in the root directory
for folder_name in os.listdir(root_dir):
    # create the full folder path
    folder_path = os.path.join(root_dir, folder_name)

    # only process this folder if it's actually a directory
    if os.path.isdir(folder_path):
        # loop over each image file in the current directory
        for image_file in os.listdir(folder_path):
            # only process this file if it's a .jpg file
            if image_file.endswith('.jpg'):
                # create the full image path
                image_path = os.path.join(folder_path, image_file)

                # prepare the image
                image = prepare_image_local(image_path)

                va = VisionAI(client, image)
                texts = va.text_detection()

                if texts:
                    # Create a DataFrame and get the first cell
                    df = pd.DataFrame(texts)
                    first_cell = df.iloc[0, 0].replace('\n', ' ')

                    # Append the folder name, image file name and the first cell to the list as a tuple
                    all_text_detections.append((folder_name, image_file, first_cell))

# Create a DataFrame of all image files and first cells
df_all_text_detections = pd.DataFrame(all_text_detections, columns=['folder_name', 'file_name', 'copy'])

print(df_all_text_detections)

# Save DataFrame to a CSV file
df_all_text_detections.to_csv('image_copy.csv', index=False)

print("CSV file has been saved.")

