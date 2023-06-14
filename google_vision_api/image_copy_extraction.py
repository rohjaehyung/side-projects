import io
import os
import pandas as pd
from google.cloud import vision
from google_vision_ai import VisionAI
from google_vision_ai import prepare_image_local, prepare_image_web, draw_boundary, draw_boundary_normalized, prepare_images_from_directory

# client autho and instances
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"service_owner.json"
client = vision.ImageAnnotatorClient()

# prepare the images (local source)
image_dir = r".\image"
image_files = os.listdir(image_dir)

all_text_detections = []

# loop over each image and get the first cell of text detection
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = prepare_image_local(image_path)

    va = VisionAI(client, image)
    texts = va.text_detection()

    if texts:
        # Create a DataFrame and get the first cell
        df = pd.DataFrame(texts)
        first_cell = df.iloc[0, 0].replace('\n', ' ')

        # Append the image file name and the first cell to the list as a tuple
        all_text_detections.append((image_file, first_cell))

# Create a DataFrame of all image files and img copy
df_all_text_detections = pd.DataFrame(all_text_detections, columns=['ImageFile', 'image_copy'])

df_all_text_detections.to_csv('image_copy.csv', index=False)
