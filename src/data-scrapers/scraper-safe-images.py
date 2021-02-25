# Extracting safe images from Lorem Picsum

import random
import requests

# Set lowest_dimension and highest_dimension
# based on the spatial constraints
# for Lorem Picsum
lowest_dimension = 10
highest_dimension = 5000

# Set the number of images
# to be generated
number_of_images = 5000

for i in range(number_of_images):
    # Create custom URL
    SFW_image_generator = r'link/to/picsum'
    + str(random.randint(lowest_dimension, highest_dimension)) + r'/'
    + str(random.randint(lowest_dimension, highest_dimension))
    response = requests.get(SFW_image_generator)
    file = open("image " + str(i) + ".jpg", "wb")
    i = i+1
    file.write(response.content)
    file.close()
