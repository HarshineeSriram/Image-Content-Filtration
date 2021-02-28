# Extracting safe images from Lorem Picsum
import random
import requests

# importing under-definable constants
from constants_scraper_safe import (
    lowest_dimension,
    highest_dimension,
    number_of_images
)

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
