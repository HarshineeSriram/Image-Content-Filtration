import os
import hashlib
import requests
import matplotlib.pyplot as plt
from scipy.misc import imread

# import constants
from constants_scraper_unsafe import (
    file1, filepath
)

Lines = file1.readlines()
img_index = 0
for line in Lines:
    try:
        response = requests.get(line)
        file = open("image " + str(img_index) + ".extension", "wb")
        img_index = img_index+1
        file.write(response.content)
        file.close()
    except Exception:
        continue

os.chdir(filepath)


def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


duplicates = []
hash_keys = dict()
for index, filename in enumerate(os.listdir('.')):
    # listdir('.') = current directory
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            filehash = hashlib.md5(f.read()).hexdigest()
        if filehash not in hash_keys:
            hash_keys[filehash] = index
        else:
            duplicates.append((index, hash_keys[filehash]))

file_list = os.listdir()
for file_indexes in duplicates[:30]:
    try:
        plt.subplot(121), plt.imshow(imread(file_list[file_indexes[1]]))
        plt.title(file_indexes[1]), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(imread(file_list[file_indexes[0]]))
        plt.title(str(file_indexes[0]) + ' duplicate')
        plt.xticks([])
        plt.yticks([])
        plt.show()  # Displays the set of duplicate images

    except OSError:
        continue


for index in duplicates:
    os.remove(file_list[index[0]])
