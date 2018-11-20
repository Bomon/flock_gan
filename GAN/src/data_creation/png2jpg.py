import numpy as np
from PIL import Image
from glob import glob
import os

def main():
    data_files = glob("./flocking_512/train/*/*.png")
    for f in data_files:
        print("Rename: " + str(f))
        im = Image.open(f)
        rgb_im = im.convert('RGB')
        rgb_im.save(f[:-3]+'jpg')
	os.remove(f)

if __name__ == "__main__":
    main()
