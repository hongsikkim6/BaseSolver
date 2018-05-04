import os
import xml.etree.ElementTree as ET

path = '/home/hongsikkim/HDD/data/Imagenet/bbox/train/n02444819'

a = os.listdir(path)

for file in a:
    filename = os.path.join(path, file)
    tree = ET.parse(filename)
    objs = tree.findall('object')
    if len(objs) > 1:
        print(file)
