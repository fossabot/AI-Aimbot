import scipy
from glob import glob
import numpy as np
import h5py
import os
from PIL import Image
from shutil import copyfile
import xml.etree.ElementTree as ET
import glob
import math
import config
from keras.applications import imagenet_utils
import skimage
config = config.Config()
cwd = os.getcwd()


def load_sis(train_path, test_path, mode):
    """ Load data for the semantic image segmentor
    """
    image_data_train_x = []
    image_data_test_x = []
    image_data_train_y = []
    image_data_test_y = []
    training_xml = glob.glob(f'{train_path}*.xml')[0]
    testing_xml = glob.glob(f'{test_path}*.xml')[0]

    x_datasets = [image_data_train_x,image_data_test_x]
    y_datasets = [image_data_train_y,image_data_test_y]
    paths = [train_path,test_path]
    xml_paths = [training_xml,testing_xml]

    # check to make sure xml exists in both datasets
    if os.path.isfile(training_xml) and os.path.isfile(testing_xml):
        for i in range(len(x_datasets)):
            #raw images
            for image in sorted(glob.glob(f'{paths[i]}*.png')):
                img = Image.open(paths[i]+image)
                image_data_train_x.append(np.array(img, dtype='uint8' ))
            x_datasets[i] = np.concatenate(x_datasets[i])
            y_datasets[i] = np.zeros((len(x_datasets[i]),
            config.RESOLUTION_CAPTURE_WIDTH,config.RESOLUTION_CAPTURE_HEIGHT,
            len(config.SIS_ENTITIES_SR)), dtype=np.uint8)

            #annotated ground truth
            #convert xml file from cvat to ground truth for y
            tree = ET.parse(xml_paths[i])
            root = tree.getroot()
            index = 0
            for image in root.findall('image'):
                for polygon in image.findall('polygon'):
                    label = polygon.get('label')
                    points = polygon.get('points').split(';')
                    #process points into readable format
                    for i in range(len(points)):
                        points[i] = points[i].split(',')
                    points = [[math.ceil(float(x)) for x in lst] for lst in points]
                    image_data_train_y[index,:,:,config.SIS_ENTITIES_SR.index(label)] = \
                    create_polygon(image_data_train_y[index,:,:,config.SIS_ENTITIES_SR.index(label)],points)
                index+=1
                        

    else:
        assert 'Missing xml annotations for dataset!'


    return image_data_train_x, image_data_train_y, image_data_test_x, image_data_test_y

def prepare_images(data_path, name):
    #TODO: move all images in folders to one folder with ordered images and file names
    
    newpath = f'{data_path}{name}/'
    #create new folder
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    i = 0
    for folder in os.walk(data_path):
        #organize images in folder
        for image in sorted(glob.glob(f'{data_path}{folder}/*.png')):  
            #add each image to new folder
            copyfile(data_path+image, f'{newpath}{i}.png')
            i+=1
    return

def check(p1, p2, input):
    """
    Uses the line defined by p1 and p2 to check array of 
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(input.shape) # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) +  p1[1]    
    sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign

def create_polygon(input, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values remain unchanged"""

    fill = np.ones(input.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k-1], vertices[k], input)], axis=0)

    # Set all values inside polygon to one
    input[fill] = 1

    return input

def crop_center(img, cropx, cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx, :]

def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)

def preprocess_input(self, x):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Input array scaled to [-1.,1.]
    """
    return imagenet_utils.preprocess_input(x, mode='tf')

def normalize(x):
    return x/255

def denormalize(x):
    return x*255
    
if __name__ == "__main__":
    # load_vae('/datasets/autoencoder/train/','/datasets/autoencoder/test/')
    load_sis(cwd+'/datasets/sis/train/',cwd+'/datasets/sis/test/')