#supress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

#import key libraries
import numpy as np
import keras
from keras.preprocessing import image
import json
import sys


#import pretrained model
model = keras.applications.VGG16(include_top=False, weights='imagenet', pooling="max")

#import prediction labels(0, 1, etc) to human labels ("splat", "drop", etc)
with open("src/codings.json", 'r') as f:
    id_to_label = json.loads(f.readline())

#import image
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def make_prediction(file_name):
    #get embedding for test image
    test_image = path_to_tensor(file_name)
    test_embedding = model.predict(test_image)[0]
#    print("test_embedding", test_embedding)

    #load precalculated image embeddings and labels
    embeddings = np.load('src/embeds.npy')
    labels = np.load("src/labels.npy")

    #take dot product of test image with saved images
    from scipy import spatial
    scores = [1 - spatial.distance.cosine(test_embedding, train_ex) for train_ex in embeddings]
#    print(scores) 
    #find most similar image
    best_match = np.argmax(scores)
#    print(best_match, np.max(scores))
    best_label = labels[best_match]
    return id_to_label[str(best_label)]

print("\n\n PREDICTION: ", make_prediction(sys.argv[1]))
