import argparse
from PIL import Image
import tensorflow as tf
import numpy as np
from utility import process_image
import tensorflow_hub as hub
import json

parser = argparse.ArgumentParser(description='calculte volume of cylinder')
parser.add_argument('image_path', type=str, help='Path of the Image')
parser.add_argument('model', type=str, help='Neural Network Trained Model')
parser.add_argument('--top_k', type=int, help='Return the top K most likely classes')
parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names')

args = parser.parse_args()

def predict(image_path, model, top_k=5):
    """
    the predict function should take an image, a model, and then returns the top_K most likely class labels along with the probabilities.
    """
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    image = np.expand_dims(processed_test_image, axis=0)
    probabilities = model.predict(image)[0]
    top_k_pred = np.argsort(probabilities)[-top_k:]
    top_k_probs = probabilities[top_k_pred] 
    return top_k_pred, top_k_probs

if __name__ == '__main__':
    custom_objects = {'KerasLayer': hub.KerasLayer}
    reloaded_model = tf.keras.models.load_model(args.model, custom_objects=custom_objects)

    json_file = 'label_map.json'
    
    if(args.category_names):
        json_file = args.category_names

    with open(json_file, 'r') as f:
        class_names = json.load(f)

    if(args.top_k):
        top_preds, top_probs = predict(args.image_path, reloaded_model, args.top_k)
    else: 
        top_preds, top_probs = predict(args.image_path, reloaded_model)
    print(f'{"Prediction": <30}{"" :<10} Probability')
    for pred, prop in zip(top_preds, top_probs):
        print(f'{class_names[f"{pred}"] :<30}{"" :<10} {prop:.4f}')

