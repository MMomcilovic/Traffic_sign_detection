from tflite_runtime.interpreter import Interpreter
import tensorflow as tf
import numpy as np
import cv2

def classify_image(interpreter, image, top_k=1):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]][0]


def load_labels(path): # Read the labels from the text file as a Python list.
    with open(path, 'r') as f:
        return [line.strip() for i, line in enumerate(f.readlines())]

if __name__ == '__main__':

    model_path = './best_saved_model/best_float32.tflite'

    label_path = './best_saved_model/label.txt'

    interpret = Interpreter(model_path=model_path)
    interpreter = tf.lite.Interpreter(model_path)


    interpreter.allocate_tensors()
    _, height, width, _ = interpret.get_input_details()[0]['shape']
    print("Image Shape (", width, ",", height, ")")

    img = cv2.imread('./img.png')
    img = cv2.resize(img, (640, 640))

    label_id, prob = classify_image(interpreter, img)

    labels = load_labels(label_path)

    class_label = labels[label_id]

    print("Image Label is :", class_label, ", with Accuracy :", np.round(prob*100, 2), "%.")
