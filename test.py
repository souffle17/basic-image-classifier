import numpy as np
import os
import keras._tf_keras.keras

model = keras.saving.load_model("model.keras")

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

while 1==1:
    print("Image file path: ", end="")

    file_name = input()
    
    if os.path.isfile(file_name):
        try: 
            img = keras.utils.load_img(file_name, target_size=(32,32))
            input_arr = keras.utils.img_to_array(img)
            input_arr = input_arr / 255.0
            input_arr = np.expand_dims(input_arr, axis=0)

            pred = model.predict(input_arr)
            predicted_class = np.argmax(pred, axis=1)
            print(f"Predicted class: {class_names[predicted_class[0]]}")
            print(f"Probabilities: {pred}")
        except:
            print("Not an image")
    else:
        print("File not found")