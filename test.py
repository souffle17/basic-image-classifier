import numpy as np
import keras._tf_keras.keras

img = keras.utils.load_img("test.jpg", target_size=(32,32))
img.show()
input_arr = keras.utils.img_to_array(img)
input_arr = input_arr / 255.0
input_arr = np.expand_dims(input_arr, axis=0)

model = keras.saving.load_model("model.keras")

pred = model.predict(input_arr)
predicted_class = np.argmax(pred, axis=1)  # Get the class with the highest probability
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print(f"Predicted class: {class_names[predicted_class[0]]}")
print(f"Probabilities: {pred}")