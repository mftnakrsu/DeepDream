from tensorflow import keras
import matplotlib.pyplot as plt
from keras.applications import inception_v3
import numpy as np
import argparse
import tensorflow as tf 


tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)


parser = argparse.ArgumentParser()

parser.add_argument('--step', help='Number of the step',
                    default=20)
parser.add_argument('--iterations', help='Number of the iterations for deepDream process',
                    default=30)
parser.add_argument('--image', help='Name of the single image to perform for DeepDream',
                    default='img/0.jpg')
parser.add_argument('--max_loss', help='Max loss value',
                    default=20.)

args = parser.parse_args()

# PROVIDE PATH TO IMAGE DIRECTORY
base_image_path= args.image

# PROVIDE THE STEP SIZE
step=int(args.step)

# PROVIDE THE NUMBER OF THE ITERATIONS
iterations=int(args.iterations)

# PROVIDE THE MAX LOSS
max_loss=float(args.max_loss)

#Number of octaves and scale
num_octave = 3
octave_scale = 1.4

def compute_loss(input_image):
    features = feature_extractor(input_image)
    loss = tf.zeros(shape=())
    for name in features.keys():
        coeff = layer_settings[name]
        activation = features[name]
        loss += coeff * tf.reduce_mean(tf.square(activation[:, 2:-2, 2:-2, :]))
    return loss

@tf.function
def gradient_ascent_step(image, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = compute_loss(image)
    grads = tape.gradient(loss, image)
    grads = tf.math.l2_normalize(grads)
    image += learning_rate * grads
    return loss, image

def gradient_ascent_loop(image, iterations, learning_rate, max_loss=None):
    for i in range(iterations):
        loss, image = gradient_ascent_step(image, learning_rate)
        if max_loss is not None and loss > max_loss:
            break
        print(f"... Loss value at step {i}: {loss:.2f}")
    return image

def preprocess_image(image_path):
    img = keras.preprocessing.image.load_img(image_path)
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.inception_v3.preprocess_input(img)
    return img

def deprocess_image(img):
    img = img.reshape((img.shape[1], img.shape[2], 3))
    img /= 2.0
    img += 0.5
    img *= 255.
    img = np.clip(img, 0, 255).astype("uint8")
    return img

plt.axis("off")
plt.imshow(keras.preprocessing.image.load_img(base_image_path))
#plt.show()

model = inception_v3.InceptionV3(weights="imagenet", include_top=False)
#print(model.summary())

#LAYERS
layer_settings = {
    "mixed7": 1.5,
    "mixed8":1.5,
    "mixed5":1.0,
}
outputs_dict = dict(
    [
        (layer.name, layer.output)
        for layer in [model.get_layer(name) for name in layer_settings.keys()]
    ]
)

feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

original_img = preprocess_image(base_image_path)
original_shape = original_img.shape[1:3]

successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]

shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])

img = tf.identity(original_img)
for i, shape in enumerate(successive_shapes):
    print(f"Processing octave {i} with shape {shape}")
    img = tf.image.resize(img, shape)
    img = gradient_ascent_loop(
        img, iterations=iterations, learning_rate=step, max_loss=max_loss
    )
    upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
    same_size_original = tf.image.resize(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img
    img += lost_detail
    shrunk_original_img = tf.image.resize(original_img, shape)

keras.preprocessing.image.save_img("dream.png", deprocess_image(img.numpy()))

