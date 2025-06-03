import os, argparse, zipfile

# os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import tensorflow as tf
import keras
from keras import ops
from keras import layers

from collections import Counter

### INITIALISATION ###
print("Static Initialising...")

show_plots = False  # Set to True to show plots

# Batch size for training and validation
batch_size = 16

# Desired image dimensions
img_width = 60
img_height = 30

# Path to the root data directory
root_data_dir = Path("data/")
# Check if the root data directory exists or empty
if not root_data_dir.exists() or not any(root_data_dir.iterdir()):
    #unzip data.zip to create root_data_dir
    print("Root data directory is missing or empty. Extracting data.zip...")
    with zipfile.ZipFile("data.zip", 'r') as zip_ref:
        zip_ref.extractall()

# Directory containing the training images
training_data_dir = Path("data/captcha_images_v5000/")

# Get list of all the images
images = sorted(list(map(str, list(training_data_dir.glob("*.jpg")))))
labels = [img.split(os.path.sep)[-1].split(".")[0] for img in images]

# check skew
if show_plots:
    counts = Counter(c for label in set(labels) for c in label)
    # Plot frequency of characters in the dataset
    chars, freqs = zip(*sorted(counts.items()))
    plt.bar(chars, freqs, color='skyblue')
    plt.title('Character Frequency')
    plt.xlabel('Character')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Extract unique characters from labels
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)

# Factor by which the image is going to be downsampled
# by the convolutional blocks. We will be using two
# convolution blocks and each block will have
# a pooling layer which downsample the features by a factor of 2.
# Hence total downsampling factor would be 4.
downsample_factor = 4

# Maximum length of any captcha in the dataset
max_length = max([len(label) for label in labels])

# Generating the mapping for characters to integers and vice versa
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

### PREPROCESSING ###

# Function to split the dataset into training and validation sets
# AT: Facilitating Shuffling and Splitting: Instead of directly shuffling large arrays of images or 
# complex data structures (which can be memory-intensive or computationally expensive), 
# you shuffle the much smaller and simpler indices array.
def split_data(images, labels, train_size=0.9, shuffle=True):
    # Get the total size of the dataset
    size = len(images)
    # Make an indices array and shuffle to ensure randomness
    # Instead of directly shuffling large arrays of images or complex data 
    # structures (which can be memory-intensive or computationally expensive), 
    # you shuffle the much smaller and simpler indices array.
    indices = ops.arange(size)
    if shuffle: indices = keras.random.shuffle(indices)
    # Get the size of training samples
    train_samples = int(size * train_size)
    # Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid

def encode_single_sample(img_path, label):
    # 1. Read image and convert to grayscale
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=1)

    # Normalizing the image to [0.0, 0.1] range for better computation efficiency
    img = tf.image.convert_image_dtype(img, tf.float32)

    # By transposing the image, the width becomes the time dimension 
    # that gets passed into the sequence model (like GRU or LSTM), 
    # and each "time step" corresponds to a vertical slice of the image.
    #   Original shape: (height, width, channels) → [0, 1, 2]
    #   After transpose: (width, height, channels) → [1, 0, 2]
    img = ops.transpose(img, axes=[1, 0, 2])

    # Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

    # Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}

### PREPARE DATASETS ###

# Split training and validation sets
x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

### CONSTRUCT MODEL ###

def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    label_length = ops.cast(ops.squeeze(label_length, axis=-1), dtype="int32")
    input_length = ops.cast(ops.squeeze(input_length, axis=-1), dtype="int32")
    sparse_labels = ops.cast(
        ctc_label_dense_to_sparse(y_true, label_length), dtype="int32"
    )

    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())

    return ops.expand_dims(
        tf.compat.v1.nn.ctc_loss(
            inputs=y_pred, labels=sparse_labels, sequence_length=input_length
        ),
        1,
    )


def ctc_label_dense_to_sparse(labels, label_lengths):
    label_shape = ops.shape(labels)
    num_batches_tns = ops.stack([label_shape[0]])
    max_num_labels_tns = ops.stack([label_shape[1]])

    def range_less_than(old_input, current_input):
        return ops.expand_dims(ops.arange(ops.shape(old_input)[1]), 0) < tf.fill(
            max_num_labels_tns, current_input
        )

    init = ops.cast(tf.fill([1, label_shape[1]], 0), dtype="bool")
    dense_mask = tf.compat.v1.scan(
        range_less_than, label_lengths, initializer=init, parallel_iterations=1
    )
    dense_mask = dense_mask[:, 0, :]

    label_array = ops.reshape(
        ops.tile(ops.arange(0, label_shape[1]), num_batches_tns), label_shape
    )
    label_ind = tf.compat.v1.boolean_mask(label_array, dense_mask)

    batch_array = ops.transpose(
        ops.reshape(
            ops.tile(ops.arange(0, label_shape[0]), max_num_labels_tns),
            tf.reverse(label_shape, [0]),
        )
    )
    batch_ind = tf.compat.v1.boolean_mask(batch_array, dense_mask)
    indices = ops.transpose(
        ops.reshape(ops.concatenate([batch_ind, label_ind], axis=0), [2, -1])
    )

    vals_sparse = tf.compat.v1.gather_nd(labels, indices)

    return tf.SparseTensor(
        ops.cast(indices, dtype="int64"),
        vals_sparse,
        ops.cast(label_shape, dtype="int64")
    )


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
        input_length = ops.cast(ops.shape(y_pred)[1], dtype="int64")
        label_length = ops.cast(ops.shape(y_true)[1], dtype="int64")

        input_length = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * ops.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

def build_model():
    # Inputs to the model
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs (TODO: may not be needed since the characters do not overlap much
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model

### TRAIN MODEL ###

# Construct the model
model = build_model()
model.summary()

# epochs = 150
# early_stopping_patience = 10

epochs = 30
early_stopping_patience = 5

# TESTING ONLY
# epochs = 1
# early_stopping_patience = 1

# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping],
)

if show_plots:
    # Plot training and validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('CTC Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

### INFERENCE ###
def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    input_shape = ops.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())
    input_length = ops.cast(input_length, dtype="int32")

    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length
        )
    else:
        (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
        )
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    return (decoded_dense, log_prob)

# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.input[0], model.get_layer(name="dense2").output
)
# prediction_model.summary()

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]

    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

# Function to predict a single image given a file path
def predict_image_from_path(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.where(img > 127, 255, img) #whiten anything that is faint
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = ops.transpose(img, axes=[1, 0, 2])  # Transpose to match model input
    img = ops.expand_dims(img, axis=0)  # Add batch dimension
    pred = prediction_model.predict(img)
    pred_texts = decode_batch_predictions(pred)
    return pred_texts[0]  # Return the first prediction

### EVALUATE ON VALIDATION SET ###

total_val_set = 0
total_val_correct = 0

for batch in validation_dataset:
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)

    # _, ax = plt.subplots(4, 4, figsize=(15, 5))
    # for i in range(len(pred_texts)):
    #     img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
    #     img = img.T
    #     title = f"Prediction: {pred_texts[i]}"
    #     ax[i // 4, i % 4].imshow(img, cmap="gray")
    #     ax[i // 4, i % 4].set_title(title)
    #     ax[i // 4, i % 4].axis("off")

    print("Original texts: ", orig_texts)
    print("Predicted texts: ", pred_texts)

    # Calculate accuracy
    correct = sum(1 for orig, pred in zip(orig_texts, pred_texts) if orig == pred)
    total_val_set += len(orig_texts)
    total_val_correct += correct
    print(f"Validation accuracy for this batch: {correct}/{len(orig_texts)}")
    
print("***********************************") 
print(f"Total validation accuracy: {total_val_correct}/{total_val_set} ({(total_val_correct / total_val_set) * 100:.2f}%)")
print("***********************************") 
# plt.show()

# Function to predict images from a given path and match with labels
def predict_images_from_path(images_dir):
    images_path = Path(images_dir)
    image_paths = sorted(list(map(str, list(images_path.glob("*.jpg")))))
    image_filenames = [img.split(os.path.sep)[-1].split(".")[0] for img in image_paths]

    result_filenames = []
    result_predictions = []

    for img_path, filename in zip(image_paths, image_filenames):
        pred_text = predict_image_from_path(img_path)
        print(f"Image: {img_path}, Predicted: {pred_text}")
        result_filenames.append(filename)
        result_predictions.append(pred_text)

    return result_filenames, result_predictions

def predict_and_output(im_path, save_path):
    if not os.path.exists(im_path):
        raise FileNotFoundError(f"The directory {im_path} does not exist.")
    if not os.path.isdir(im_path):
        raise NotADirectoryError(f"The path {im_path} is not a directory.")        
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    success_result_filenames, result_predictions = predict_images_from_path(im_path)
    for filename, prediction in zip(success_result_filenames, result_predictions):
        txt_filename = filename + '.txt'
        txt_filepath = os.path.join(save_path, txt_filename)
        with open(txt_filepath, 'w') as f:
            f.write(prediction)

# test predict_and_output
# predict_and_output('./data/raw_images', './output')
# predict_and_output('./data/test_images', './output')

### Main Application ###
class Captcha(object):
    def __init__(self):
        print("Captcha generator initialized.")
        pass

    def __call__(self, im_path, save_path):
        print(f"Processing images from {im_path} and saving results to {save_path}...")
        """
        Algo for inference
        args:
            im_path: directory containing .jpg image files
            save_path: output directory to save .txt files with filenames
        """
        predict_and_output(im_path, save_path)

if __name__ == '__main__':
    print("Main function called.")
    parser = argparse.ArgumentParser(description="Predict captcha text from .jpg images and save results as .txt files.")

    parser.add_argument('--im_path', type=str, default='./data/raw_images',
                        help='Path to directory containing .jpg images')
    parser.add_argument('--save_path', type=str, default='./output',
                        help='Path to directory to save .txt files')

    args = parser.parse_args()

    captcha = Captcha()
    captcha(args.im_path, args.save_path)