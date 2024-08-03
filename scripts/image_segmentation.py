import os
import numpy as np
import skimage as ski
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.stats import mode
import pandas as pd
from scipy.ndimage import uniform_filter
import h5py
from tensorflow.keras.layers import Layer, Conv2D, Concatenate, Activation
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

class SpatialAttentionModule(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttentionModule, self).__init__(**kwargs)

        # Convolutional layer for feature descriptor
        self.convolution = tf.keras.layers.Conv2D(1, (4, 4), padding='same', activation='sigmoid')

    def call(self, inputs):
        # Apply max-pooling and average-pooling along the channel axis
        max_pooled = tf.math.reduce_max(inputs,axis=-1,keepdims=True)
        avg_pooled = tf.math.reduce_mean(inputs,axis=-1,keepdims=True)
        # Concatenate the pooled features
        concatenated_features = Concatenate(axis=-1)([max_pooled, avg_pooled])

        # Apply convolutional layer with Sigmoid activation
        attention_map = self.convolution(concatenated_features)

        # Multiply the input features with the attention map
        output = tf.multiply(inputs, attention_map)

        return output

def recall(predicted, ground_truth):
    true_positive = np.sum(np.logical_and(ground_truth, predicted))
    false_negative = np.sum(np.logical_and(ground_truth, np.logical_not(predicted)))
    if true_positive + false_negative == 0:
        return 0
    
    recall = true_positive / (true_positive + false_negative)
    return recall

def precision(predicted, ground_truth):
    true_positive = np.sum(np.logical_and(ground_truth, predicted))
    false_positive = np.sum(np.logical_and(np.logical_not(ground_truth), predicted))
    if true_positive + false_positive == 0:
        precision = 0.0
    else:
        precision = true_positive / (true_positive + false_positive)
    return precision

def dice(predicted, ground_truth):
    intersection = np.logical_and(ground_truth, predicted)
    dice = (2 * np.sum(intersection)) / (np.sum(ground_truth) + np.sum(predicted))
    return dice

def jaccard(predicted, ground_truth):
    intersection = np.logical_and(ground_truth, predicted)
    union = np.logical_or(ground_truth, predicted)
    
    # Avoid division by zero
    if np.sum(union) == 0:
        return 0.0
    
    return np.sum(intersection) / np.sum(union)

def probability_thresholding_default_zero(probabilities, threshold):
    # Assuming probabilities is a 2D array where each row represents a pixel's probabilities for different classes
    class_labels = np.argmax(probabilities, axis=-1)  # Select the class with the highest probability
    max_probabilities = probabilities.max(axis=-1)

    # Apply threshold to probabilities, defaulting to 0 if no class passes the threshold
    mask = np.where(max_probabilities > threshold, class_labels, 0)

    return mask

def accuracy(predicted, ground_truth):
    # Ensure both arrays have the same shape
    assert ground_truth.shape == predicted.shape, "Arrays must have the same shape"

    # Calculate accuracy
    correct_predictions = np.sum(ground_truth == predicted)
    total_predictions = np.prod(ground_truth.shape)

    accuracy_value = correct_predictions / total_predictions
    return accuracy_value


def illumination_normalization(image,beta): # adapted from https://www.hindawi.com/journals/isrn/2013/516052/
    gmage = np.log(image.astype(np.float32))

    ghat = uniform_filter(gmage,size=7,mode='mirror')

    d = gmage - ghat

    alpha = np.mean(np.abs(d))

    h = np.exp(np.divide(d,alpha*beta))
    o = np.minimum(h,1)

    return o

def reconstruct_image_from_patches(patches, stride, original_shape, agg_mode='mode'):
    # Calculate the dimensions of the reconstructed image
    num_rows, num_cols, num_chan = original_shape
    patch_height, patch_width, patch_depth = patches.shape[1], patches.shape[2], patches.shape[3]
    stride_height, stride_width = map(int,stride)

    # Calculate the maximum number of times a pixel was sampled
    max_samples_height = np.ceil(patch_height / stride_height).astype(np.int16)
    max_samples_width = np.ceil(patch_width / stride_width).astype(np.int16)

    # Initialize a 3D matrix with NaNs
    reconstructed_image = np.empty((num_rows, num_cols, num_chan, max_samples_height * max_samples_width))
    reconstructed_image[:] = np.nan
    print("Reconstructed image: ", reconstructed_image.shape)
    print("Reconstructed image dims: ", patch_height,patch_width,patch_depth)
    vw = ski.util.view_as_windows(reconstructed_image[:,:,:,0],
                                  (patch_height, patch_width, patch_depth),
                                  step=(stride_height,stride_width,patch_depth))
    # Iterate through patches and populate the 3D matrix
    idx = 0
    for i in range(patches.shape[0]):
        print(f"\rReconstructing with patch {i} of {patches.shape[0]}",end='',flush=True)
        col = (i % vw.shape[1]) * stride_width
        row = (i // vw.shape[1]) * stride_height
        
        reconstructed_image[row:row + patch_height, col:col + patch_width, :, idx] = patches[i]
        
        idx += 1
        if idx == max_samples_height * max_samples_width:
            idx = 0
    print(f"\n")

    # Calculate the mode along the third dimension
    if agg_mode == 'mode':
        reconstructed_image = mode(reconstructed_image, axis=3, nan_policy='omit')[0]
    elif agg_mode == 'mean':
        reconstructed_image = np.nanmean(reconstructed_image, axis=3)

    return np.nan_to_num(reconstructed_image, nan=1)

class ImageSegmentation:
    def __init__(self, model_weights_path, mode=None, lr=0.001, defect_properties=None, spatial_attention=False, post_process=False):
        # Load your pretrained UNet model
        self.defect_labels = ['000','002','006','010',
                              '016','019','022','023',
                              '025','027','029','030','036']
        
        #AITEX Dataset
        self.tamX = 4096
        self.tamY = 256
        self.x_start = 0
        self.roi_width = 4096 
        #AVIDataset
        # self.tamX = 2048
        # self.tamY = 1024
        # self.x_start = 130
        # self.roi_width = 1664 
        #AvibadDataset
        # self.tamX = 2048
        # self.tamY = 512
        # self.x_start = 0
        # self.roi_width = 2048 

        if not mode and not model_weights_path:
            Exception('Please input either a model or a preprocessing mode')        
        
        model_image_path = ""
        if model_weights_path:
            if spatial_attention is False:
                if '_sa_' in model_weights_path:
                    spatial_attention = True
                
            self.spatial_attention = spatial_attention
            if self.spatial_attention:
                self.unet = self.initialize_saunet(lr)
                model_image_path = model_weights_path + "_unet_sa_model.png"
            else:
                self.unet = self.initialize_unet(lr)
                model_image_path = model_weights_path + "_unet_model.png"

            if 'v2' in model_weights_path:
                self.mode = 'v2'
            else:
                self.mode = 'v1'

            self.load_pretrained_model(model_weights_path)

            self.input_size = self.unet.input_shape[1:]
            self.output_size = self.unet.output_shape[1:]                 
        else:
            #RUBEN: I added this part to allow for training of the model
            self.input_size =  (64, 64, 6) 
            self.output_size =  (64, 64, 13) 

            self.spatial_attention = spatial_attention
            if self.spatial_attention:
                self.unet = self.initialize_saunet(lr)
                model_image_path = "unet_sa_model.png" 
            else:
                self.unet = self.initialize_unet(lr)
                model_image_path = "unet_model.png" 
            # self.input_size = self.unet.input_shape[1:]
            # self.output_size = self.unet.output_shape[1:]           

            self.mode=mode

        if post_process:
            self.post_process = self.initialize_cnn_pp(lr)

        # Generate a plot of the model
        #plot_path = "unet_model.png"
        # plot_model(self.unet, to_file=model_image_path, show_shapes=True, show_layer_names=True)

        # print(f"Model plot saved to {model_image_path}")

        # Sizes of the model input and output
        # self.input_size = self.unet.input_shape[1:]
        # self.output_size = self.unet.output_shape[1:]
        # print("input shape:", self.unet.input_shape)
        # print("input_size:", self.input_size)
        # print("output shape:", self.unet.output_shape)
        # print("ouput_size:", self.output_size)
        self.threshold = 0.5

        if defect_properties is None:
            self.defect_properties = ('area', 'area_bbox', 'area_convex', 'extent', 'axis_major_length',
                                      'axis_minor_length', 'eccentricity', 'equivalent_diameter_area', 
                                      'intensity_max', 'intensity_mean', 'intensity_min', 'intensity_std',
                                      'perimeter', 'solidity', 'centroid')
        else:
            self.defect_properties = tuple(defect_properties)

    # def initialize_unet(self, lr):
    #     # Define and load your UNet model architecture
    #     encoder = tf.keras.models.Sequential([
    #         tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=(64, 64, 6),
    #                             kernel_initializer='glorot_uniform'),
    #         tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same', input_shape=(64, 64, 6),
    #                             kernel_initializer='glorot_uniform'),
    #         tf.keras.layers.Dropout(0.2),  # Add Dropout
    #         tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform'),
    #         tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform'),
    #         tf.keras.layers.Dropout(0.2),
    #         tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform'),
    #         tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform'),
    #     ])

    #     encoder.build(input_shape=(None, 64, 64, 6))
    #     # dummy_input = tf.keras.layers.Input(shape=(64, 64, 6))  
    #     # x = encoder.output
    #     # Crear una entrada ficticia
    #     # dummy_input = tf.keras.layers.Input(shape=(64, 64, 6))
    #     # dummy_output = encoder(dummy_input)
    
    #     # x = dummy_output

    #     x = tf.keras.layers.Conv2DTranspose(32,(3,3),strides=(2,2), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform')(x)
    #     x = tf.keras.layers.Concatenate()([x,encoder.layers[6].output])
    #     x = tf.keras.layers.Conv2DTranspose(64,(3,3),strides=(1,1), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform')(x)
    #     x = tf.keras.layers.Conv2DTranspose(64,(3,3),strides=(2,2), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform')(x)
    #     x = tf.keras.layers.Concatenate()([x,encoder.layers[3].output])
    #     x = tf.keras.layers.Conv2DTranspose(128,(3,3),strides=(1,1), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform')(x)
    #     x = tf.keras.layers.Conv2DTranspose(128,(3,3),strides=(2,2), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform')(x)
    #     x = tf.keras.layers.Conv2D(64,(5,5),strides=(1,1), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform')(x)
    #     x = tf.keras.layers.Conv2D(64,(5,5),strides=(1,1), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform')(x)
    #     x = tf.keras.layers.Conv2D(len(self.defect_labels),(1,1),strides=(1,1), activation='softmax', padding='same',
    #                             kernel_initializer='glorot_uniform')(x)
        
    #     # Load the weights
    #     unet = tf.keras.models.Model(inputs=encoder.input, outputs=x)
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    #     unet.compile(optimizer=optimizer,
    #                  loss=tf.keras.losses.CategoricalFocalCrossentropy(),
    #                  metrics=[tf.keras.metrics.CategoricalAccuracy(),
    #                           tf.keras.metrics.CategoricalCrossentropy()])
        
    #     return unet

    def initialize_unet(self, lr):
        # Encoder
        inputs = tf.keras.layers.Input(shape=(64, 64, 6))
        x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(inputs)
        x = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x1)
        x = tf.keras.layers.Dropout(0.2)(x)
        x2 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x2)

        # Decoder
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Concatenate()([x, x2])
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Concatenate()([x, x1])
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        outputs = tf.keras.layers.Conv2D(len(self.defect_labels), (1, 1), strides=(1, 1), activation='softmax', padding='same', kernel_initializer='glorot_uniform')(x)

        # Load the weights
        unet = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        unet.compile(optimizer=optimizer,
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=[tf.keras.metrics.CategoricalAccuracy(),
                            tf.keras.metrics.CategoricalCrossentropy()])
        
        # Generate a plot of the model
        plot_path = "unet_model.png"
        plot_model(unet, to_file=plot_path, show_shapes=True, show_layer_names=True)

        print(f"Model plot saved to {plot_path}")

        return unet

    # def initialize_saunet(self, lr):
    #     # Define and load your UNet model architecture
    #     encoder = tf.keras.models.Sequential([
    #         tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=(64, 64, 6),
    #                             kernel_initializer='glorot_uniform'),
    #         tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same', input_shape=(64, 64, 6),
    #                             kernel_initializer='glorot_uniform'),
    #         tf.keras.layers.Dropout(0.2),  # Add Dropout
    #         tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform'),
    #         tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform'),
    #         tf.keras.layers.Dropout(0.2),
    #         tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform'),
    #         tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform'),
    #     ])

    #     x = encoder.output
    #     x = tf.keras.layers.Conv2DTranspose(32,(3,3),strides=(2,2), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform')(x)
    #     x = tf.keras.layers.Concatenate()([x,encoder.layers[6].output])
    #     x = tf.keras.layers.Conv2DTranspose(64,(3,3),strides=(1,1), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform')(x)
    #     x = tf.keras.layers.Conv2DTranspose(64,(3,3),strides=(2,2), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform')(x)
    #     x = tf.keras.layers.Concatenate()([x,SpatialAttentionModule()(encoder.layers[3].output)])
    #     x = tf.keras.layers.Conv2DTranspose(128,(3,3),strides=(1,1), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform')(x)
    #     x = tf.keras.layers.Conv2DTranspose(128,(3,3),strides=(2,2), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform')(x)
    #     x = tf.keras.layers.Conv2D(64,(5,5),strides=(1,1), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform')(x)
    #     x = tf.keras.layers.Conv2D(64,(5,5),strides=(1,1), activation='relu', padding='same',
    #                             kernel_initializer='glorot_uniform')(x)
    #     x = tf.keras.layers.Conv2D(len(self.defect_labels),(1,1),strides=(1,1), activation='softmax', padding='same',
    #                             kernel_initializer='glorot_uniform')(x)
        
    #     # Load the weights
    #     unet = tf.keras.models.Model(inputs=encoder.input, outputs=x)
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    #     unet.compile(optimizer=optimizer,
    #                  loss=tf.keras.losses.CategoricalFocalCrossentropy(),
    #                  metrics=[tf.keras.metrics.CategoricalAccuracy(),
    #                           tf.keras.metrics.CategoricalCrossentropy()])
        
    #     return unet


    def initialize_saunet(self, lr):
        inputs = tf.keras.layers.Input(shape=(64, 64, 6))
        # Encoder
        x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(inputs)
        x = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv3)
        x = tf.keras.layers.Dropout(0.2)(x)
        conv6 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv6)

        # Decoder
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Concatenate()([x, conv6])
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Concatenate()([x, SpatialAttentionModule()(conv3)])
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
        #outputs = tf.keras.layers.Conv2D(3, (1, 1), strides=(1, 1), activation='softmax', padding='same', kernel_initializer='glorot_uniform')(x)
        outputs = tf.keras.layers.Conv2D(len(self.defect_labels), (1, 1), strides=(1, 1), activation='softmax', padding='same', kernel_initializer='glorot_uniform')(x)

        unet = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        # Crear el modelo
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        unet.compile(optimizer=optimizer,
                    loss=tf.keras.losses.CategoricalFocalCrossentropy(),
                    metrics=[tf.keras.metrics.CategoricalAccuracy(),
                            tf.keras.metrics.CategoricalCrossentropy()])

        #https://graphviz.org/download/
        #pip install pydot
        #pip install graphviz

        # Generate a plot of the model
        plot_path = "unet_model_spatialattention.png"
        plot_model(unet, to_file=plot_path, show_shapes=True, show_layer_names=True)

        print(f"Model plot saved to {plot_path}")

        return unet

    def load_pretrained_model(self, model_weights_path):
        
        self.unet.load_weights(model_weights_path)

        return

    def initialize_cnn_pp(self, lr):

        post_process = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',
                                   padding='same', input_shape=(16,256,19)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',
                                   padding='same'),
            tf.keras.layers.Conv2D(13, (1,1), activation='softmax',
                                   padding='same')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        post_process.compile(optimizer=optimizer,
                             loss=tf.keras.losses.CategoricalFocalCrossentropy(),
                             metrics=[tf.keras.metrics.CategoricalAccuracy(),
                                      tf.keras.metrics.CategoricalCrossentropy()])

        return post_process
        

    def load_image(self, image_path):
        # Load an image from the given path
        image = ski.io.imread(image_path)
        print("Dimensiones:", image.ndim)
        print("Type of image:", type(image))
        print("Shape of image:", image.shape)
        print("Data type of image:", image.dtype)

        if len(image.shape) > 2:
            image = image[:, :, 0]

        if image.shape[0] < self.tamY:
            image = np.pad(image,((self.tamY-image.shape[0],0),(0,0)),mode='constant',constant_values=255)

        if image.shape[1] < self.tamX:
            image = np.pad(image,((0,0),(self.tamX-image.shape[1],0)),mode='constant',constant_values=255)

        if image.shape[0] > self.tamY or image.shape[1] > self.tamX:
            raise Exception("Loaded image is larger in at least one of the image dimensions.")

        self.x_end = self.x_start + self.roi_width
        if self.x_end > image.shape[1]:
            self.x_end = image.shape[1]

        # Extraer el ROI
        roi = image[:, self.x_start:self.x_end]
        image = roi
        self.tamX = image.shape[1]

        return image

    def watershed_image(self, image):
        image = ski.filters.gaussian(image,13)
        image = ski.filters.sobel(image)

        peak = np.sum(image,0).argmax()

        #print("Peak:", peak)

        steps = [-50,50]
        marker_start = np.zeros_like(image, dtype=np.int32)
        #print("Marker start:", marker_start)
        for j,step in enumerate(steps):
            #print("Step:", step)
            if(peak+step>self.tamX):
                peak=self.tamX-peak-1
            if(peak+step<0):
                peak=-step+1
            marker_start[:,np.min((np.max((0,peak+step)),image.shape[1]))] = j+1

        # print("Dimensiones:", image.ndim)
        # print("Type of image:", type(image))
        # print("Shape of image:", image.shape)
        # print("Data type of image:", image.dtype)
        # print("Type of marker_start:", type(marker_start))
        # print("Shape of marker_start:", marker_start.shape)
        # print("Data type of marker_start:", marker_start.dtype)

        segmented = ski.segmentation.watershed(image,marker_start)
        return segmented
       
    def generate_feature_images(self, image_path):
        # Load and preprocess a single image
        image = self.load_image(image_path)
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(image, cmap='gray', vmin=0, vmax=255)  
        # ax[0].set_title('Input Image')
                 
        image11 = ski.exposure.rescale_intensity(1.0*image,in_range=(0,255))
        image12 = ski.util.invert(image11)
        if self.mode == 'v1':
            image21 = ski.exposure.equalize_adapthist(image11,16)
            image22 = ski.exposure.equalize_adapthist(image12,16)
            image31 = ski.exposure.adjust_sigmoid(image21)
            image32 = ski.exposure.adjust_sigmoid(image22)
        elif self.mode == 'v2':
            image21 = ski.exposure.equalize_adapthist(image11,16)
            image22 = ski.exposure.equalize_adapthist(image12,16)
            image21 = ski.exposure.adjust_sigmoid(image21)
            image22 = ski.exposure.adjust_sigmoid(image22)
            image31 = illumination_normalization(image,0.5)
            image32 = ski.util.invert(image31)
        
        segmented = self.watershed_image(image11)

        background_mask = np.equal(segmented,1).astype(np.uint8)

        # ax[1].imshow(background_mask)
        # ax[1].set_title('Bg mask')
        # plt.show()        
        
        feature_image = np.concatenate((image11[:,:,np.newaxis],
                                    image21[:,:,np.newaxis],
                                    image31[:,:,np.newaxis],
                                    image12[:,:,np.newaxis],
                                    image22[:,:,np.newaxis],
                                    image32[:,:,np.newaxis]),axis=2)
        
        return feature_image, background_mask

    def split_to_patches(self,image,patch_size,step):
        patches = ski.util.view_as_windows(image, patch_size, step=step)
        if len(patch_size) == 3:
            patches = patches.reshape(-1, patch_size[0], patch_size[1], patch_size[2])
        elif len(patch_size) == 2:
            patches = patches.reshape(-1, patch_size[0], patch_size[1])

        return patches

    def segment_image(self, image_path, grain=8, post_process='argmax'):
        # Split input image into foreground, background and generate feature image
        print(f"Image path: {image_path}")
        feature_image, background_mask = self.generate_feature_images(image_path)

        feature_patches = self.split_to_patches(feature_image,self.input_size,(grain,grain,self.input_size[2]))
        background_patches = self.split_to_patches(background_mask,self.input_size[0:2],(grain,grain))
        
        p = []
        for i,(f,b) in enumerate(zip(feature_patches,background_patches)):
            #print(f.shape)
            #print(b.shape)
            print(f'Patch {i} of {len(feature_patches)}')
            p.append(np.zeros(self.output_size))  
            if np.any(b):
                p[-1][:, :, 0][b==1] = 1  # Update only the 0-index where b is True to 1
                p[-1][:, :, 0][b==0] = np.nan
                #print(p[-1][:,:,0])
                print('Background patch')
            else:
                # Perform segmentation using the loaded model
                o = self.unet.predict(np.expand_dims(f, axis=0))
                p[-1] = o.squeeze()  # Use squeeze to remove dimensions of size 1

        # likely_seg_image = reconstruct_image_from_patches(np.array(p), (grain, grain), 
        #                                                   (256, 4096, 13), agg_mode='mean')
        likely_seg_image = reconstruct_image_from_patches(np.array(p), (grain, grain), 
                                                          (self.tamY, self.tamX, 13), agg_mode='mean')
        
        # plt.imshow(likely_seg_image[ :, :, 0], cmap='gray', vmin=0, vmax=255)
        # plt.show()
        # num_channels = 13
        # cols = 4  # Número de columnas
        # rows = (num_channels + cols - 1) // cols  # Número de filas necesarias
        # fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        # for i in range(num_channels):
        #     row = i // cols
        #     col = i % cols
        #     ax = axes[row, col]
        #     ax.imshow(likely_seg_image[:, :, i], cmap='gray', vmin=0, vmax=255)
        #     ax.set_title(f'Canal {i}')
        #     ax.axis('off')  # Oculta los ejes

        # # Oculta los subplots vacíos si hay
        # for j in range(num_channels, rows * cols):
        #     fig.delaxes(axes.flatten()[j])

        # plt.tight_layout()
        # plt.show()  

        # for i in range(13):
        #     plt.figure(figsize=(5, 5))
        #     plt.imshow(likely_seg_image[:, :, i], cmap='gray', vmin=0, vmax=255)
        #     plt.title(f'Canal {i}')
        #     plt.show()

        if post_process == 'argmax':
            segmented_image = np.argmax(likely_seg_image,axis=-1)
        elif post_process == 'prob_thresh':
            segmented_image = probability_thresholding_default_zero(likely_seg_image,threshold=self.threshold)
        else:
            Exception('Please provide a supported post processing type')
        #print(f'{segmented_image.shape}')

        # plt.imshow(segmented_image, cmap='gray', vmin=0, vmax=255)
        # plt.show()

        return segmented_image, likely_seg_image, feature_image

    def extract_properties(self, component_props, label_props):
        # extracted_properties = {'Label': self.defect_labels[region_label]}
        extracted_properties = {'label': self.defect_labels[int(getattr(label_props, 'intensity_mean', None))]}
        for prop in self.defect_properties:

            if not isinstance(prop, str):
                raise ValueError("Property names in self.defect_properties must be strings.")
        
            # Ensure component_props[0] has an attribute with the given property name
            if not hasattr(component_props, prop):
                if prop == 'intensity_std':
                    extracted_properties[prop] = np.std(component_props.image_intensity[component_props.image])
                else:
                    raise AttributeError(f"Property '{prop}' not found in component_props.")

            extracted_properties[prop] = getattr(component_props, prop, None)

        return extracted_properties
    
    def analyze_segmented_mask(self, segmented_image, image, defect_labels, matching_masks, image_path, skip_bg=False):
        labels = np.unique(segmented_image)
        print(f"Numero de labels: {labels}")
        properties_list = []
        for label in labels:
            if label == 0.0 and skip_bg:
                continue

            connected_components = ski.measure.label(np.equal(segmented_image,label))
            w = ski.measure.regionprops(connected_components, segmented_image)
            regions = ski.measure.regionprops(connected_components, image)
            properties_list = []
            for ww,region in zip(w,regions):
                properties_dict = self.extract_properties(region, ww)
                properties_dict['defect'] = ' '.join(defect_labels)
                properties_dict['masks'] = ' '.join(matching_masks)
                properties_dict['image'] = image_path
                properties_list.append(properties_dict)

        df = pd.DataFrame(properties_list)
        return df

    def prepare_mask(self, image, mask_paths=None, type='label'):
        mask_image = np.zeros_like(image,dtype=np.float32)
        defect_labels = ['000']
        for mask_path in mask_paths:
            loi, defect_labels = map(list, zip(*[(i,label) for i,label in enumerate(self.defect_labels) if f"_{label}_" in mask_path]))
            
            imask_image = self.load_image(mask_path)
            mask_image += float(loi[0])*(imask_image>0).astype(np.float32)
            del imask_image
            #print(np.unique(mask_image))
        
            if len(np.unique(mask_image)) < 2:
                return None, None

        mask_image = mask_image.astype(np.int16)
        if type == 'one-hot':
            mask_image = np.eye(len(self.defect_labels))[mask_image]

        return mask_image, defect_labels

    def evaluate_performance(self, image_path, mask_paths, grain, post_process):
        # Evaluate the model's performance against a given mask
        segmented_image, likely_seg_image, _ = self.segment_image(image_path,grain=grain,post_process=post_process)
        
        mask_image, _ = self.prepare_mask(segmented_image,mask_paths,type='label')
        if isinstance(mask_image,np.ndarray):
            labs = np.union1d(np.unique(mask_image),np.unique(segmented_image))
            def_l = [self.defect_labels[int(lab)] for lab in labs]
            acc = []
            prec = []
            rec = []
            dice_ = []
            iou = []

            for lab in labs:
                S = (segmented_image == lab).astype(np.uint8)
                M = (mask_image == lab).astype(np.uint8)
                a,i,d,p,r = self.compute_performance_metrics(S,M)

                acc.append(a)
                prec.append(p)
                rec.append(r)
                dice_.append(d)
                iou.append(i)

            performance_metrics = pd.DataFrame({'Label':def_l,
                                                'Accuracy':acc,
                                                'Precision': prec,
                                                'Recall': rec,
                                                'Dice': dice_,
                                                'IoU':iou})
        else:
            performance_metrics = None
            print('Ground-truth mask was empty, but a defect is indicated in the image based on the name/directory of the image.\n**No performance metrics were computed**.')


        return performance_metrics, segmented_image, likely_seg_image

    def compute_performance_metrics(self, segmented_image, mask_image):
        acc = accuracy(segmented_image, mask_image)
        iou = jaccard(segmented_image, mask_image)
        dice_ = dice(segmented_image, mask_image)
        prec = precision(segmented_image, mask_image)
        rec = recall(segmented_image, mask_image)
        return acc, iou, dice_, prec, rec


    def training_patch_extraction(self, image_path, mask_paths, grain, max_clean):
        feature_image, background_mask = self.generate_feature_images(image_path)

        mask_image, _ = self.prepare_mask(feature_image[:,:,0],mask_paths)
        if mask_image is None:
            return None, None
        #print(mask_image.shape)
        feature_patches = self.split_to_patches(feature_image,self.input_size,(grain,grain,self.input_size[2]))
        background_patches = self.split_to_patches(background_mask,self.input_size[0:2],(grain,grain))
        mask_patches = self.split_to_patches(mask_image,self.output_size[0:2],(grain,grain))

        # Verificar tamaños de los parches generados
        print("Tamaño de feature_patches:", feature_patches.shape)
        print("Tamaño de background_patches:", background_patches.shape)
        print("Tamaño de mask_patches:", mask_patches.shape)

        del feature_image, background_mask, mask_image
        patch_indices = np.arange(feature_patches.shape[0])
        np.random.shuffle(patch_indices)

        d = []
        p = []
        clean_patches_from_this_image = 0
        for i in patch_indices:
            if i >= len(background_patches):
                print(f"Índice {i} fuera de los límites de background_patches con tamaño {len(background_patches)}")
                continue
            #print(f.shape)
            #print(b.shape)
            #print(f'Patch {i} of {len(feature_patches)}')
            # p.append(np.zeros(self.output_size))  
            if np.any(background_patches[i]):
                #print('Ommitting background patch')
                continue
                #print(p[-1][:,:,0])
            elif np.any(mask_patches[i]>0):
                d.append(np.concatenate((feature_patches[i],mask_patches[i][:,:,np.newaxis]),axis=-1))
            else:
                if clean_patches_from_this_image < max_clean:
                    p.append(np.concatenate((feature_patches[i],mask_patches[i][:,:,np.newaxis]),axis=-1))
                    clean_patches_from_this_image += 1

        return d,p

    def train_unet_model(self, data_path, training_data, validation_data, batch_size=32, epochs=10):
        if not training_data or not validation_data:
            raise ValueError("Training or validation data is not provided.")

        training = h5py.File(training_data)
        #print(training['patches'].shape)
        validation = h5py.File(validation_data)
        # Define the ImageDataGenerator for training data


        # Verificar las dimensiones de los datos y etiquetas de entrenamiento
        train_data_shape = training['patches'][:,:,:,0:6].shape
        train_labels_shape = self.one_hot_encoding(training['patches'][:,:,:,6]).shape
        print("Training data shape:", train_data_shape)
        print("Training labels shape:", train_labels_shape)

        # Verificar las dimensiones de los datos y etiquetas de validación
        val_data_shape = validation['patches'][:,:,:,0:6].shape
        val_labels_shape = self.one_hot_encoding(validation['patches'][:,:,:,6]).shape
        print("Validation data shape:", val_data_shape)
        print("Validation labels shape:", val_labels_shape)

        # Comprobar que las formas son correctas
        if train_data_shape[0] == 0 or val_data_shape[0] == 0:
            raise ValueError("No data found in training or validation datasets.")
        
        train_datagen = ImageDataGenerator(
            # No rescaling needed
            # Randomly flip images horizontally
            horizontal_flip=True,
            # Randomly flip images vertically
            vertical_flip=True,
            # Convert y to one-hot encoding
            #preprocessing_function=self.one_hot_encoding
        )

        # Define the ImageDataGenerator for validation data
        val_datagen = ImageDataGenerator(
            # No rescaling needed for validation data
            # Convert y to one-hot encoding
            #preprocessing_function=self.one_hot_encoding
        )

        # Define the generators for training and validation data
        train_generator = train_datagen.flow(
            x=training['patches'][:,:,:,0:6],
            y=self.one_hot_encoding(training['patches'][:,:,:,6]),
            batch_size=batch_size,
            shuffle=True
        )

        val_generator = val_datagen.flow(
            x=validation['patches'][:,:,:,0:6],
            y=self.one_hot_encoding(validation['patches'][:,:,:,6]),
            batch_size=batch_size,
            shuffle=False  # No need to shuffle validation data
        )

        # Define the path for saving weights after each epoch
        intermediate_dir = f'{data_path}/intermediate_weights'
        os.makedirs(intermediate_dir, exist_ok=True)

        # Define the ModelCheckpoint callback to save weights after each epoch
        if not self.spatial_attention:
            checkpoint_filepath = f'{intermediate_dir}/{self.mode}_weights_epoch_{{epoch:02d}}.weights.h5'
        else:
            checkpoint_filepath = f'{intermediate_dir}/{self.mode}_sa_weights_epoch_{{epoch:02d}}.weights.h5'

        print("Arixus per a guardar progres:", checkpoint_filepath)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            verbose=1
        )

        # Inspeccionar una muestra de los datos del generador
        x_batch, y_batch = next(train_generator)
        print("Batch data shape:", x_batch.shape, "Tipo: ", x_batch.dtype)
        print("Batch labels shape:", y_batch.shape, "Tipo: ", y_batch.dtype)
        # for i in range(5):
        #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        #     ax[0].imshow(x_batch[i, :, :, 0], cmap='gray')  # Visualizando el primer canal
        #     ax[0].set_title('Input Image')
            
        #     ax[1].imshow(np.argmax(y_batch[i], axis=-1), cmap='tab10')  # Visualizando las etiquetas
        #     ax[1].set_title('Label')
            
        #     plt.show()

        self.unet.summary()
        
        # Train the model
        try:
            history = self.unet.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=[model_checkpoint_callback]
            )
        except Exception as e:
            print("Error during model fitting:", e)
            raise

        history_df = pd.DataFrame(history.history)
        if not self.spatial_attention:
            history_df.to_csv(f'{data_path}/{self.mode}_training_history.csv', index=False)
            self.unet.save_weights(f'{data_path}/final_model_{self.mode}.weights.h5')

        else:
            history_df.to_csv(f'{data_path}/{self.mode}_sa_training_history.csv', index=False)
            self.unet.save_weights(f'{data_path}/final_sa_model_{self.mode}.weights.h5')

        # Save the final model


    def train_pp_convnet_model(self, data_path, feature_images, logit_images, mask_images, batch_size=32, epochs=10):
        if not feature_images or not logit_images or not mask_images:
            raise ValueError("Training or validation data is not provided.")

        features = h5py.File(feature_images)
        logits = h5py.File(logit_images) 
        #print(training['patches'].shape)
        masks = h5py.File(mask_images)
        # Define the ImageDataGenerator for training data
        train_datagen = ImageDataGenerator(
            # No rescaling needed
            # Randomly flip images horizontally
            horizontal_flip=True,
            # Randomly flip images vertically
            vertical_flip=True,
            # Convert y to one-hot encoding
            #preprocessing_function=self.one_hot_encoding
        )

        # Define the ImageDataGenerator for validation data
        # val_datagen = ImageDataGenerator(
        #     # No rescaling needed for validation data
        #     # Convert y to one-hot encoding
        #     #preprocessing_function=self.one_hot_encoding
        # )

        # Define the generators for training and validation data
        train_generator = train_datagen.flow(
            x=ski.transform.rescale(np.concatenate((features['patches'][:110],logits['patches'][:110]),axis=-1),(1.0,0.0625,0.0625,1)),
            y=ski.transform.rescale(self.one_hot_encoding(masks['patches'][:110]),(1.0,0.0625,0.0625,1)),
            batch_size=batch_size,
            shuffle=True
        )

        # val_generator = val_datagen.flow(
        #     x=validation['patches'][:,:,:,0:6],
        #     y=self.one_hot_encoding(validation['patches'][:,:,:,6]),
        #     batch_size=batch_size,
        #     shuffle=False  # No need to shuffle validation data
        # )

        # Define the path for saving weights after each epoch
        intermediate_dir = f'{data_path}/intermediate_weights'
        os.makedirs(intermediate_dir, exist_ok=True)

        # Define the ModelCheckpoint callback to save weights after each epoch
        checkpoint_filepath = f'{intermediate_dir}/pp_convnet_weights_epoch_{{epoch:02d}}.weights.h5'
        

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            verbose=1
        )

        # Train the model
        history = self.post_process.fit(
            train_generator,
            epochs=epochs,
            #validation_data=val_generator,
            callbacks=[model_checkpoint_callback]
        )
        history_df = pd.DataFrame(history.history)
    
        history_df.to_csv(f'{data_path}/pp_convnet_training_history.csv', index=False)
        self.post_process.save_weights(f'{data_path}/final_model_weights_pp_convnet.weights.h5')


    # def one_hot_encoding(self, y):
    #     # Convert y to one-hot encoding

    #     # Convert y to one-hot encoding
    #     y = tf.math.greater(y, 0)
    #     y = tf.cast(y, tf.int32)  # Convert to int32
    #     one_hot_y = tf.one_hot(y, depth=len(self.defect_labels), axis=-1)
       
    #     # one_hot_y = tf.one_hot(tf.math.greater(y,0), depth=len(self.defect_labels), axis=-1)
    #     return one_hot_y
    def one_hot_encoding(self, labels):
        labels_np = np.array(labels)
        # Verificar el rango de las etiquetas
        print("Valores mínimos y máximos de las etiquetas:", labels_np.min(), labels_np.max())

        # Inspeccionar una muestra de las etiquetas antes y después de la codificación
        #print("Original labels:", labels[:5])
        encoded_labels = tf.keras.utils.to_categorical(labels, num_classes=13)
        #print("Encoded labels:", encoded_labels[:5])
        return encoded_labels


    def compile_prior_table(self, image_paths, mask_paths):
        dfs = []

        for image_path in image_paths:
            matching_masks = [mask_path for mask_path in mask_paths if os.path.basename(image_path)[:-4] in mask_path]
            if not matching_masks:
                print(f'There are no input masks with matching names for {image_path}. Please double-check your mask input and adjust accordingly.')
                continue
            print(matching_masks)
            feature_image, _ = self.generate_feature_images(image_path)
            mask_image, defect_labels = self.prepare_mask(feature_image[:,:,0], matching_masks)  # Implement your mask preparation logic here
            
            if isinstance(mask_image, np.ndarray):
                df = self.analyze_segmented_mask(mask_image, feature_image[:,:,0:3], defect_labels, matching_masks, image_path, skip_bg=True)
                dfs.append(df)
            else:
                print('Mask had no defects. Skipped in table')

        results = pd.concat(dfs)
        return results
    # def select_optimal_threshold(self, thresholds, metric='iou'):
    #     best_metric_value = 0
    #     best_threshold = 0

    #     for threshold in thresholds:
    #         predicted_masks = self.apply_thresholds(self.generate_probability_scores(), threshold)
    #         iou, _, _, _, _ = self.evaluate_performance(predicted_masks)

    #         if metric == 'iou' and iou > best_metric_value:
    #             best_metric_value = iou
    #             best_threshold = threshold

    #     return best_threshold