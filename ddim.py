import os, math
import tensorflow as tf
from tensorflow import keras
from keras import layers

import numpy as np
import matplotlib.pyplot as plt

# Adapted from https://keras.io/examples/generative/ddim/

class KID(keras.metrics.Metric):
    def __init__(self, name, params, **kwargs):
        super().__init__(name=name, **kwargs)
        self.params = params
        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=(self.params.image_size[0], self.params.image_size[1], 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=self.params.kid_image_size, width=self.params.kid_image_size),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(self.params.kid_image_size, self.params.kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()


#------ ultility functions for building the NN blocks -------
def sinusoidal_embedding(x, embedding_dims=32, embedding_max_frequency=1000.0):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def get_network(image_size: tuple, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size[0], image_size[1], 3))
    noise_variances = keras.Input(shape=(1, 1, 1))   # variance for both noise and real is 1

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    # build the dowblocks then add 2 residual blocks, then the upblocks
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")  # return a NN as x, and [] as inputs


#------ Denoising diffusion implicit models ---------
class DiffusionModel(keras.Model):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.normalizer = layers.Normalization()
        self.network = get_network(self.params.image_size, self.params.widths, self.params.block_depth)
        self.ema_network = keras.models.clone_model(self.network)
        self.generated_images = None
   
    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss") # average per batch
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss") # average per batch
        self.kid = KID("kid", self.params)

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(self.params.max_signal_rate)
        end_angle = tf.acos(self.params.min_signal_rate)
        
        # diffusion times determine the noise level for each process; 
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)  # when noise_level=0, almost 0
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates
    
    #first train the denoiser, then test it 
    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)  # noise, got from the trained U-Net 
        # when nosie=0, there will be a little bit change
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates   # equation 4 in DDIM paper

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, start_input_percent, diffusion_steps, show_print=False):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]

        #step_size = start_noise_percent / diffusion_steps # seems like it doesnt matter if start_noise_percent**0.5
        scheduler_start_noise = (math.acos(start_input_percent**0.5)-math.acos(self.params.max_signal_rate)) / \
                                (math.acos(self.params.min_signal_rate)-math.acos(self.params.max_signal_rate))
        
        scheduler_start_noise = max(min(scheduler_start_noise, 1.0), 0.0)
        if show_print:
            print(f'scheduler_start_noise level {scheduler_start_noise}')
        step_size = scheduler_start_noise / diffusion_steps # seems like it doesnt matter if start_noise_percent**0.5
        
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1))*scheduler_start_noise - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )

        return pred_images

    def generate(self, num_images, diffusion_steps, input_images=None, start_noise_percent=0.0, show_print=False):
        # noise -> images -> denormalized images
        start_input_percent = 1.0 - start_noise_percent
        rand_noise = tf.random.normal(shape=(num_images, self.params.image_size[0], self.params.image_size[1], 3)) # noise need to have normal distribution 
        if input_images is not None:
            input_images = self.normalizer(input_images)
            initial_noise = start_input_percent**0.5*input_images + start_noise_percent**0.5*rand_noise
            generated_images = self.reverse_diffusion(initial_noise, start_input_percent, diffusion_steps, show_print)
        else:
            generated_images = self.reverse_diffusion(rand_noise, 0.0, diffusion_steps, show_print)
        
        return self.denormalize(generated_images)
    
    @tf.function
    def train_step(self, images):
        # normalize images to have standard deviation of 1
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(self.params.batch_size, self.params.image_size[0], self.params.image_size[1], 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.params.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises # equation 4 in ddim paper 

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training, evarge of bactch
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights) # use noise loss for calculating the gradients
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights)) # backpropagation 

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.params.ema * ema_weight + (1 - self.params.ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}
    
    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(self.params.batch_size, self.params.image_size[0], self.params.image_size[1], 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.params.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small!
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=self.params.batch_size, diffusion_steps=self.params.kid_diffusion_steps
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def generate_images(self, 
                        num_images=1, 
                        input_images=None, 
                        start_noise_percent=0.0,
                        color_mode = 'grayscale', 
                        out_path=None,
                        diffusion_steps = None):
        """
            Generate and show images from the trained diffusion model.
            Args:
                num_images, int:                number of images to generate, default=1
                input_images, tensor/array:     the images to diffuse noise and denoise
                start_noise_percent, float:     noise level [0.0, 1.0], defalt=0.0
                color_mode, str:                'rgb' or 'grayscale' (default)
                out_path, str:                  If given, the generated images will be saved to the directory.
                diffusion_steps, int:           if not given, the diffusion step defined in modelConfig will be used.

        """
        
        steps = diffusion_steps if diffusion_steps is not None else self.params.plot_diffusion_steps
        show_noise_level = True if input_images is not None else False
        generated_images = self.generate(
            num_images=num_images,
            diffusion_steps=steps,
            input_images = input_images,
            start_noise_percent=start_noise_percent,
            show_print = show_noise_level
        )
        if out_path is not None:
            for i in range(num_images):
                if color_mode == 'grayscale':
                    im = tf.image.rgb_to_grayscale(generated_images[i])
                self.show_images(input_images=im)
                tf.keras.utils.save_img(f'{out_path}/image_{round(start_noise_percent*100)}pcnt_noise_{i}.png', im)
    
    
    def show_images(self, 
                    input_images=None, 
                    color_mode='grayscale', # or 'rgb'  
                    num_rows=1, 
                    num_cols=1, 
                    out_path=None):
        """ 
            Plot (and/or save) the generated images. The number of input_images must be no smaller than num_rows*num_cols.
            Args: 
                input_images, tensor/array:   If given, only plots the input_images. \
                                              Otherwise, generates num_rows*num_cols number of new images and plot these images.
                num_rows, int:                number of rows to display the images, default=1
                num_cols, int:                number of columns to display the images, default=1
                color_mode, str:              'rgb' or 'grayscale' (default)
                output_dir, str:              If given, the generated images will be saved to the directory. 
        """
        N = num_rows*num_cols

        if input_images is not None:
            # input_images will be converted to 4-d tensor (N, W, H, C)
            generated_images = input_images if len(input_images.shape) == 4 else tf.expand_dims(input_images, axis=0)
            assert generated_images.shape[0] >= N, f'{N} images to plot, which is larger than the number of input images' 
        
        else:
            generated_images = self.generate(
                num_images=num_rows * num_cols,
                diffusion_steps=self.params.plot_diffusion_steps,
            )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                im = generated_images[index]
                if color_mode == 'grayscale' and im.shape[-1] == 3: 
                    im = tf.image.rgb_to_grayscale(im)
                
                if out_path is not None:
                    tf.keras.utils.save_img(f'{out_path}/image_{index}.png', im)
                plt.imshow(im)
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()