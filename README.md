# mlex\_tf\_ddim
This is a cookbook for training a Denoising Diffusion Implicit Model (DDIM) from scratch and inferencing new images from the trained weights. The diffusion model code (ddim.py) is adapted from an open-source Keras example [https://keras.io/examples/generative/ddim/](https://keras.io/examples/generative/ddim/). 


**Software requirements:**  tensorflow, matplotlib, jupyter, numpy, pydantic  

Install Tensorflow with GPU support on Apple M1/M2, follow [https://github.com/deganza/Install-TensorFlow-on-Mac-M1-GPU/blob/main/Install-TensorFlow-on-Mac-M1-GPU.ipynb](https://github.com/deganza/Install-TensorFlow-on-Mac-M1-GPU/blob/main/Install-TensorFlow-on-Mac-M1-GPU.ipynb)

**New features:**

- added capability to diffuse noise (and denoise) to (from) an arbitrary level
- support resuming training from the saved checkpoint 
- added a dataloader and preprocessing pipeline 
- support arbitrary image size ratio  
- added saving options for training history and the generated images
