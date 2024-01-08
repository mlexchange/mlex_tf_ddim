# mlex_tf_ddim
This is a cookbook for training a Denoising Diffusion Implicit Model (DDIM) from scratch and inferencing new images from the trained weights. The diffusion model code (ddim.py) is adapted from an open-source Keras example [https://keras.io/examples/generative/ddim/](https://keras.io/examples/generative/ddim/). 


**Software requirements:**  tensorflow, matplotlib, jupyter, numpy, pydantic  


**New features:**

- added capability to diffuse noise (and denoise) to (from) an arbitrary level
- support resuming training from the saved checkpoint 
- added a dataloader and preprocessing pipeline 
- support arbitrary image size ratio  
- added saving options for training history and the generated images
