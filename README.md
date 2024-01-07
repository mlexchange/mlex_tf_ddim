# mlex_tf_ddim
This code is adapted from an open-source Keras example [https://keras.io/examples/generative/ddim/](https://keras.io/examples/generative/ddim/). 

requirements:  
```
tensorflow	     
matplotlib 
jupyter
numpy  
```

The changelog:

- added capability to diffuse noise (as well as denoise) to an arbitrary level
- support resuming training from the saved checkpoint 
- added a dataloader and preprocessing pipeline 
- added saving options for training history and generated images