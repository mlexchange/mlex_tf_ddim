# mlex_tf_ddim
This ddim model code (ddim.py) is adapted from an open-source Keras example [https://keras.io/examples/generative/ddim/](https://keras.io/examples/generative/ddim/). 


requirements:  
```
tensorflow	     
matplotlib 
jupyter
numpy
pydantic  
```

The changelog:

- added capability to diffuse noise (as well as denoise) to an arbitrary level
- support resuming training from the saved checkpoint 
- added a dataloader and preprocessing pipeline 
- added saving options for training history and generated images