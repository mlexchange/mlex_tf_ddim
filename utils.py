class TrainCallbacks:
    def __init__(self, model=None, log_fn=None, n=5):
        self.model = model
        self.log_fn = log_fn 
        self.n = n
    
    def fn_epoch_end(self, epoch, logs):
        if self.log_fn is not None:
            self.log_fn.write(f'n_loss: {logs["n_loss"]}\ti_loss: {logs["i_loss"]}\n')
        
        if self.model is not None and epoch%self.n == 0:
            self.model.show_images()

    def fn_train_end(self):
        if self.log_fn is not None:
            self.log_fn.close() 

        if self.model is not None:
            self.model.show_images() 