
import warnings
from timeit import default_timer as timer

from tensorflow.keras.callbacks import Callback


class MyCallback(Callback):
    def __init__(self, monitor='val_loss', value=50, verbose=1, patience=1000):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.patience=patience  
        self.starttime = timer()
        self.best_weights = None # Save the best weights across all folds 
        
    def on_train_begin(self, logs=None):
        # El numero de epoch que ha esperado cuando la perdida ya no es minima.
        self.wait = 0
        # Initialize el best como infinito.
        self.best = 0 #np.Inf

    def on_epoch_end(self, epoch, logs={}):
        if epoch%100 == 0:
            elapsed_time = float(timer()-self.starttime)
            self.starttime = timer()
            print('Epoch {}/{} - Elapsed_time: {:3.2f} s - Train loss: {:7.2f} - Train ACC {:7.2f} - Train AUC {:7.2f} - Val loss: {:7.2f} - Val ACC: {:7.2f} - Val AUC {:7.2f}.'.format(
                epoch+1, self.params.get('epochs'), elapsed_time, logs['loss'], logs['accuracy'], logs['auc'], logs['val_loss'], logs['val_accuracy'], logs['val_auc']))
            
        current = logs.get(self.monitor)   
        if current > self.best:
            self.best = current
            # Guardar los mejores pesos si el resultado actual es mejor (menos).
            self.best_weights = self.model.get_weights()
        
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current > self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
                
            elapsed_time = float(timer()-self.starttime)
            print('Epoch {}/{} - Elapsed_time: {:3.2f} s - Train loss: {:7.2f} - Train ACC {:7.2f} - Train AUC {:7.2f} - Val loss: {:7.2f} - Val ACC: {:7.2f} - Val AUC {:7.2f}.'.format(
                epoch+1, self.params.get('epochs'), elapsed_time, logs['loss'], logs['accuracy'], logs['auc'], logs['val_loss'], logs['val_accuracy'], logs['val_auc']))
            
            self.model.stop_training = True   
            
        if ((epoch+1) == self.params.get('epochs')) and (epoch%100 != 0):
            elapsed_time = float(timer()-self.starttime)
            print('Epoch {}/{} - Elapsed_time: {:3.2f} s - Train loss: {:7.2f} - Train ACC {:7.2f} - Train AUC {:7.2f} - Val loss: {:7.2f} - Val ACC: {:7.2f} - Val AUC {:7.2f}.'.format(
                epoch+1, self.params.get('epochs'), elapsed_time, logs['loss'], logs['accuracy'], logs['auc'], logs['val_loss'], logs['val_accuracy'], logs['val_auc']))
    # Restore the best weights once training is finished       
    def on_train_end(self, epoch, logs={}):
        print('Restaurando los pesos del modelo del final de la mejor epoch.')
        self.model.set_weights(self.best_weights)       
