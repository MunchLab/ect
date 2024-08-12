import torch
import matplotlib.pyplot as plt
import os
import pandas as pd 
from sklearn.metrics import confusion_matrix, accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import LogNorm
import seaborn as sn
import numpy as np
import random

plt.style.use('ggplot')

# Make outputs directory

if not os.path.exists('outputs'):
    os.makedirs('outputs')



def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

class SaveBestModel:
    """
    A class that saves the best model during training by comparing current epoch validation loss to the existing lowest validation loss.
    Adapted from https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
    """
    def __init__(
        self, best_valid_loss=float('inf') #initialize to infinity so the model loss must be less than existing lowest valid loss
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch}\n")
            #save model 
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                         'loss': criterion,}, 'outputs/best_model.pth')
            

def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model. 
    Adapted from https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/final_model.pth')
    

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Max validation accuracy: '+ "%.2f%%" % (np.max(valid_acc)))
    plt.savefig('outputs/accuracy.png',dpi=1000, bbox_inches='tight')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png',dpi=1000, bbox_inches='tight')

def save_cf(y_pred,y_true,classes):
    """
    Function to save the confusion matrix plots.
    """
    cf_matrix = confusion_matrix(y_true, y_pred, normalize = 'true')
    

    #df_norm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(cf_matrix, annot=True, fmt=".3f", cmap = 'Blues', norm=LogNorm(),xticklabels=classes,yticklabels=classes)
    plt.xlabel('Predicted Label', weight='bold')
    plt.ylabel('True Label', weight='bold')
    plt.savefig('outputs/cf_norm_logscale.png', dpi=1000, bbox_inches='tight')

    # SAVE THE CLF REPORT   
    clf_report = pd.DataFrame(classification_report(y_true,y_pred, output_dict=True))
    clf_report.to_csv('outputs/outputCLFreport.csv')
    print("Test Result:\n================================================")        
    print(f"Accuracy Score: {accuracy_score(y_true,y_pred,) * 100:.2f}%")
    print("_______________________________________________")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")
    print("_______________________________________________")
    print(f"Confusion Matrix: \n {confusion_matrix(y_true,y_pred,)}\n")


## TRANSFORMS ####
    
class ROTATE_TRANSLATE(object):
    """Transform the ECT matrix by randomly rotating the input image (randomly select a rotation angle) and translating the columns of the ECT image accordingly.
    """

    def __call__(self, sample):
        # get random rotation angle, use to determine how many columns to shift
        r_angle = random.uniform(0, 2*np.pi)

        cols = np.linspace(0,2*np.pi, 32)
        col_idx = min(range(len(cols)), key=lambda i: abs(cols[i]-r_angle))
        #Translate columns of image according to the random angle
        first = sample[:,col_idx:,:]
        second = sample[:,0:col_idx,:]
        new_image = torch.concatenate((first,second), axis=1)


        """         plt.style.use('default')
        fig, axes = plt.subplots(figsize=(10,5), ncols=2)
        axes[0].imshow(sample[0,:,:], cmap='gray')
        axes[1].imshow(new_image[0,:,:], cmap='gray')
        plt.show()  """   
        return new_image

    