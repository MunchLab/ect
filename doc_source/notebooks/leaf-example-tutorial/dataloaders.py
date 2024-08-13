import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
os.getcwd()
from utils import ROTATE_TRANSLATE

BATCH_SIZE = 4
VALID_SPLIT = 0.2
NUM_WORKERS = 0





#Dataset class for numpy array data (ECT, SECT)
class NPYDataset(Dataset):
    def __init__(self, data, target, classes, transform=None):
        #self.data = torch.from_numpy(data).float()
        self.data = data
        self.target = torch.from_numpy(target).long()
        #self.target = torch.from_numpy(target).float()
        self.transform = transform
        self.num_classes = len(classes)
        self.classes = classes
        
    def __getitem__(self, index):
        
        x = self.data[index]
        #x = x.unsqueeze(0)
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        #x = x.unsqueeze(0)
        return x, y
    
    def __len__(self):
        return len(self.data)
    


# Transforms included in utils.py
def rescale(arr):
    new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
    return new_arr
    
# function to create the datasets
def create_datasets():
    """
    Function to load in images (ect or outline) to dataset
    """

    image_path = os.getcwd()+'/data/ect/'
    print(image_path)
    # Dict to map class names to indices
    classes = []
    for (dirpath, dirnames, filenames) in os.walk(image_path):
        print(dirnames)
        classes.extend(dirnames)
        break
    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}
    num_classes = len(classes)
    print('num_classes=',num_classes)


    numpy_data=[]
    numpy_target=[]
    classes_count=[0 for c in classes]
    for path, subdirs, files in os.walk(image_path):
        
        files = [f for f in files if not f[0] == '.']
        subdirs[:] = [d for d in subdirs if not d[0] == '.']
        
        for name in files:
            input_filedir = os.path.join(path, name)
            image = rescale(np.load(input_filedir))

            splitpath = os.path.normpath(input_filedir).split(os.path.sep)
            label = list(set(splitpath).intersection(classes))[0]
            
            #if classes_count[class_to_idx[label]]< 50:
            if True:
                numpy_data.append(image)
                numpy_target.append(class_to_idx[label])
                classes_count[class_to_idx[label]]+=1
                
            
    numpy_data = np.array(numpy_data)
    numpy_target = np.array(numpy_target)
   
    train, valid_data, y_train, y_valid = train_test_split(numpy_data, numpy_target, test_size=VALID_SPLIT, shuffle=True)

    print('ECT data; using only normalize, rotation transforms on training data')
    train_data = NPYDataset(train, y_train, classes, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5), ROTATE_TRANSLATE()]))   
    valid_data = NPYDataset(valid_data, y_valid, classes, transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5), ROTATE_TRANSLATE()]))

    return train_data, valid_data







def create_data_loaders(train_data, valid_data):
    """
    Function to build the data loaders.
    Parameters:
    :param train_data: The training dataset.
    :param valid_data: The validation dataset.
    """

    train_loader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    valid_loader = DataLoader(dataset = valid_data, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)    

    return train_loader, valid_loader

