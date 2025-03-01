import tensorflow as  tf
from keras.preprocessing.image import ImageDataGenerator

def get_datasets(train_path, val_path, test_path) : 

    train_dg = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True )        

    val_dg = ImageDataGenerator(rescale=1./255)
    

    test_dg = ImageDataGenerator(rescale=1./255)
    
    train_dataset = train_dg.flow_from_directory(train_path,
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='binary',
                                                 shuffle=True)
    
    val_dataset = val_dg.flow_from_directory(val_path,
                                             target_size=(224, 224),
                                             batch_size=32,
                                             class_mode='binary')
    
    test_dataset = test_dg.flow_from_directory(test_path,
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode='binary',
                                               shuffle=False)
    
    return train_dataset, val_dataset, test_dataset


