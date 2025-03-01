from keras.applications import MobileNetV2
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import os
from get_datasets import get_datasets
from keras.regularizers import l2


def get_model():
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # Başlangıçta sadece son katmanları eğitmek için diğer tüm katmanları dondur
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=["accuracy"])

    return model


def unfreeze_layers(model, num_layers_to_unfreeze):
    """
    Katmanları kademeli olarak açan fonksiyon.
    """
    for layer in model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=["accuracy"])


def train(model, train_dataset, val_dataset, test_dataset, callbacks, batch_size=32, epochs=20):
    # Modeli eğit (ilk başta sadece son katmanları eğit)
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks, batch_size=batch_size)

    # Gövde katmanlarını yavaşça aç
    unfreeze_layers(model, num_layers_to_unfreeze=10)

    # Modeli tekrar eğit (katmanlar açıldıktan sonra)
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks, batch_size=batch_size)

    # Test verisi üzerinde değerlendirme yap
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    return model


def get_callbacks():
    call_backs = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min'),  # Validasyon kaybı izlenerek erken durdurma
        ModelCheckpoint('Model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)  # En iyi modelin kaydedilmesi
    ]
    return call_backs
