import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


normalizedTrainingSet = ImageDataGenerator(rescale=1 / 255)
normalizedTestingSet = ImageDataGenerator(rescale=1 / 255)


trainingClass = normalizedTrainingSet.flow_from_directory("DataSet/Training",
                                                          target_size=(100, 100),
                                                          class_mode="categorical",
                                                          shuffle=True)

testingClass = normalizedTrainingSet.flow_from_directory("DataSet/Testing",
                                                         target_size=(100, 100),
                                                         class_mode="categorical",
                                                         shuffle=False)

print(trainingClass.class_indices)
print(testingClass.class_indices)

model = tf.keras.models.Sequential \
        ([
        tf.keras.layers.Conv2D(8, (3, 3), activation="relu", input_shape=(100, 100, 3)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(8, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

sampleSize = 1104
epochs = 32

monitor = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, verbose=1
                                           , restore_best_weights=True)
fittedModel = model.fit(trainingClass,
                        epochs=epochs,
                        callbacks=[monitor],
                        validation_data=testingClass,
                        shuffle=True,
                        )

# model1: 0.4667
# model2: 0.5259
# model3: 0.5550
# model3GrayScale: 0.53
model.save('newModel3')
