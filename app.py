import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 1. Ορισμός Διαδρομών
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(BASE_DIR, 'data', 'Cars Dataset', 'train')
test_dir = os.path.join(BASE_DIR, 'data', 'Cars Dataset', 'test')
model_path = os.path.join(BASE_DIR, 'car_classifier_model.h5')

# 2. Προετοιμασία Δεδομένων (Data Generators)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Για το test (validation) κάνουμε μόνο rescale
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# 3. Χτίσιμο του CNN από το μηδέν
model = models.Sequential([
    # Layer 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    
    # Layer 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Layer 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Layer 4
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Classification Head
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5), # Προστασία από overfitting
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# 4. Compile και Callback
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Σταματάει την εκπαίδευση αν το val_loss δεν βελτιωθεί για 5 συνεχόμενες εποχές
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 5. Εκπαίδευση
print("Ξεκινάει η εκπαίδευση...")
history = model.fit(
    train_generator,
    epochs=50, # Βάζουμε 50, αλλά το EarlyStopping θα το σταματήσει νωρίτερα αν χρειαστεί
    validation_data=test_generator,
    callbacks=[early_stop]
)

# 6. Αποθήκευση Μοντέλου
model.save(model.path)
print(f"Το μοντέλο αποθηκεύτηκε στο: {model_path}")

# 7. Οπτικοποίηση Αποτελεσμάτων
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()