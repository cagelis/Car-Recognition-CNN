import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'car_classifier_model.h5')
test_dir = os.path.join(BASE_DIR, 'data', 'Cars Dataset', 'test')

model = tf.keras.models.load_model(model_path)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print("Model evaluation on test set in progress. Please wait...")
Y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred_classes)
class_names = list(test_generator.class_indices.keys())

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Car Recognition')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred_classes, target_names=class_names))
