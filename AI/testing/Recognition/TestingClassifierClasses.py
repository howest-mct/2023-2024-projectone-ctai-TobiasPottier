import pickle
import numpy as np

# Load the trained classifier
with open('SVM_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Assuming you saved the labels with the model
try:
    with open('labels.npy', 'rb') as f:
        labels = np.load(f)
    print("Labels saved with the model:", labels)
except FileNotFoundError:
    print("No labels file found. Using unique labels from the classifier's training data if available.")
    # Alternative approach if labels were not saved separately
    if hasattr(classifier, 'classes_'):
        print("Classes in the classifier:", classifier.classes_)
