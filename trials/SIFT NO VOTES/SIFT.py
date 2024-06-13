import cv2
import numpy as np
import joblib
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors
def prepare_data(images):
    # Extract SIFT features for each image
    features = [extract_sift_features(image) for image in images]
    
    # Filter out None values
    features = [desc for desc in features if desc is not None]
    
    # Convert the list of arrays into a single numpy array
    if features:
        features = np.vstack(features)
    else:
        features = np.array([])  # Return an empty array if no descriptors found
    return features
def compute_bovw_representation(features, codebook, num_clusters=100):
    bovw_representation = []

    
    if len(features) > 0:
        # Assign each feature to a cluster
        image_features=features.reshape(-1,128)
        cluster_assignments = codebook.predict(image_features)

        # Create a histogram of cluster frequencies
        histogram = np.bincount(cluster_assignments, minlength=num_clusters)

        # Normalize the histogram
        histogram = histogram / np.sum(histogram)

        bovw_representation.append(histogram)
    else:
        # Handle cases where no features were detected
        bovw_representation.append(np.zeros(num_clusters))

    return bovw_representation
def getCodebook():
    # Load the codebook
    codebook = joblib.load('codebook.pkl')

    return codebook