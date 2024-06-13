import cv2
import numpy as np
import joblib
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors
def prepare_data(images):
    features = []
    for image in images:
        descriptors = extract_sift_features(image)
        if descriptors  is not None:
            features.append(descriptors)
    return features
def compute_bovw_representation(features, codebook):
    num_clusters = codebook.n_clusters
    bovw_representation = []

    for image_features in features:
        if len(image_features) > 0:
            # Assign each feature to a cluster
            image_features=image_features.reshape(-1,128)
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