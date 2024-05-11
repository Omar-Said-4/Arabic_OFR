from preprocess import *
from LPQ import *
from classifiers_manager import *
import os
import datetime

if __name__ == "__main__":
    with open('results.txt', 'w') as file:
        pass

    with open('time.txt', 'w') as file:
        pass  

    classifiers_manager = ClassifiersManager()
    classifiers_manager.load_model(Classifiers.svm)

    folder = os.path.join(os.getcwd(), "data")
    files = [int(os.path.splitext(file)[0]) for file in os.listdir(folder) if file.endswith('.jpeg')]
    files.sort()
    for i, filename in enumerate(files):
        img = cv2.imread(os.path.join(folder, str(filename)+".jpeg"), cv2.IMREAD_GRAYSCALE)

        start_time = datetime.datetime.now()
        processed_img = preprocess_image(img)

        # Extract LPQ features
        lpq_features = lpq(processed_img)
        lpq_features = np.array(lpq_features).reshape(1, -1)

        predicted_label= classifiers_manager.predict(Classifiers.svm, lpq_features)

        end_time = datetime.datetime.now()
        time_difference = end_time - start_time
        time_difference_seconds = round(time_difference.total_seconds(), 3)

        with open('results.txt', 'a') as file:
            file.write(str(predicted_label[0]) + '\n')

        with open('time.txt', 'a') as file:
            file.write("{:.3f}\n".format(time_difference_seconds))