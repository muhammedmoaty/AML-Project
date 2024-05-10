import numpy as np 
from tqdm import tqdm
import cv2
import os
import imutils
import joblib
import seaborn as sns

from skimage.feature import hog
from skimage import exposure
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

def load_data(directory, use_hog=True):
    images = []
    labels = []
    for class_folder in os.listdir(directory):
        class_path = os.path.join(directory, class_folder)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                if use_hog:
                    # Extract HOG features
                    hog_features = hog(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2))

                    images.append(hog_features)
                # else:
                #     images.append(image.flatten())  # Flatten the image array
                labels.append(class_folder)
    num_features = len(hog_features)
    #print("Number of features:", num_features)
    return np.array(images), np.array(labels)

# Directory containing preprocessed images
cleaned_dir = 'cleaned'

# Load training and testing data with HOG feature extraction
X_train, y_train = load_data(os.path.join(cleaned_dir, 'training'), use_hog=True)
X_test, y_test = load_data(os.path.join(cleaned_dir, 'testing'), use_hog=True)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

k = 9 # no folds
cv_scores = cross_val_score(svm, X_train, y_train, cv=k)

y_pred_train = svm.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)


# Calculate mean accuracy and standard deviation of the cross-validation scores
mean_cv_accuracy = np.mean(cv_scores)
std_cv_accuracy = np.std(cv_scores)

# Calculate accuracy on validation set
y_pred_val = svm.predict(X_val)
validation_accuracy = accuracy_score(y_val, y_pred_val)
#print("Validation Accuracy:", validation_accuracy)

# Calculate accuracy on test set
y_pred_test = svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
#print("Test Accuracy:", test_accuracy)

# Calculate accuracy scores for each fold
accuracy_scores = []
for fold, score in enumerate(cv_scores, start=1):
    accuracy_scores.append(score)


# Plot confusion matrix
conf_matrix_train = confusion_matrix(y_train, y_pred_train)
conf_matrix_val = confusion_matrix(y_val, y_pred_val)
conf_matrix_test = confusion_matrix(y_test, y_pred_test)




# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_test)
# Calculate precision
precision = precision_score(y_test, y_pred_test, pos_label='tumor')
# Calculate recall
recall = recall_score(y_test, y_pred_test, pos_label='tumor')
# Calculate F1-score
f1 = f1_score(y_test, y_pred_test, pos_label='tumor')



def crop_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

   
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    
    return new_img


def preprocess_image(image_path, output_size=256):
    
    image = cv2.imread(image_path)
    if image is None:
        #print("Error: Unable to read the image.")
        return None
    
    processed_image = crop_img(image)
    processed_image = cv2.resize(processed_image, (output_size, output_size))
    
    return processed_image



def predict(image_path):

    processed_image = preprocess_image(image_path)

    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)

    #HOG
    hog_features, hog_image = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)

    # prdect the big big tumor
    prediction1 = svm.predict(hog_features.reshape(1,-1))

    return prediction1[0]

def classify_img(image_path):
    # Load the saved SVM model
    return str(predict(image_path))
