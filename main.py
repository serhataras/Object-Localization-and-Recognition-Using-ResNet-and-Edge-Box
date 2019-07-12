import numpy as np
import PIL
from PIL import Image
import cv2
import torch
import sys
import os
import sklearn
from sklearn.svm import SVC
import pickle

sys.path.insert(0, 'data/')
import resnet

def preprocessImage(image): # input is of type PIL Image
    image_arr = np.asarray(image) # convert to a numpy array                            
    org_size = image_arr.shape
    max_dim = max(org_size[0], org_size[1])

    padded_size = (max_dim, max_dim)
    padded_im = Image.new("RGB", padded_size)
    padded_im.paste(image, ((int)((padded_size[0] - org_size[0])/2), (int)((padded_size[1] - org_size[1])/2))) #pastes image in middle of a black image
    padded_im = np.asarray(padded_im)
    res_im = cv2.resize(padded_im, dsize=(224, 224), interpolation=cv2.INTER_LANCZOS4)  # resize to 224x224 using Lanczos interpolation
    res_im = np.asarray(res_im)
   
    res_im = res_im /255
    subs_vals = np.array([0.485, 0.456, 0.406])
    res_im = res_im - subs_vals
    div_vals =  np.array([0.229, 0.224, 0.225])
    res_im = res_im / div_vals
 
    res_im = res_im.astype(np.float32)
    # Part 3.2
    # Feature extraction
    # append a dimension to indicate batch_size, which is one
    res_im = np.reshape(res_im, [1, 224, 224, 3])
    # model accepts input images of size [batch_size, 3, im_height, im_width]
    res_im = np.transpose(res_im, [0, 3, 1, 2])
    # convert the Numpy image to torch.FloatTensor
    res_im = torch.from_numpy(res_im)
    # extract features
    feature_vector = model(res_im)
    # convert the features of type torch.FloatTensor to a Numpy array
    # so that you can either work with them within the sklearn environment
    # or save them as .mat files
    feature_vector = feature_vector.detach().numpy().ravel()
    
    return feature_vector

def intersectionArea(proposal, truth):
    mmdif_x = min(proposal[2], truth[2]) - max(proposal[0], truth[0])
    mmdif_y = min(proposal[3], truth[3]) - max(proposal[1], truth[1])

    if mmdif_x > 0 and mmdif_y > 0:
        return mmdif_x * mmdif_y
    else:
        return 0

def unionArea(proposal, truth):
    return abs(proposal[2] - proposal[0]) * abs(proposal[3] - proposal[1]) + abs(truth[2] - truth[0]) * abs(truth[3] - truth[1]) - intersectionArea(proposal, truth)

def localizationAccuracy(proposal, truth):
    return intersectionArea(proposal, truth) / float(unionArea(proposal, truth))

def confusionMatrix(predictions, truth):
    confusion_matrix = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for i in range(len(predictions)):
        if truth[i] == 1 and predictions[i] == 1: # tp
            confusion_matrix["tp"] = confusion_matrix["tp"] + 1
        elif truth[i] == 0 and predictions[i] == 0: # tn
            confusion_matrix["tn"] = confusion_matrix["tn"] + 1
        elif truth[i] == 0 and predictions[i] == 1: # fp
            confusion_matrix["fp"] = confusion_matrix["fp"] + 1
        elif truth[i] == 1 and predictions[i] == 0: # fn
            confusion_matrix["fn"] = confusion_matrix["fn"] + 1
    return confusion_matrix


# The sample train data, this must be converted to the original directory after all process finished
rootDir = 'data/train/'
model = resnet.resnet50(pretrained=True)
model.eval()

t_class_names = []
t_feature_vectors = []
print("Starting train phase...")
# Part 3.1
# Data Reading
# Getting the training data
# Can be converted to the function
for dirName, subdirList, fileList in os.walk(rootDir):
    # print('Found directory: %s' % dirName)
    class_name = dirName[len(rootDir):]
    for fname in fileList:
        # print('\t%s' % fname) 
        image = Image.open(dirName + '/' + fname).convert('RGB')    

        # apply preprocessing to the image
        feature_vector = preprocessImage(image)
        t_class_names.append(class_name)
        t_feature_vectors.append(feature_vector)

# Part 4: Training one-vs-all classifiers
# Normalize feature vectors
t_feature_vectors = (t_feature_vectors - np.mean(t_feature_vectors, axis = 0, dtype=np.float32)) / np.std(t_feature_vectors, axis = 0, dtype=np.float32)

# Train one-vs-all classifiers for each object type
bsvm = []
unique_labels = np.unique(t_class_names)

for i in unique_labels:
    # Set labels belonging to class i to 1, rest to 0
    new_train_labels = np.asarray(t_class_names)
    new_train_labels[new_train_labels != i] = 0
    new_train_labels[new_train_labels == i] = 1
    # Train binary SVM
    # TODO: Experiment with gamma value
    bsvm_i = SVC(probability=True, random_state=None)
    bsvm_i=bsvm_i.fit(t_feature_vectors, new_train_labels)
    bsvm.append(bsvm_i)

print("Training complete. Starting test phase...")

# Part 5: Testing
rootDirTest = 'data/test/'
edge_detection = cv2.ximgproc.createStructuredEdgeDetection(rootDirTest + "model.yml.gz")
test_predictions = []
object_proposals_predicted = []

for i in range(100):
    print("Test image " + str(i), file=open("output.txt", "a"))

    # 5.1: Create edge boxes for each test image
    image = cv2.imread(rootDirTest + 'images/' + str(i) + '.jpeg')    
    rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(50)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)
    object_proposals_predicted.append(boxes)
    #for b in boxes:
    #    x, y, w, h = b
    #    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    #cv2.imshow("edgeboxes", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()    

    pil_image = Image.open(rootDirTest + 'images/' + str(i) + '.jpeg').convert('RGB')
    test_features_i = []
    for b in boxes:
        x, y, w, h = b    
        subimage_b = pil_image.crop((x, y, x + w, y + h))
        feature_vector = preprocessImage(subimage_b)
        test_features_i.append(feature_vector)

    test_features_i = np.asarray(test_features_i)

    # Normalize features
    test_features_i = (test_features_i - np.mean(test_features_i, axis = 0, dtype=np.float32)) / np.std(test_features_i, axis = 0, dtype=np.float32)

    # 5.2: Localization result
    predictions_i = []
    for j in range(len(bsvm)):
        preds = bsvm[j].predict_proba(test_features_i)        
        predictions_i.append(preds[:, 1])
     
    predictions_i = np.asarray(predictions_i)
    best_prediction = np.unravel_index(np.argmax(predictions_i, axis=None), predictions_i.shape)
    print(best_prediction, file=open("output.txt", "a"))
   
    test_predictions.append(best_prediction)

test_predictions = np.asarray(test_predictions)
object_proposals_predicted = np.asarray(object_proposals_predicted)
print("Testing complete. Starting validation phase...")

# Part 6: Validation
# Read test labels
test_data = []
test_data_file = open("data/test/bounding_box.txt", "r")
for line in test_data_file:
    test_data.append(line.rstrip('\n').split(","))
test_data_file.close()
test_data = np.asarray(test_data)
test_labels = test_data[:, 0]
test_proposals = test_data[:, 1:].astype(np.int)

# Validation statistics 1: Confusion matrix, precision, recall, f-score
# Replace labels with integers corresponding indices in unique_labels array
index = 0
for i in unique_labels:
    test_labels[test_labels == i] = index
    index += 1

test_labels = test_labels.astype(np.int)
test_proposals = np.asarray(test_proposals).astype(np.int)
statistics = []
for i in range(10):
    new_predictions = test_predictions[:, 0].copy()
    new_predictions[new_predictions != i] = -1
    new_predictions[new_predictions == i] = 1
    new_predictions[new_predictions == -1] = 0

    new_truth = test_labels.copy().astype(np.int)
    new_truth[new_truth != i] = -1
    new_truth[new_truth == i] = 1
    new_truth[new_truth == -1] = 0

    confusion_matrix = confusionMatrix(new_predictions, new_truth)
    precision = confusion_matrix["tp"] / float(confusion_matrix["tp"] + confusion_matrix["fp"] + 0.000000001) # to prevent division by zero
    recall = confusion_matrix["tp"] / float(confusion_matrix["tp"] + confusion_matrix["fn"] + 0.000000001) # to prevent division by zero
    f_score = 2 * recall * precision / float(recall + precision + 0.000000001) # to prevent division by zero
    statistics.append({"class_index": i, "cm": confusion_matrix, "precision": precision, "recall": recall, "f": f_score})

# Validation statistics 2: Overall Accuracy & Localization Accuracy
correct_label_count = 0
localization_accuracies = []
for i in range(len(test_predictions)):
    prediction = test_predictions[i]
    label_predicted = prediction[0]
    x, y, w, h = object_proposals_predicted[i, prediction[1]]

    localization_accuracy = localizationAccuracy([x, y, x + w, y + h], test_proposals[i])
    localization_accuracies.append(localization_accuracy)

    if label_predicted == test_labels[i]:
        correct_label_count += 1

overall_accuracy = correct_label_count / float(len(test_predictions))
localization_accuracies = np.asarray(localization_accuracies)
print("Statistics -> CM, Precision, Recall, F-Score: ", file=open("output.txt", "a"))
print(statistics, file=open("output.txt", "a"))
print("Overall Accuracy: ", file=open("output.txt", "a"))
print(overall_accuracy, file=open("output.txt", "a"))
print("Localization Accuracies: ", file=open("output.txt", "a"))
print(localization_accuracies, file=open("output.txt", "a"))
print("Execution completed.")