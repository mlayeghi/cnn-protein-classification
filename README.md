## A 3-layer CNN model to classify protein sequences - in Python & TensorFlow

 
A Convolutional Neural Network (CNN) model to classsify protein sequences. It takes a csv file containig at least two columns: sequence (protein sequences) and label (what protein family or cluster the sequence belongs to) as input, splits the data into 80% training : 20% test, trains the model, classifies the protein sequences in the test dataset, and finally spits out the train & test accuracy measures. Users can tune the model hyperparameters, such as the number of layers, number of neurons in each layer, dropout value, learning rate, etc., in the main function.

It also plots a confusion matrix and a precision-recall curve for each epoch, when an improvement is seen in the test accuracy. Moreover, the whole model, with all the hyperparameters, their distributions, results (accuracy, errors, etc.), will be saved and could be be restored later for more trainig or predictions. User can also view/visualize the model and its details in Tensorboard.


## Requirements

- Python 3
- Tensorflow > 1.0.1

## Training

```
python cnn_classifier_3l.py
```

## Tensorboard visualization

```
tensorboard --logdir=/tmp/model_log
```
Then open http://127.0.1.1:6006/ in your web browser.

## Help

```
python cnn_classifier_3l.py -h
```
