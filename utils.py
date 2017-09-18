import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn import preprocessing

def load_data(infile, input_type='sequence'):
    '''
    Load & prepare fasta protein sequences.
    '''
    # Loading the protein fasta sequences; csv file containing sequences,
    # their labels, and seuence names
    data = pd.read_csv(infile, header=0, sep=',')

    # Get the set of all letters (single letter amino acid codes) in the data
    char_set = list(set("".join(data[input_type])))
    vocab_size = len(char_set)
    # Creat a dictionary of aino acid letters and their index value in the list.
    # So that we have a numerical code for each letter.
    vocab = dict(zip(char_set, range(1, vocab_size+1)))

    # Embedding the characters using the dictioary built above. Basically,
    # replacing  each character with its numerical code in the dictionary.
    data_embed = pd.DataFrame([list(map(vocab.get, k)) for k in data[input_type]])

    # Replace nan with 0; equivalent of padding with 0
    data_embed[np.isnan(data_embed)] = 0

    # Rename columns
    data_embed.columns = ['S'+str(x+1) for x in range(data_embed.shape[1])]

    # Bin label quantitative values into two classes based on median
    mybins = [data['label'].min()-0.1, data['label'].median(), data['label'].max()]
    labels = np.array(pd.cut(data['label'], bins=mybins, labels=False), dtype=int)

    # Add binned labels column to dataframe
    data_embed = data_embed.assign(label=labels)

    # Convert to float values to int
    data_embed = data_embed.astype(int)

    # Split data into train & test dataset; using a constant random seed for consistency
    train_df, test_df = \
        train_test_split(data_embed, test_size=0.2, random_state=42)

    # Get the embedded train & test sequences as numpy array
    train_seqs = np.array(train_df.drop('label', axis=1))
    test_seqs = np.array(test_df.drop('label', axis=1))

    # Converting categorical labels to actual numbers For two categories of
    # labels, for example, we'll have classes [0, 1].
    train_lebels = np.array(train_df['label'].astype('category').cat.codes)
    test_lebels = np.array(test_df['label'].astype('category').cat.codes)

    return train_seqs, test_seqs, train_lebels, test_lebels

def next_batch(x_data, y_data, batch_size):
    '''
    Returns batches of x & y
    '''
    # Choose a random set of row indices with the size of batch_size
    idx = np.random.choice(np.arange(len(x_data)), size=batch_size, replace=False)
    # Return the subset (batch) of data using the randomly chosen row indices
    return x_data[idx, :], y_data[idx]

def next_rnn_batch(x_data, y_data, seqs_len, batch_size):
    '''
    Returns batches of x & y
    '''
    # Choose a random set of row indices with the size of batch_size
    idx = np.random.choice(np.arange(len(x_data)), size=batch_size, replace=False)
    # Return the subset (batch) of data using the randomly chosen row indices
    return x_data[idx, :], y_data[idx], np.array(seqs_len)[idx]

###########
### Convert labels vector to one-hot matrix
def labels_one_hot(y_data):
    """Return the data, x & y, from the input file"""
    labels = sorted(list(set(y_data.tolist())))
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    y_one_hot = np.zeros(shape=[len(y_data), len(labels)], dtype=int)
    for k, v in label_dict.items(): y_one_hot[y_data==k, :] = v
    return y_one_hot

def plot_confusion_matrix(confmat,
                          classes,
                          plot_name,
                          epoch,
                          normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(confmat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]

    thresh = confmat.max() / 2.
    for i, j in itertools.product(range(confmat.shape[0]), range(confmat.shape[1])):
        plt.text(j, i, confmat[i, j],
                 horizontalalignment="center",
                 color="white" if confmat[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_name + str(epoch+1) + ".png", dpi=150)
    plt.close()

def plot_precrecal_curve(num_classes,
                         labels_test,
                         t_scores,
                         plot_name,
                         epoch):
    ''' Compute Precision-Recall and plot curve'''
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(labels_test[:, i],
                                                            t_scores[:, i])
        average_precision[i] = average_precision_score(labels_test[:, i],
                                                       t_scores[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = \
        precision_recall_curve(labels_test.ravel(), t_scores.ravel())
    average_precision["micro"] = average_precision_score(labels_test,
                                                         t_scores,
                                                         average="micro")
    # Plot Precision-Recall curve for each class
    # setup plot details
    colors = itertools.cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    plt.clf()
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=2,
             label='micro-average Precision-recall curve (area = {0:0.2f})'
             ''.format(average_precision["micro"]))
    for i, color in zip(range(num_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                 ''.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve', fontweight='bold')
    plt.legend(loc="lower right")
    plt.savefig(plot_name + "_epoch-" + str(epoch+1) + ".png", dpi=150)
    plt.close()
