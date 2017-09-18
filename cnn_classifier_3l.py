'''
TensorFlow-CNN 3-layer classifier for protein binding affinity.
'''
import argparse
import time
import os
from itertools import cycle
import tensorflow as tf
import numpy as np
import utils
from cnn_model_3l import CnnModel
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# Parameters
# ==================================================
# Use CPU instead of GPU
#os.environ["CUDA_VISIBLE_DEVICES"]=""

# Data loading params
tf.flags.DEFINE_float("test_percentage", .2, "Percentage of the validation data")
tf.flags.DEFINE_string("data_file", "proteins.csv",
                       "Input data.")
tf.flags.DEFINE_string("input_type", "sequence", "sequence1/2")
tf.flags.DEFINE_string("output", "Results.txt", "Output file")

# Model Hyperparameters
tf.flags.DEFINE_integer("rows", 1, "Number of rows; number of aa indices")
tf.flags.DEFINE_integer("cols", 315, "Number of columns; length of sequence")
tf.flags.DEFINE_integer("example_len", 315, "Length of each example")
tf.flags.DEFINE_integer("example_shape", 315, "Shape of each example")
tf.flags.DEFINE_integer("num_channels", 1, "Number of channels")
tf.flags.DEFINE_string("filter_sizes", "8,6,4", "Filter sizes")
tf.flags.DEFINE_string("num_filters", "64,64,64", "Number of filters per filter size")
tf.flags.DEFINE_integer("fc_size", 750, "Fully connected layer size")
tf.flags.DEFINE_float("keep_prob", 1, "Dropout keep probability")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda")
tf.flags.DEFINE_integer("num_classes", 2, "Number of classes")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 120, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs")
tf.flags.DEFINE_integer("display_step", 5, "Display step")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_string("logdir", "/tmp/model_log", "Logging directory for TensorBoard")
tf.flags.DEFINE_string("resultsdir", "./model_results", "Directory for results & plots")


FLAGS = tf.flags.FLAGS

# Data Preparation
# ==================================================

def main():
    """Return the data, x & y, from the input file"""
    # Load data
    print("\nLoading data...")
    x_train, x_test, y_train, y_test = \
        utils.load_data(FLAGS.data_file, FLAGS.input_type)

    # Get the length of protein seqs
    train_seqs_len = [np.count_nonzero(x) for x in x_train]
    test_seqs_len = [np.count_nonzero(x) for x in x_test]

    n_inputs = max(train_seqs_len + test_seqs_len)
    # Data size
    num_samples, num_features = x_train.shape
    classes = np.sort(np.unique(y_train))
    print("\n=================================\nData details:")
    print("- Training-set:\t\t{}".format(len(y_train)))
    print("- Test-set:\t\t{}".format(len(y_test)))
    print("- Features:\t\t{}".format(num_features))
    print("- Examples:\t\t{}".format(num_samples))
    print("- Classes:\t\t{}".format(classes))
    print("=================================\n\n")

    train_cnn(x_train, x_test, y_train, y_test)

# Training
# ==================================================
def train_cnn(x_train, x_test, y_train, y_test):
    """Return the data, x & y, from the input file"""
    labels_train = utils.labels_one_hot(y_train)
    labels_test = utils.labels_one_hot(y_test)
    tf.reset_default_graph()

    with tf.Session() as sess:
        num_examples, _ = x_train.shape

        with tf.name_scope('Model'):
            cnn = CnnModel(FLAGS)

        # Output directory for models and summaries
        out_dir = os.path.join(FLAGS.logdir, time.strftime("%Y-%m-%d-%H-%M-%S"))
        res_dir = os.path.join(FLAGS.resultsdir, time.strftime("%Y-%m-%d-%H-%M-%S"))
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        # # Train Summaries
        summaries = tf.summary.merge_all()
        summary_dir = os.path.join(out_dir, "summaries", "all")
        summary_writer = tf.summary.FileWriter(summary_dir, graph=tf.get_default_graph())

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.cost)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "test")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Optimize
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cnn.cost)

        # Initializing the variables
        init = tf.global_variables_initializer()
        sess.run(init)

        checkpoint_dir = os.path.join(out_dir, 'model')
        saver = tf.train.Saver(tf.global_variables())

        # Early stopping: if no improvement in test accuracy seen within n epochs
        best_accuracy = 0.0
        last_improvement = 0
        check_improvement = 10

        with open(FLAGS.output, 'w') as outfile:
            outfile.write("==========   {}   Parameters:\n\n".format(model_type))
            for k, v in FLAGS.__flags.items():
                outfile.write('{}:\t{}\n'.format(k, v))
            outfile.write("===============================\n\n")

            # Training cycle
            for epoch in range(FLAGS.num_epochs):
                avg_cost = 0.
                total_batch = int(num_examples/FLAGS.batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_xs, batch_ys = \
                        utils.next_batch(x_train, labels_train, FLAGS.batch_size)
                    # Run optimization op (backprop), cost op (to get loss value)
                    # and summary nodes
                    feed_dict = {
                        cnn.x_seq: batch_xs,
                        cnn.data_y: batch_ys,
                        cnn.keep_prob: FLAGS.keep_prob,
                        cnn.training: 1
                    }
                    asumm, summ, _, cost, acc =\
                        sess.run([summaries, train_summary_op, optimizer, cnn.cost, cnn.accuracy],
                                 feed_dict=feed_dict)
                    # Write logs at every iteration
                    train_summary_writer.add_summary(summ, epoch * total_batch + i)
                    summary_writer.add_summary(asumm, epoch * total_batch + i)
                    # Compute average loss
                    avg_cost += cost / total_batch
                # Display logs per epoch step
                if (epoch+1) % FLAGS.display_step == 0:
                    feed_dict = {
                        cnn.x_seq: x_test,
                        cnn.data_y: labels_test,
                        cnn.keep_prob: 1,
                        cnn.training: 0
                    }
                    t_summ, t_cost, t_acc, t_preds, t_scores = \
                        sess.run([dev_summary_op, cnn.cost, cnn.accuracy, cnn.pred_cls, cnn.logits],
                                 feed_dict=feed_dict)
                    dev_summary_writer.add_summary(t_summ, epoch)

                    if t_acc > best_accuracy:
                        best_accuracy = t_acc
                        last_improvement = epoch
                        saver.save(sess=sess, save_path=checkpoint_dir)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    msg1 = "- Train cost:\t\t{:.5f}\n- Train Acc:\t\t{:.5f}\n"\
                           "- Test cost:\t\t{:.5f}\n- Test Acc:\t\t{:.5f} {}".\
                           format(avg_cost, acc, t_cost, t_acc, improved_str)
                    outfile.write("\n==============  Epoch: {:04d}  ==============\n".format(epoch+1))
                    outfile.write(msg1)
                    print("\n==============  Epoch: {:04d}  ==============\n".format(epoch+1))
                    print(msg1)

                    cnf_matrix = confusion_matrix(y_true=y_test, y_pred=t_preds)

                    class_names = ['Non-binding', 'Binding']
                    # Plot normalized confusion matrix
                    plot_name, _ = os.path.splitext(os.path.basename(FLAGS.data_file))
                    conf_plot_name = res_dir + "/" + plot_name + "_cf"
                    precrec_plot_name =  res_dir + "/" + plot_name + "_pr"
                    # Plot normalized confusion matrix
                    utils.plot_confusion_matrix(cnf_matrix,
                                                           class_names,
                                                           conf_plot_name,
                                                           epoch,
                                                           normalize=False)
                    #OUT.write(cnf_matrix)
                    prec = precision_score(y_true=y_test, y_pred=t_preds)
                    recall = recall_score(y_true=y_test, y_pred=t_preds)
                    fone = f1_score(y_true=y_test, y_pred=t_preds)
                    msg2 = "- Precision:\t\t{:.5f}\n- Recall:\t\t\t{:.5f}\n"\
                           "- f1_score:\t\t{:.5f}\n".format(prec, recall, fone)
                    outfile.write(msg2)
                    print(msg2)

                    # Compute Precision-Recall and plot curve precrec_plot_name
                    utils.plot_precrecal_curve(FLAGS.num_classes,
                                                          labels_test,
                                                          t_scores,
                                                          precrec_plot_name,
                                                          epoch)

                    # If no improvement found in the required number of iterations.
                    if epoch - last_improvement > check_improvement:
                        print("#### No improvement found in a while, stopping optimization.\n")
                        break

            print('\t\tBest Accuracy: ', best_accuracy, "\n")

        # Test model
        # Calculate accuracy
        #print("Accuracy:", accuracy.eval({cnn.x_seq: x_test, cnn.data_y: labels_test}))

        print("=========================\nOptimization Completed!\n"\
              "Run the following command line in terminal:\n" \
              "    tensorboard --logdir=/tmp/model_log " \
              "\nThen open http://127.0.1.1:6006/ in your web browser\n"\
              "=========================\n\n")
if __name__ == "__main__":
    # execute only if run as a script
    main()
