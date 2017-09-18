import tensorflow as tf
import numpy as np

class CnnModel():
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, FLAGS):
        self.training = tf.placeholder(tf.int32, name="training")
        filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
        num_filters = list(map(int, FLAGS.num_filters.split(",")))
        # Placeholders for input, output and dropout
        self.x_seq = tf.placeholder(tf.float32, [None, FLAGS.example_len], name="data_x")
        self.data_x = tf.reshape(self.x_seq, [-1, FLAGS.rows, FLAGS.cols, FLAGS.num_channels])
        self.data_y = tf.placeholder(tf.float32, [None, FLAGS.num_classes], name="data_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        
        print("\n\n===================================================")
        print("Tensor shapes:\n")
        
        # Create convolutional Layers: 2 layers
        self.layer_conv1 = \
            self.conv_layer(x_input=self.data_x,
                            num_input_channels=FLAGS.num_channels,
                            filter_size=filter_sizes[0],
                            num_filters=num_filters[0],
                            layer_no=1,
                            use_pooling=True)

        self.layer_conv2 = \
            self.conv_layer(x_input=self.layer_conv1,
                            num_input_channels=num_filters[0],
                            filter_size=filter_sizes[1],
                            num_filters=num_filters[1],
                            layer_no=2,
                            use_pooling=True)

        self.layer_conv3 = \
            self.conv_layer(x_input=self.layer_conv2,
                            num_input_channels=num_filters[1],
                            filter_size=filter_sizes[2],
                            num_filters=num_filters[2],
                            layer_no=3,
                            use_pooling=True)

        print("- Convolution layer 1:\t\t{}".format(self.layer_conv1.get_shape()))
        print("- Convolution layer 2:\t\t{}".format(self.layer_conv2.get_shape()))
        print("- Convolution layer 3:\t\t{}".format(self.layer_conv3.get_shape()))
        print("===================================================\n\n")
                # Flatten last convolution layer: required for 1st fully connected layer
        self.layer_flat, self.len_features = self.flatten_layer(self.layer_conv3)

        # Fully-Connected Layers: 2 layers
        self.layer_fc1 = self.fc_layer(input_x=self.layer_flat,
                                       num_inputs=self.len_features,
                                       num_outputs=FLAGS.fc_size,
                                       use_relu=True)

        self.layer_fc2 = self.fc_layer(input_x=self.layer_fc1,
                                       num_inputs=FLAGS.fc_size,
                                       num_outputs=FLAGS.num_classes,
                                       use_relu=False)

        with tf.name_scope('dropout'):
            self.dropout = tf.layers.dropout(inputs=self.layer_fc2, rate=FLAGS.keep_prob)
        with tf.name_scope('logits'):
            self.logits = tf.layers.dense(inputs=self.dropout, units=FLAGS.num_classes)

        self.pred_cls = tf.argmax(self.logits, dimension=1)
        self.true_cls = tf.argmax(self.data_y, dimension=1)
        #tf.print(self.pred_cls)

        with tf.name_scope('Cost'):
            # Cost-function to be optimized
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.data_y)
            self.cost = tf.reduce_mean(self.cross_entropy)


        with tf.name_scope('Accuracy'):
            self.correct_prediction = tf.equal(self.pred_cls, self.true_cls)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        if self.training is 1:
            tf.summary.histogram('logits', self.logits)
            tf.summary.scalar('train_acc', self.accuracy)
            tf.summary.scalar('train_loss', self.cost)
        elif self.training is 0:
            tf.summary.histogram('logits', self.logits)
            tf.summary.scalar('test_acc', self.accuracy)
            tf.summary.scalar('test_loss', self.cost)

    ###########
    ### Weights & Biases
    def new_weights(self, shape):
        """Creates weights"""
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(self, length):
        """Creates biases"""
        return tf.Variable(tf.constant(0.05, shape=[length]))

    ###########
    ### Convolutional layer
    def conv_layer(self,
                   x_input,            # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   layer_no,
                   use_pooling=True):  # Use 1x2 max-pooling.
        """Return the data, x & y, from the input file"""

        shape = [filter_size, filter_size, num_input_channels, num_filters]
        weights = self.new_weights(shape=shape)

        print("- Convolution layer {} input:\t{}".format(layer_no, x_input.get_shape()))
        print("- Weights for layer {}:\t\t{}".format(layer_no, weights.get_shape()))

        biases = self.new_biases(length=num_filters)

        layer = tf.nn.conv2d(input=x_input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        layer += biases

        # Use pooling to down-sample the image resolution?
        if use_pooling:
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        layer = tf.nn.relu(layer)
        tf.summary.histogram("weights_"+str(layer_no), weights)
        tf.summary.histogram("biases_"+str(layer_no), biases)
        tf.summary.histogram("activations_"+str(layer_no), layer)
        #print(layer.get_shape())
        return layer

    ###########
    ### Flattening function
    def flatten_layer(self, layer):
        """Flattens the layer."""
        # Get the shape of the input layer.
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat, num_features

    ###########
    ### Fully-Connected Layer
    def fc_layer(self,
                 input_x,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?
        """Creates fully connected layer."""
        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(input_x, weights) + biases

        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)

        tf.summary.histogram("fc_weights", weights)
        tf.summary.histogram("fc_biases", biases)
        tf.summary.histogram("fc_activations", layer)
        return layer
