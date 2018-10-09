#-*- coding: utf-8 -*-

import gzip
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import datetime
import tensorflow as tf



# Parameters
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  
epochs = 1000

# Augment training data
def expend_training_data(images, labels):

    expanded_images = []
    expanded_labels = []

    j = 0 # counter
    for x, y in zip(images, labels):
        j = j+1
        if j%100==0:
            print ('expanding data : %03d / %03d' % (j,np.size(images,0)))

        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)

        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value 
        bg_value = np.median(x) # this is regarded as background's value        
        image = np.reshape(x, (-1, 28))

        for i in range(4):
            # rotate the image with random degree
            angle = np.random.randint(-15,15,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = np.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)

            # register new training data
            expanded_images.append(np.reshape(new_img_, 784))
            expanded_labels.append(y)

    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken
    expanded_train_total_data = np.concatenate((expanded_images, expanded_labels), axis=1)
    np.random.shuffle(expanded_train_total_data)

    return expanded_train_total_data


# Extract the images
def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = np.reshape(data, [num_images, -1])
    return data



# Extract the labels
def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data,NUM_LABELS))
        one_hot_encoding[np.arange(num_labels_data),labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding


# Prepare MNISt data
def prepare_MNIST_data(use_data_augmentation=True):
    # Get the data.
    train_data_filename = './data/train-images-idx3-ubyte.gz'
    train_labels_filename = './data/train-labels-idx1-ubyte.gz'
    test_data_filename = './data/t10k-images-idx3-ubyte.gz'
    test_labels_filename = './data/t10k-labels-idx1-ubyte.gz'

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, :]
    validation_labels = train_labels[:VALIDATION_SIZE,:]
    train_data = train_data[VALIDATION_SIZE:, :]
    train_labels = train_labels[VALIDATION_SIZE:,:]

    # Concatenate train_data & train_labels for random shuffle
    if use_data_augmentation:
        train_total_data = expend_training_data(train_data, train_labels)
    else:
        train_total_data = np.concatenate((train_data, train_labels), axis=1)

    train_size = train_total_data.shape[0]

    return train_total_data, train_size, validation_data, validation_labels, test_data, test_labels

print("Preparing of mnist data is Success !!!!")


train_total_data, train_size, validation_data, validation_labels, test_data, test_labels = prepare_MNIST_data(True)


np.random.shuffle(train_total_data)
train_data = train_total_data[:, : -NUM_LABELS]
train_labels = train_total_data[:, -NUM_LABELS: ]

print("Sample image of training data :")
image = np.reshape(train_data[10], [-1,28,28,1])
img = image.squeeze()
plt.imshow(img, cmap='gray_r')
plt.show()

tf.reset_default_graph() 
sess = tf.InteractiveSession()
x = tf.placeholder("float", shape = [None, 28,28,1]) #shape (None x height x width x color channels)
y_ = tf.placeholder("float", shape = [None, 10]) #shape (None x number of classes)


#############   CNN MODEL  ##############

# model inputs

tf.reset_default_graph() 
sess = tf.InteractiveSession()
x = tf.placeholder("float", shape=[None, 28, 28, 1]) #shape (None x height x width x color channels)
y_ = tf.placeholder("float", shape=[None, 10]) #shape (None x number of classes)


# Architecture

w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)) #(filter x filter x input channels x output channels)
b1 = tf.Variable(tf.constant(.1, shape = [32])) #bias shape matches to output channels

# layer 1
h1 = tf.nn.conv2d(input=x, filter=w1, strides=[1, 1, 1, 1], padding='SAME') + b1
h1 = tf.nn.relu(h1)
h1 = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv2d(x, W):
  return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# layer 2
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(.1, shape = [64]))

h_conv2 = tf.nn.relu(conv2d(h1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# Fully Connected Layer 1
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(.1, shape = [1024]))

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout Layer
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully Connected Layer 2
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(.1, shape = [10]))

#Final Layer
y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2



crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))


trainStep = tf.train.AdamOptimizer().minimize(crossEntropyLoss)


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


###################################################




sess.run(tf.global_variables_initializer())


tf.summary.scalar('Cross Enropy Loss', crossEntropyLoss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)



def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

###### TRAINING ######

batch_size = 100





for epoch in range(epochs):
    trainInputs, trainLabels = next_batch(batch_size, train_data, train_labels)
    trainInputs = trainInputs.reshape([batch_size, 28, 28, 1])
    if epoch % 10 == 0:
        summary = sess.run(merged, {x: trainInputs, y_: trainLabels, keep_prob: 1.0})
        writer.add_summary(summary, epoch)
    if epoch % 100 == 0:
        validInputs, validLabels = next_batch(batch_size, validation_data, validation_labels)
        validInputs = validInputs.reshape([batch_size, 28, 28, 1])
        validAccuracy = accuracy.eval(session=sess, feed_dict={x:validInputs, y_: validLabels, keep_prob: 1.0})
        trainAccuracy = accuracy.eval(session=sess, feed_dict={x:trainInputs, y_: trainLabels, keep_prob: 1.0})
        print("Epoch {:>3} - Train Accuracy: {:>6.4f} - Validation Accuracy: {:>6.4f}".format(epoch, trainAccuracy, validAccuracy))
    
    trainStep.run(session=sess, feed_dict={x: trainInputs, y_: trainLabels, keep_prob: 0.7})




saver = tf.train.Saver()

saver.save(sess, "model/model.ckpt")


#### Testing ######

for i in range(100, 111):
    test_data = validation_data[i].reshape([-1, 28, 28, 1])
    test_label = validation_labels[i].reshape(1,10)
    

    test = tf.argmax(y,1)
    predict = sess.run([test], feed_dict = {x: test_data, y_: test_label, keep_prob: 1.0})

    print("Sample image of Test data :")
    img = test_data.squeeze()
    plt.imshow(img, cmap='gray_r')
    plt.show()

    print(" Predicted Label : " + str(predict[0]))
    