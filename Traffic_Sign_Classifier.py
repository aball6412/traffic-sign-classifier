
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission, if necessary. Sections that begin with **'Implementation'** in the header indicate where you should begin your implementation for your project. Note that some sections of implementation are optional, and will be marked with **'Optional'** in the header.
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[90]:

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "../Data/traffic-signs-data/train.p"
testing_file = "../Data/traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below.

# In[91]:

### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of a traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = 43

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.

# In[92]:

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import random
# Visualizations will be shown in the notebook.
get_ipython().magic('matplotlib inline')

# Get a random number
random_number = random.randint(0, n_train)

# Grab the random image from the dataset
plt.figure(figsize=(2,2))
sample_image = X_train[random_number]
sample_label = y_train[random_number]

# Show the image and corresponding label
print(sample_label)
plt.imshow(sample_image)





# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.
# 
# **NOTE:** The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play!

# ### Implementation
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.

# # Preprocess Data

# In[93]:

### Preprocess the data here.
### Feel free to use as many code cells as needed.

from sklearn.utils import shuffle

#Shuffle the data. 
X_train, y_train = shuffle(X_train, y_train)



# ### Question 1 
# 
# _Describe how you preprocessed the data. Why did you choose that technique?_

# **Answer:** 
# 1. Shuffled the data because the data was currently in groups. Aka all 20kph speed limit signs were one after another.
# 

# # Split data up to get cross validation testing sets

# In[94]:

### Generate additional data (OPTIONAL!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.

from sklearn.model_selection import train_test_split

# Split off some test data from the current data
X_train_inputs, X_test_inputs, y_train_labels, y_test_labels = train_test_split(X_train, y_train, test_size=0.20, train_size=0.80)

#See what the new data lenghts are
print("Original...")
print("X_train: {}".format(len(X_train)))
print("y_train: {}".format(len(y_train)))
print("")
print("For testing...")
print("X_train_inputs: {}".format(len(X_train_inputs)))
print("y_train_labels: {}".format(len(y_train_labels)))
print("X_test_inputs: {}".format(len(X_test_inputs)))
print("y_test_labels: {}".format(len(y_test_labels)))





# ### Question 2
# 
# _Describe how you set up the training, validation and testing data for your model. **Optional**: If you generated additional data, how did you generate the data? Why did you generate the data? What are the differences in the new dataset (with generated data) from the original dataset?_

# **Answer:**
# 1. I took out 20% of the original training data and used that 20% for cross validation testing

# # Define neural network architecture

# In[151]:

### Define your architecture here.
### Feel free to use as many code cells as needed.

from tensorflow.contrib.layers import flatten
import tensorflow as tf

# Gather all information for training
epochs = 10
batch_size = 128
classes = n_classes

n_samples = len(X_train_inputs)
n_labels = len(y_train_labels)


# Create Network Architecture function
def Lenet(x, weights, biases):
    
    #---- Also play around with filter size, stride, depth ----
    

    
    #First hidden layer
    layer_1 = tf.nn.conv2d(x, weights['h1'], strides=[1, 1, 1, 1], padding='VALID') + biases['b1']
    
    #RELU Activation
    layer_1 = tf.nn.relu(layer_1)
    
    #Max Pooling
    layer_1 = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    


    
    
    #Second hidden layer
    layer_2 = tf.nn.conv2d(layer_1, weights['h2'], strides=[1, 1, 1, 1], padding='VALID') + biases['b2']
    
    #RELU Activation
    layer_2 = tf.nn.relu(layer_2)
    
    #Layer 2 Max Pooling
    layer_2 = tf.nn.max_pool(layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    

    
    #Flattten layer
    fc_layer0 = flatten(layer_2)
    
    #Create fully connected layer 1
    fc_layer1 = tf.matmul(fc_layer0, weights['fc1_w']) + biases['fc1_b']
    
    #RELU Activation
    fc_layer1 = tf.nn.relu(fc_layer1)
    
    
    
    
    #Create fully connected layer 2
    fc_layer2 = tf.matmul(fc_layer1, weights['fc2_w']) + biases['fc2_b']
    
    #RELU Activation
    fc_layer2 = tf.nn.relu(fc_layer2)
    
    
    
    
    #Create fully connected layer 3
    fc_layer3 = tf.matmul(fc_layer2, weights['fc3_w']) + biases['fc3_b']
    
    #Create logits variable just because
    logits = fc_layer3
    
    return logits
    



#Create hyper perameters
mu = 0
sigma = 0.1

#Set up weights and biases for the multiple layers
#Weight shape will also be the filter size
#in example shape=(5,5,1,6), the 5,5 is the filter size. 1 is 3rd dimension of input (for image it's 3 rgb) 6 is how many neurons we are connecting to.

weights = {
    'h1': tf.Variable(tf.truncated_normal(shape=(5,5,3,6), mean = mu, stddev = sigma)),
    'h2': tf.Variable(tf.truncated_normal(shape=(5,5,6,16), mean = mu, stddev = sigma)),
    'fc1_w': tf.Variable(tf.truncated_normal(shape=(400,120), mean = mu, stddev = sigma)),
    'fc2_w': tf.Variable(tf.truncated_normal(shape=(120,84), mean = mu, stddev = sigma)),
    'fc3_w': tf.Variable(tf.truncated_normal(shape=(84,43), mean = mu, stddev = sigma))
}
                      
                     
biases = {
    'b1': tf.Variable(tf.zeros(6)),
    'b2': tf.Variable(tf.zeros(16)),
    'fc1_b': tf.Variable(tf.zeros(120)),
    'fc2_b': tf.Variable(tf.zeros(84)),
    'fc3_b': tf.Variable(tf.zeros(43))
}

#Create placeholders for features and labels
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)


#Create the training pipeline
learning_rate = 0.001

logits = Lenet(x, weights, biases)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss_operation)



#Create the model evaluation function
correct_prediction = tf.equal(tf.arg_max(logits, 1), tf.arg_max(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    total_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    
    for i in range(0, total_examples, batch_size):
        batch_x, batch_y = X_data[i:i+batch_size], y_data[i:i+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        
    return total_accuracy/total_examples



# In[ ]:




# ### Question 3
# 
# _What does your final architecture look like? (Type of model, layers, sizes, connectivity, etc.)  For reference on how to build a deep neural network using TensorFlow, see [Deep Neural Network in TensorFlow
# ](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/b516a270-8600-4f93-a0a3-20dfeabe5da6/concepts/83a3a2a2-a9bd-4b7b-95b0-eb924ab14432) from the classroom._
# 

# **Answer:**

# # Train the Model

# In[152]:

### Train your model here.
### Feel free to use as many code cells as needed.

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    total_examples = len(X_train_inputs)
    print("Training...")
    print("")
    
    for i in range(epochs):
        X_train_inputs, y_train_labels = shuffle(X_train_inputs, y_train_labels)
        
        for offset in range(0, total_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train_inputs[i:end], y_train_labels[i:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        
        validation_accuracy = evaluate(X_test_inputs, y_test_labels)
        print("Epoch: {}...".format(i+1))
        print("Validation Accuracy: {:.3f}".format(validation_accuracy))
        print("")
        
    saver.save(sess, 'first_train_attempt')
    print("Model saved.")





# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# ### Question 4
# 
# _How did you train your model? (Type of optimizer, batch size, epochs, hyperparameters, etc.)_
# 

# **Answer:**

# ### Question 5
# 
# 
# _What approach did you take in coming up with a solution to this problem? It may have been a process of trial and error, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think this is suitable for the current problem._

# **Answer:**

# ---
# 
# ## Step 3: Test a Model on New Images
# 
# Take several pictures of traffic signs that you find on the web or around you (at least five), and run them through your classifier on your computer to produce example results. The classifier might not recognize some local signs but it could prove interesting nonetheless.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Implementation
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.

# In[ ]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.


# ### Question 6
# 
# _Choose five candidate images of traffic signs and provide them in the report. Are there any particular qualities of the image(s) that might make classification difficult? It could be helpful to plot the images in the notebook._
# 
# 

# **Answer:**

# In[ ]:

### Run the predictions here.
### Feel free to use as many code cells as needed.


# ### Question 7
# 
# _Is your model able to perform equally well on captured pictures when compared to testing on the dataset? The simplest way to do this check the accuracy of the predictions. For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate._
# 
# _**NOTE:** You could check the accuracy manually by using `signnames.csv` (same directory). This file has a mapping from the class id (0-42) to the corresponding sign name. So, you could take the class id the model outputs, lookup the name in `signnames.csv` and see if it matches the sign from the image._
# 

# **Answer:**

# In[ ]:

### Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.


# ### Question 8
# 
# *Use the model's softmax probabilities to visualize the **certainty** of its predictions, [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. Which predictions is the model certain of? Uncertain? If the model was incorrect in its initial prediction, does the correct prediction appear in the top k? (k should be 5 at most)*
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# **Answer:**

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# In[ ]:



