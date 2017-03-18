#**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./test-images/29_one.png "Traffic Sign 1"
[image2]: ./test-images/2_two.png "Traffic Sign 2"
[image3]: ./test-images/11_three.png "Traffic Sign 3"
[image4]: ./test-images/5_four.png "Traffic Sign 4"
[image5]: ./test-images/37_five.png "Traffic Sign 5"
[image6]: ./examples/histo.png "Training data distribution"
[image7]: ./examples/panel.png "Table of traffic signs"
[image8]: ./examples/sample.png "Sample sign"
[image9]: ./examples/sample_gray.png "Sample sign in gray"
[image10]: ./examples/transformed.png "Comparison"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kindoblue/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed

![alt text][image6]

And these are 5 random samples for every class in the data set:

![alt text][image7]   

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook. See in particular the pipeline function.

As a first step, I decided to convert the images to grayscale so we can do convolutions quicker and there will be lesse weights in the model. I also decided to equalize the images as they come in different light condition. The last step I did was normalization. You usually normalize features to improve the learning phase, avoid oscillations in the gradient descent. In this case the relative pixel values are already more or less the same, so I am not sure if this is really mandatory.   

Here is an example of a traffic sign image before and after the processing.

Original      | Gray scaled
:------------:|:-------------------------:
![][image8]   |  ![][image9] 


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

There was not need to split data in training, test and validation as three different pickle files were already provided.   

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because for complex neural networks as LeNet you need a lot of data to train and avoid underfitting. If you take a look at the histogram above, you can see that for certain classes the samples were simply not enough. 

I decided to augment the data by sampling, for each class, a set of images that would be transformed by changing perspective (by using homography) and rotated. The transformation to apply for each image in the sampled set is chosen randomly.  

Here is an example of an original image and an augmented image:

![][image10]


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 8th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description 
|:-----------------:|:-------------------------------------------:| 
| Input         		| 32x32x1 gray scale, normalized image        | 
| Convolution 5x5  	| 1x1 stride, valid padding, outputs 28x28x12 |
| RELU					|						                          |
| Max pooling	      	| 2x2 stride,  outputs 14x14x12               |
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x32 |
| RELU					|						                          |
| Max pooling	      	| 2x2 stride,  outputs 14x14x12               |
| Fully connected   | 800x200                                     |
| RELU					|						                          |
| Fully connected   | 200x84                                      |
| RELU					|						                          |
| Fully connected   | 84x43                                       |
| Softmax				| |
 
 

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 9th cell of the ipython notebook. 

To train the model, I used an AdamOptimizer on the calculated cross entropy. I increased the batch size since the memory constraints were not a big problem. With a bigger batch size you have more stability in gradient. I didnt touch the learning rate as it seems to be good enough.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 10th cell of the Ipython notebook.

My final model results were:
* training set accuracy of ~0.995
* validation set accuracy of ~0.98 
* test set accuracy of ~0.96

I started from the tensorflow implementation of LeNet of the lab. Without any modification I tried on the augmented data set and I got 97% accuracy on the training set and 92% on the validation set. So LeNet seemed a good starting point, as suggested. The first thing I tried, then, was to add dropout since the accuracy on the validation set was suggesting perhaps poor generalization. Things improved but after some trials it appeared that the net as it was could not go farther. So I added more depth to the convolutional layers and conseguently I enlarged the fully connected layers. Another thing I modified was the initialization of the weights. The standard deviation was calculated by sqrt(2/n) where is the fan in, as suggested [here](http://cs231n.github.io/neural-networks-2/)


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 