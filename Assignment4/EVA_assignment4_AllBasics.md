**Details:**

Prajna,Prajnaraipn@gmail.com

Notes:Gathered from Net and EVA class.

<b>1. Image Normalization</b>

The pixel values in images must be scaled before sending them as input to the neural network model , which will be taken into consideration while training or evaluating the model.In our case, in mnist dataset.We have normalized the image to 0 to 255.Basically, this is a process to change the pixel intensity values in the input image. For example if we are using a grey scale image, then the pixel intensity values range from 0 to 255.It is done to ensure the handling of the outliers.

<b>2. 3,3 Convolution </b>

Ideally 3,3 is the best kernel considered in CNN. That is a 3,3 matrix is strided over an image to get the next layer with feature maps. If a 5,5 matrix is strided over a 3,3 convolution kernel, we get a 2,2 image. If an image of 28,28 with channel is convolved with 3,3 convolutional kernel of 32 channels, we get the next layer of matrix of pixels with 26,26 and 32 channels. It goes this way:

26,26,32 -------convolved(64,3,3)-------24,24,64--------max pool(2,2)------12,12,64


![alt text](https://github.com/prajnaraipn7/Eva_Assignment/blob/master/Assignment4/pictures/1_EuSjHyyDRPAQUdKCKLTgIQ.png)

​                                                         Fig1: Explains the working of 3,3 Convolution

In the above fig 1 , 3,3 convolution filter is used.

<b>3.How many layers</b>

An image is treated as Input. The image is represent in the form of matrix. Each matrix has pixel values. The Image is passed through the kernel to form the next level of layers. This is done to an extent, we receive the best features that very loudly states about the image. The layers learn in this fashion, Edges – Pattern – part of the object –finally, the object. The intention of convolution is to find the features which will help us to classify any image in future.

This is based on the image size .We assume that the size of the image is equal to the size of the object. Suppose if am taking a image of size 400x400 and if I want to reach 1x1 using 3x3 kernels (with no consideration of other concepts like 1x1, maxpooling etc.) then we need to add 400/2 = 200 convolution layers need to be added.

![](C:\Users\prajna.neerchal\Desktop\1_mcBbGiV8ne9NhF3SlpjAsA.png)

​               							Fig2, Explains the convolution

![](C:\Users\prajna.neerchal\Desktop\1_mlPKapLTRuVVhLMLfq46pQ.png)

​                                                                    Fig 3shows up to 3 layers

<b> Kernels and how do we decide the number of kernels</b>

These are those containers, which helps to get the best feature after striding through the layer (the matrix of pixels) from one layer to another .It is a sliding window, which slides through the layer to get the best feature map that will be compiled for the next layer. A kernel of suitable size in each stage  is chosen to convolve over the input image. The key point here is to convolve and get the key features in the image. If we have a 5,5 matrix (layer), we convolve with 3,3 kernel to get 2,2 image.

These are the feature extractors.For example to extract vertical edges, horizontal edges out of an image.
The number of are decided depending on the several factors such as the number of features each kernel has to extract i.e. more number of unique features to be extracted, then more kernels. 

![](C:\Users\prajna.neerchal\Desktop\1_gSlM_2hk3hqxzELgu0B2Ag.png)

​                                                Fig 4:View of Kernel

A kernel is a matrix with the dimensions *[h2 \* w2 \* d1],* which is *one* yellow cuboid of the multiple cuboid (kernels) stacked on top of each other (in the kernels layer) in the above image.Here dl  is the number of channels.

<b>5.Receptive Field</b>

The receptive field in a convolutional neural network refers to the part of the image(region of a image) that is visible to a layer.Or we can also state , what region the layer is looking at .

There is a Global Receptive Field and Local receptive fold.

Global receptive field,helps us to tell us the information of where we started with convolution and where I am in my convolution network. For example if I convolved an image of resolution of 40,40 to 38,38  and then 36,36 using 3x3 kernels, then my global receptive field at 36th kernel is 5.

In our case, local receptive field is always 3.

Its basically, how much a layer is able to see a region from the below layers!!!

![](C:\Users\prajna.neerchal\Desktop\nnn.JPG)

​                                                  Fig 5: View of receptive field at kernel 2

<b>6.Max Pooling</b>

There are two types of pooling:

​		**Max Pooling**

​               **Average Pooling**

The main purpose of a pooling layer is to reduce the number of parameters of the input tensor and thus

- Extract representative features from the input tensor
- Reduces computation and thus aids efficiency
- Helps reduce over fitting

In case of Max Pooling, an example of which is shown in the *Fig, below* a kernel of size `n*n` *(2x2 in the above example)* is moved across the matrix and for each position the **max value is taken** and put in the corresponding position of the output matrix.

![](C:\Users\prajna.neerchal\Desktop\1_wTu-73e3QjibbAplN4DIQA.png)

​                                                          Fig 6: Max pooling Logic

The function of max pooling is to pgradually reduce the spatial size of the representation to reduce the amount of parameters and computation in the network. 

**7.Position of Max Pooling**

Max pooling is desired after convolution layers and it’s positioned after every 2 to 3 convolution layers. Also, it’s always better to avoid using the same before last layer, where final convolved image is obtained.

**8. 1,1 Convolution**

The number of feature maps(channels) often increases with the depth of the network. This problem can result in a great increase in the number of parameters and computation(in turn leading to more resource complexity] required when larger filter sizes are used, such as 5×5 and 7×7.To address this issue 1,1 convolution came to picture.This is also called channel-wise pooling.

This is used generally to reduce the number of channels(feature maps), basically dimensionality reduction. 

This is usually done after some convolution layers, where all the pixel values of each channel is merged with pixels values of the other and an output is obtained for each pixel i. e. it merges the pixel values and provides a scaled output thus keeping the resolution of the image same. But ,whenever one want to reduce the channels in their model they can use this.

![](C:\Users\prajna.neerchal\Pictures\full_padding_no_strides_transposed_small.gif)

​                                                          Fig 7: Depict of  1,1 convolution

**9. Concept of Transition layers**

Transition layer is a combination of 1x1 and Max pooling and is usually applied post some convolution layers. There is no clear evidence as to use 1x1 first or max pooling first but preferably, we use 1x1 first and then Max pooling.

**10.Position of Transition Layer**

This is always preferred post 2-3 layers of convolution and avoided at the last 2-3 layers.

**11.The distance of Max pooling from Prediction**

Max pooling should be at least 2-3 layers before the prediction layer. For example, an image 3,3 is convolved, if max pooling is applied at the end layer, all that remains in the image will be a representation of dot.Apparently, max pooling at the end causes a lesion by loosing all the important features required in ones model to trace and predict the labels in future.One looses lot of essential features .

**12.Batch Normalization**

This is used to decide the output of each batch i.e. finalize the outputs by normalizing the values from -1 to 1. This ensures that Kernels has learnt with some definite output at the end of each batch.

Batch normalization reduces the amount by what the hidden unit values shift around.It is a technique for improving the performance and stability of neural networks .This stabilizes the learning process and reduce the number of training epochs required to train deep networks.

**13.The distance of Batch Normalization from Prediction**

It’s better to avoid batch normalization in the last layer since it normalizes the values from -1 to 1 and we don't want to reduce the features in the last layer.Or in simpler terms, we don't want to loose any information in the last layer by adding this as a scope.

**14.Softmax**

Softmax ensures that network gives a decisive and a better accuracy, when applied at the last layer. This is better when compared to the results without applying the softmax. Having said that, this is not a complete probability but its probability like.

![](C:\Users\prajna.neerchal\Desktop\Capture.JPG)

​                                                                   Fig 8: Formula of softmax

We use the softmax activation function in the output layer of a deep neural network to represent a categorical distribution over class labels, and obtaining the probabilities of each input element belonging to a label.

**15. DropOut**

Here, we learn less to learn better. Here few of the neurons(kernel features) becomes non -active.That is,dropout negates and deactivates few pixels before the next convolution or transition layer is invoked. This ensures other kernels learn more to compensate this loss of information. Here, basically we drop few of the units(both hidden and visible).That is we ignore few units while forward pass and backward propagation.

![](C:\Users\prajna.neerchal\Pictures\withoutdropout.JPG)

​                                                                Fig 9:FC(Dense) layer without dropout

![](C:\Users\prajna.neerchal\Pictures\withdropout.JPG)

​                                                              Fig 10:FC(Dense) layer without dropout

**16.When do we introduce Dropout, or when do we know we have some overfitting**

When the training accuracy of the model is increasing, i.e. the data is mugged up for each iteration and validation accuracy does not perform well, in this case there is overfitting of the model. Droput is used to prevent overfitting and it increases the validation accuracy.Dropout is a way to regularize the Neural Network Models.

We can use dropout after every batch normalization.Dropout not to be used in the last layer, if we use then the validation accuracy increases more compared to training accuracy.Whereas, training accuracy stays same.

**17.When to add validation checks**

Validation check should be applied for each epoch to keep track on our validation accuracies. This helps us to conclude whether our network is performing well or not.Otherwise, after completing all the epochs, when we check ; we might not have reached the desired accuracy.So we can show training and validation accuracy together.

**18. How do we know our network is not going well, comparatively, very early** 

When the validation checks are applied for each epoch, the validation accuracies observed on the first few epochs, helps us to conclude whether the network is doing good or bad.If the network is taking time to convolve and get better results.That is, when there is huge difference between the training and validation accuracies.

**19.Batch Size, and effects of batch size**

Batch size is used to specify the number of observation after which you we would update weight.Batch size is number of images in the dataset chosen for training. For example of batch size of 32, means 32 forward propagation and one backward propagation are performed. This is a important hyper parameter.Higher batch size = More Memory.

Advantages of batch size:

- It requires less memory. Since one trains the network using fewer samples, the overall training requires less memory. That's especially important if you are not able to fit the whole dataset in your machine's memory.

- Typically networks train faster with mini-batches. 

**20.Number of Epochs and when to increase them**

Epoch is the complete training iteration of all the items in the dataset once. When the validation accuracy is getting better, we can add more epochs to if it continues to improve and obtained the designated accuracy.

one **epoch** = one forward pass and one backward pass of *all* the training example.

**21.Learning Rate**

The amount at which the weights are updated during training is referred to as the step size or the “*learning rate*.”When the validation accuracy of a model reaches a saturation point and neither increases nor decreases, then learning rate is applied to the model. 

**22.LR schedule and concept behind it**

It schedules the learning rate based on the number of epochs. Training or validation accuracy does not affect the learning rate and it predefines the learning rate for each epoch

**23.Adam vs SGD**

Adam is an optimization algorithm to update network weights iteratively whereas SGD only computes on a small subset or random selection of data examples.

**24. When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)**

Taking the example of convolving an image having a representation of number 3, after several convolutions, the whole image is achieved in let’s say 7x7 resolutions. Now at this point, if we continue applying 3x3 kernels and convolve, the image 3 will be reduced to the representation of a dot. In other terms when we are able to extract all features of the image at 7x7, then it is better to apply a 7x7 kernel instead of 3x3 kernel, to get the final output
