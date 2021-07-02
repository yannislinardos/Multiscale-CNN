#  Multiscale Convolutions for an Artificial Neural Network 

The study investigates the possibility of using convolutional neural networks across input of different sampling rates, focusing on one-dimensional convolutions. This is an idea that has not been adequately studied although it may produce useful results that expand the usefulness of convolutional neural networks. The problem was approached from the perspective of algebraic multigrid. Three interpolation methods were tested on audio classification neural networks trained for input of different sampling rates: nearest neighbor, linear interpolation and inverse distance weighting. The approach was extended to pooling and fully connected layers. In the case of using a neural network trained for high sampling rate input with input of low resolution, the method of linear interpolation gave promising results. Moreover, the results hint that pooling layers should not be changed in the process of multiscaling. In the case of training for low sampling rate and testing with input of high sampling rate, there is no unique solution to the system of weight equations. In dealing with this problem, the approach of directly prolonging the convolution kernels was tried using the three interpolation methods that were explained above as well as the method of kernel dilation. The last method, kernel dilation, appeared to be considerably effective in upscaling.

Read more(https://essay.utwente.nl/78850/)
