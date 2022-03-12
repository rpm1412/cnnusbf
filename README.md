# cnnusbf
An implementation of the network in the paper ![**"Towards Fast Region Adaptive Ultrasound Beamformer For Plane Wave Imaging Using Convolutional Neural Networks"**](https://ieeexplore.ieee.org/document/9630930) published in the Proceedings of 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), Online, October 2021. https://doi.org/10.1109/EMBC46164.2021.9630930  

The implementation can be found in the code snippet titled *cnnusbf.py*

Standard tensorflow backend with Keras API is used to train the network. Hence it is essential to have tensorflow and Keras installed. The code can be easily implemented and hosted on Google Colab that provides hardware support for training the network. With a hardware accelerator (GPU) the speed of execution is manifold greater.

***Pictorial representation of training method and details for implementation***

Existing method from baseline study:

![click to view image](https://raw.githubusercontent.com/rpm1412/cnnusbf/main/img/img1.png)

Proposed method:

![click to view image](https://raw.githubusercontent.com/rpm1412/cnnusbf/main/img/img2.png)

With the proposed method, we aim to leverage on the spatial information employed in the convolution kernels in the Convolutional Architecture to provide superior results in adaptive beamforming when compared to a pixel based strategy in the baseline. The advantages seen are that of : 1) Superior metrics with a lesser requirement in angles of insonification for similar image metrics, 2) Lesser angles of insonification could mean a possibility of increased frame rates.

A few plots are attached below to present the advantages in a nutshell to the readers. Both the baseline and the proposed work is trained using the same training data to maintain a pivot for comparison and standard PICMUS dataset is used for evaluation.

![click to view image](https://raw.githubusercontent.com/rpm1412/cnnusbf/main/img/img3.png)
