# cnnusbf
An implementation of the network in the paper "Towards Fast Region Adaptive Ultrasound Beamformer For Plane Wave Imaging Using Convolutional Neural Networks"

The implementation can be found in the code snippet titled cnnusbf.py
Standard tensorflow backend with Keras API is used to train the network. Hence it is essential to have tensorflow and Keras installed. The code can be easily implemented and hosted on Google Colab that provides hardware support for training the network. With a hardware accelerator (GPU) the speed of execution is manifold greater.

The complete implementation and details of the study will be available after the review by the IEEE committee. This work has been submitted to the IEEE for possible publication. The links will be provided as and when available.

With the proposed method, we aim to leverage on the spatial information employed in the convolution kernels in the Convolutional Architecture to provide superior results in adaptive beamforming when compared to a pixel based strategy in the baseline. The advantages seen are that of : 1) Superior metrics with a lesser requirement in angles of insonification for similar image metrics, 2) Lesser angles of insonification could mean a possibility of increased frame rates.

A few plots are attached below to present the advantages in a nutshell to the readers. Both the baseline and the proposed work is trained using the same training data to maintain a pivot for comparison and standard PICMUS dataset is used for evaluation.
