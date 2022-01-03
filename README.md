<img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>
<img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/></a>

#
# Fine-grained depressed emotion recognition using deep metric learning loss functions
Recently Transformer based pre-trained language models like BERT have rapidly advanced the-state-of-the-art on many NLP tasks. While great performance has been improved by using pre-trained language model, it faces difficulties due to the limitation of cross-entropy. Since the cross-entropy loss function does not have the property of maximizing the margin between classes, it can lead to local optimization and  low generalization performance.

In order to overcome these limitations, a deep metric learning-based loss function has been proposed. It can improve good generalization performance in that it finds similarities between samples of the same class and contrasts them with other classes. 


In the field of text-based emotion classification previous studies showed better performance by adding a deep metric learning-based loss functions. However, there are few studies on how to improve the performance of the model through the design of a loss function that combines two or more various deep metric learning-based functions.
Therefore in this paper, we propose combined loss function of various deep metric learning-based loss functions. which leads to improvement of performance in fine-grained depression emotions classification task. 
<img width="768" alt="model_architecture" src="https://user-images.githubusercontent.com/70733246/147923445-5a304481-c864-4660-bdc5-b7a959bf36ae.png">

![image](https://user-images.githubusercontent.com/70733246/147922871-220fd29b-8b12-44f8-86a4-b8404dbae119.png)
