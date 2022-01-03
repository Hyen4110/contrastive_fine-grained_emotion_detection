<img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>
<img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/></a>

#
# Fine-grained depressed emotion recognition using deep metric learning loss functions 
## Abstract
Recently **Transformer based pre-trained language models** like BERT have rapidly advanced the-state-of-the-art on many NLP tasks. <br>
While great performance has been improved by using pre-trained language model, <br>
it faces difficulties due to the limitation of **cross-entropy** (<a href="https://arxiv.org/abs/1612.02295">Liu et al 2016;</a> <a href="https://papers.nips.cc/paper/2019/hash/621461af90cadfdaf0e8d4cc25129f91-Abstract.html">Cao et al, 2019)</a>. <br>
> cross-entropy loss function does not have the property of maximizing the margin between classes, <br>
  -->  local optimization and  low generalization performance.<br>


In order to overcome these limitations, a **deep metric learning-based loss function** has been proposed. <br>
It can improve good generalization performance<br>
in that it finds similarities between samples of the same class and contrasts them with other classes. <br>

> In the field of text-based emotion classification <br>
> previous studies showed better performance by adding a **deep metric learning** based loss functions. <br>
However, there are few studies on how to improve the performance of the model through the design of a loss function <br>
that combines two or more various deep metric learning-based functions.<br>


Therefore in this paper, we propose ***combined loss function of various deep metric learning-based loss functions***. <br>
which leads to improvement of performance in fine-grained depression emotions classification task. <br>

<img width="768" alt="model_architecture" src="https://user-images.githubusercontent.com/70733246/147923445-5a304481-c864-4660-bdc5-b7a959bf36ae.png">


## Objective functions
- Triplet Loss <br>
    <img width="400" alt="model_architecture" src="https://user-images.githubusercontent.com/70733246/147949006-a65a2124-63a0-4278-8595-6a2aad13b68a.png"><br>
    <img width="600" alt="model_architecture" src= "https://user-images.githubusercontent.com/70733246/147950294-38e30708-e4ba-4837-b3de-f065e87de323.png"><br>

- Arcface Loss<br>
  <img width="400" alt="model_architecture" src="https://user-images.githubusercontent.com/70733246/147950151-2ee16c2c-c371-40d4-b8f4-f54ae8dd4c1e.png"><br>
  <img width="600" alt="model_architecture" src="https://user-images.githubusercontent.com/70733246/147950208-41138539-46ee-414f-9ae3-1a5c29ffd24d.png"><br>


## Visualization (t-SNE)
![image](https://user-images.githubusercontent.com/70733246/147922871-220fd29b-8b12-44f8-86a4-b8404dbae119.png)
