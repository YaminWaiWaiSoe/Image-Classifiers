# Image-Classifiers

There are the proposed different models.

![image](https://user-images.githubusercontent.com/47670208/161647115-f6d7df45-812b-4ff2-8bdd-6b313f9f5039.png)

In ANN models, we get the following results. 

![image](https://user-images.githubusercontent.com/47670208/161647292-dfa89281-657b-4274-b1a2-23e05ba4df28.png)
![image](https://user-images.githubusercontent.com/47670208/161647459-5e9ae417-00bf-4df3-a6f6-82470a984351.png)

In CNN models, there are the results.
![image](https://user-images.githubusercontent.com/47670208/161647518-e3e5a330-1216-46e3-a956-f9a76c9ef898.png)
![image](https://user-images.githubusercontent.com/47670208/161647570-c0ddc711-a3c3-4023-9391-2de881292a1c.png)

Findings

In ANN, Model 2 ( ANN1 + DN1 ) gave the best performance on both datasets.
In CNN, Model 3 (Conv3+MaxPool2+Dense2 ) works best with 5 epochs on both datasets.
Model 3 accuracy decrease with increasing epochs while model 1 (Conv3+MaxPool2+Dense) accuracy increase.
Performance on the fashion dataset is lower than on the digit in all models. One possible reason is that pixels on fashion images are sensitive and more important than digits and  28x28 grayscale fashion images are not sufficient to get high accuracy. Distinguishing features like colour, texture which are only available in RGB images may be important for the algorithm. 
