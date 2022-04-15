# Gender Classification of Image Data
Godwin Anguzu | Haoliang Jiang | Emma Mavis | Nansu Wang  
Machine Learning, Spring 2022

### Abstract
We present an analysis of gender classification on image data that is imbalanced across race, gender, and age. The purpose is to emphasize how the imbalance greatly impacts model performance, despite the reality that image classification applications still operate on imbalanced data. We ran experiments across extremely skewed gender and racially imbalanced groups of image data. We first generated baseline models and narrowed down the choice of a neural network on the data set with no experimentation. Then we applied the best performing neural network model to various imbalanced data and compared the performance on the different subgroups. We found that model performance differs greatly across different subgroups, implying that equitable representation in data is of the utmost importance for model generalizability across subgroups.

Below is a flowhart depicting the project process of this analysis
![image](https://user-images.githubusercontent.com/69800932/163570845-ed18f371-c114-40b7-83c8-f4256ee82d05.png)

### Conclusions
This experiment has clearly shown that even the best-performing models are still quite sensitive to the data they are trained on, especially if that data is biased. We have seen drastic performance differences when applying a model trained on a subset of racial groups, and most notably seen that the performance on the Black subgroup was low each time. It appears that the classification of gender is not as robust as one might think – to perform well on all groups of people, it needs to train equally on all groups of people.
We can confidently state that poor performance across subgroups by neural network classifiers can be attributed to imbalance in those groups, specifically race in gender in this case. Specifically balancing racial groups before facial estimation models is vital to achieve better classification results. But of course reality hardly lends itself to provide balanced data, so methods for mitigating bias need to be at the forefront of any machine learning pipeline.
