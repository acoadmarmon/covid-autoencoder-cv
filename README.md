# COVID-19 CT Scan Autoencoder Computer Vision

## Introduction/Background
 
Coronavirus (COVID-19) is an illness caused by SARS-CoV-2 that can spread from person to person primarily through aerosolized respiratory droplets. It has infected over 30 million people and caused around a million deaths worldwide. COVID-19 symptoms can range from mild (or no symptoms) to severe illness. The virus can infect the upper or lower part of the respiratory tract. It travels down the airways, inflaming the lining, and can reach all the way down into the alveoli in lungs. Doctors have reported signs of respiratory inflammation on a chest CT scan of COVID-19 patients. Machine learning approaches on CT scan images can help differentiate between healthy and COVID-19 patients, and also predict prognosis of COVID-19 infected patients. With the current load on the healthcare system worldwide, an automated prognosis prediction model could help physicians more quickly identify, triage, and aggressively treat COVID-19 patients.
 
## Problem definition
 
Develop a machine learning approach to aid the differentiation between healthy and COVID-19 patients, and in the prognosis of COVID-19 infected patients using CT scan images.

## Methodology

Because the X-ray and CT images are not well labelled, we propose an unsupervised learning approach that can be tied back to metadata that does exist, like mortality, age, BMI, etc. To accomplish this, we will first train an Auto-encoder model to create a low-dimensional representation of each image, and then use a number of different clustering methods to determine optimal groupings for these images based on their encoding. Once these groups are instantiated, we can then associate metadata about the images to each cluster to determine whether there are statistically significant attributes tied to specific clusters. If it could be proven that attributes like mortality rate or success with intubation are linked to certain clusters, that information could be incredibly valuable for clinical outcomes.
 
## Potential Results
 
The ideal results of this research will be an algorithm that can receive an unseen CT scan image, and give an immediate prognosis to assist doctors.  This prognosis should be able to indicate the severity of the disease, getting severe cases treated faster. 

## Cluster Analysis (Long)

 We can divide the cluster analysis into 3 parts: 1) clustering tendency, 2) number of clusters, and 3) clustering performance.

### Clustering Tendency

Not all data is meant to be clustered (pause for dramatic effect). How would you cluster a uniform distribution? Thus, we must first check to make sure the data is not uniformly distributed. We can do this via the Hopkins test, which yields the Hopkins Statistic, H. We can calculate this using the library pyclustertend (https://pypi.org/project/pyclustertend/). If H>.5 it is very likely the data can be clustered.

### Number of Clusters, K

We can find the optimal number of clusters using the Silhouette coefficient, S, discussed in class. There is an sklearn implementation of S (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html). The higher S is, the more optimal the clustering. K can also be added as a hyper-parameter where -S is the loss function. Packages like scikit-optimize (https://scikit-optimize.github.io/stable/) allow for Bayesian-based optimization of custom hyper-parameters and custom loss functions. 

### Clustering Performance

Once optimal clusters have been found, it is desirable to understand what factors the model is reliably clustering. In fact, there is no guarantee that the model will split the images into COVID and non-COVID cases. Since a portion of the labelled images will be in the test set we will look at the distribution of COVID/non-COVID case ratio among clusters. 

## Discussion

## Challenges
Our model will be big and running for a large data set which would require significant CPU resources and time.
Though we have finalised our analysis to be based on CT scans, as we progress in the project we might need to consider X-Ray scans due to data availability.Model able to cluster scans which have similar metadata
Implications
Inability to find sufficient processing power for our models to run, we might not be able to perform our analysis on the complete data set. 
If our model gives a near accurate prognosis for the testing data set, our project will be able to provide a means to speed up the process of COVID detection.

## Cluster Analysis (Short)

We will perform three types of clustering analyses: clustering tendency, number of clusters, clustering performance.

### Clustering Tendency

Not all data is meant to be clustered. How would you cluster a uniform distribution? Thus, we must first check to make sure the data is not uniformly distributed. We can do this via the Hopkins test, which yields the Hopkins Statistic, H. If H>.5 it is very likely the data can be clustered.

### Number of Clusters, K

We can find the optimal number of clusters using the Silhouette coefficient, S. The higher S is, the more optimal the clustering. K can also be added as a hyper-parameter where -S is the loss function. 

### Clustering Performance

Once optimal clusters have been found, it is desirable to understand what the properties of each cluster are. In fact, there is no guarantee that the model will split the images into COVID and non-COVID cases. Since a portion of the labelled images will be in the test set we will look at the distribution of COVID/non-COVID case ratio among clusters. 

## References

https://arxiv.org/pdf/2003.05991.pdf 
https://link.springer.com/chapter/10.1007/978-3-642-41822-8_15
https://www.researchgate.net/profile/Xifeng_Guo/publication/320658590_Deep_Clustering_with_Convolutional_Autoencoders/links/5a2ba172aca2728e05dea395/Deep-Clustering-with-Convolutional-Autoencoders.pdf
