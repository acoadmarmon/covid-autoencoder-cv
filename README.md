# COVID-19 CT Scan Autoencoder Computer Vision

<figure class="video_container">
  <video controls="true" allowfullscreen="true" poster="covid-autoencoder-cv.jpg">
    <source src="covid-autoencoder-cv.mp4" type="video/mp4">
  </video>
</figure>


## Introduction/Background
 
COVID-19 has infected over 30 million people and caused around a million deaths worldwide. The virus can infect the upper or lower part of the respiratory tract. Doctors have reported signs of respiratory inflammation on a chest CT scan of COVID-19 patients. Machine learning approaches on CT scan images can help differentiate between healthy and COVID-19 patients, and also predict prognosis of COVID-19 infected patients. An automated prognosis prediction model could help physicians quickly treat patients.
 
## Problem definition
 
Develop a machine learning approach to aid the differentiation between healthy and COVID-19 patients, and in the prognosis of COVID-19 infected patients using CT scan images.

## Methodology

Because the CT images are not well labelled, we propose an unsupervised learning approach that can be tied back to existing metadata, like mortality, age, BMI, etc. To accomplish this, we will train an Autoencoder model to create a low-dimensional representation of each image, and then use different clustering methods to determine optimal groupings for these images based on their encoding. Once these groups are instantiated, we can then associate image metadata to each cluster to determine whether there are statistically significant attributes tied to specific clusters. If it could be proven that attributes like mortality rate or success with intubation are linked to certain clusters, that information could be incredibly valuable for clinical outcomes.

## Cluster Analysis

### Clustering Tendency

Not all data is meant to be clustered. How would you cluster a uniform distribution? Thus, we must first check to make sure the data is not uniformly distributed. We can do this using the Hopkins Statistic.

### Number of Clusters, K

We can find the optimal number of clusters using the Silhouette coefficient, S. The higher S is, the more optimal the clustering. K can also be added as a hyper-parameter where -S is the loss function. 

### Clustering Performance

Once optimal clusters are found, it’s desirable to understand the properties of each cluster. Since a portion of the labelled images will be in the test set we will look at the distribution of COVID/non-COVID case ratio among clusters. 

## Potential Results
 
The ideal results of this research will be an algorithm that can receive an unseen CT scan image, and give an immediate prognosis to assist doctors.  This prognosis should be able to indicate the severity of the disease, getting severe cases treated faster. 

## Discussion

## Challenges
Running a large data set which would require significant CPU resources and time. Inability to find sufficient processing power for our models to run, we might not be able to perform our analysis on the complete data set. Also, though we have finalised our analysis to be based on CT scans, as we progress in the project we might need to consider X-Ray scans due to data availability. Our Model should be able to cluster scans which have similar metadata. If our model is unable to give a near accurate prognosis for the testing data set, our project may not present meaningful results.


## References

1. Bank, Dor, Noam Koenigstein, and Raja Giryes. "Autoencoders." arXiv preprint arXiv:2003.05991 (2020)
2. Song, Chunfeng, et al. "Auto-encoder based data clustering." Iberoamerican congress on pattern recognition. Springer, Berlin, Heidelberg, 2013
3. Guo, Xifeng, et al. "Deep clustering with convolutional autoencoders." International conference on neural information processing. Springer, Cham, 2017
