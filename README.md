# chemeng_4h03_project

## Problem Introduction:
Handwriting recognition software has significant potential in improving the quality-of-life of individuals with visual impairment. Furthermore, the same technology could be used for the purpose of digitally preserving archival handwritten documents for future generations (e.g. Don Woods’ costing tables). These examples represent potential applications for handwriting recognition, which has sparked recent research interest in the past decade. Nowadays, the technology is significantly advanced, with technologies such as Google’s WaveNet text-to-speech software [1]. With this project, we would like to investigate some of the basic statistical and machine learning methods applied in the development and implementation of handwriting recognition technologies.

## Data Source:
The data used for our project will come from the Semeion Handwritten Digit Data Set available from UC Irvine’s Machine Learning Repository. This dataset contains digits handwritten by 80 different people for a total of 1593 samples. Each person wrote the digits 0 
– 9 at least twice: once accurately and once quickly. The original grayscale image pixel values were converted to Boolean based on a threshold value of 127. The 16 x 16 digits are stored in rows vectors, where the last 10 elements in the row are binary values used to indicate the true value of the digit [2].

Ultimately, the goal of this project is to identify the digits written by one of the group members, a classmate or the course instructor. Through this project we hope to learn key concepts from data pre-processing to model evaluation. For example, during the training and testing of the model we will need to determine how to select a proper number of cross validation folds which result in the best classification. The source states that a 50:50 train to test split is best, but this should be confirmed for the specific classification method we choose.

## Preliminary Ideas:
The primary objective of this project is to develop a predictive model that would be able to identify any handwritten digit with a high degree of accuracy. Some tools that we could use to achieve this objective include:
* Latent variable methods (LVMs) such as PCA and PLS to reduce the dimensionality of our data and make model predictions within a reasonable tolerance of error.
* The application of dimension-reduction data visualization tools explored early in the course to visualize UC Irvine’s data set (i.e. PCA score plots, loading contribution plots, etc.)
* Data clustering algorithms such as 𝑘-Means Clustering to identify the effect of handwriting style on the model’s predictive performance
* Boolean decision algorithms such as decision trees that look for certain features in the images that the model uses to guide its predictions
* Validating our model using a testing set, with the intention of further validation using scans of handwriting samples of group members, course classmates, and the course instructor.

This term project is intended to be coded on Python as an opportunity for the team to explore the similarities and differences in syntax beyond MATLAB.

Proper usage of the abovementioned tools ties directly to course concepts such model fitting and validation, data visualization, and feature selection. Based on the results of our model, we expect to learn more about the application of the same concepts to a broader set of problems such as image and voice recognition.

## References
[1]
A. van den Oord and S. Dieleman, "WaveNet: A generative model for raw audio," DeepMind, 8 September 2016. [Online]. Available: https://deepmind.com/blog/article/wavenet-generative-model-raw-audio. [Accessed 14 February 2020].

[2]
Machine Learning Repository, "Semeion Handwritten Digit Data Set," University of California - Irvine, [Online]. Available: https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit. [Accessed 14 February 2020].
