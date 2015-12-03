# Machine-Learning-Final-Project

## Get Started

* Create a train directory and put training data into the folder
* Create a test directory and put testing data into the folder

## Overview

For this project, you will be developing a system for predicting someone’s gender (male/female) from the language of their tweets and the image they post with their twitter profile. You will be given a training dataset of 5,000 labeled training samples and tested on around 5,000 testing samples. The features of the dataset are described in more detail in the slides.
The format of the project is a competition, with live leaderboards.

## Links
* Leaderboard
   - http://www.seas.upenn.edu/~cis520/fall15/leaderboard.html
* Website
   - https://alliance.seas.upenn.edu/~cis520/dynamic/2014/wiki/index.php?n=Project.Project

## Evaluation
* Error metric
   - Your predictions will be evaluated based on their L_0 Err. (I.e. the number of predictions you get wrong)
   - Your code should produce an Nx1 vector of predictions, each of which is 0 or 1.

## Deadlines
* Nov. 19 Submit group.txt – group name (1%)
* Nov. 22 Beat 1st baseline (9%)
* Dec. 2 Beat 2nd baseline (20%)
* Dec. 5 Submit final prediction code (50%)
   - **By the final submission (Dec 5), implement at least 4 of the following:**
      - A generative method (NB, HMMs, k-means clustering, GMMs, etc.)
      - A discriminative method (logistic regression, decision trees, SVMs, etc.)
      - An instance based method (kernel regression, k-nearest neighbors, etc.)
      - Your own regularization method (other than the standard L0, L1 or L2 penalty)
      - A semi-supervised dimensionality reduction of the data
* Dec. 10 Submit final report (20%)
