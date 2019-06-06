***TransPOL***
--------------

>**Trans**portation data **O**n**L**ine **P**rediction (***TransPOL***).

Contents
--------

-   [Strategic aim](#strategic-aim)
-   [Tasks and challenges](#tasks-and-challenges)
-   [What we care about](#what-we-care-about)
-   [Selected References](#selected-references):


Strategic aim
--------------

>Minning the spatial temporal characteristics of transportation data to predict the future transportation status. And the saptial temporal characteristics are constantly updated and calibrated with the acquisition of new observation data.

Tasks and challenges
--------------
> Tasks
- ### **Online traffic prediction**

  - Forecasting **without missing values**. (★★★)
  - Forecasting **with incomplete observations**. (★★★★★)

> Challenges
- ### **Incomplete observations**
> The data we acquired may not be complete due to detector mailfunction, data transmission error and so on. We need to mine the data characteristic and make predictions with insufficient information. There are basically two forms of data missing:

  - **Random missing**: Each sensor lost their observations at completely random. (★★★)
  - **Non-random missing**: Each sensor lost their observations during several days. (★★★★)


What we care about!
--------------

- Best algebraic structure for spatial teemporal data prediction.
- The context of urban transportation (e.g., biases).
- Better minning of saptial temporal data characteristic.
- Data noise avoidance.
- Competitive data prediction performance.
- Capable of various missing data scenarios.

Overview
--------------

   >With the development and application of intelligent transportation systems, large quantities of urban traffic data are collected on a continuous basis from various sources, such as loop detectors, cameras, and floating vehicles. These data sets capture the underlying states and dynamics of transportation networks and the whole system and become beneficial to many traffic operation and management applications, including routing, signal control, travel time prediction, and so on. The massive data we acquired gives us the opportunity to look into urban mobility and to mine patterns or characteristics of it. With finely acquired patterns and characteristics, we are able to precisely predict the future traffic status.



Selected references
--------------

- ### **Spatio-temporal forecasting**

  - San Gultekin, John Paisley, 2019. [*Online Forecasting Matrix Factorization*](https://ieeexplore.ieee.org/document/8590686/). IEEE Transactions on Signal Processing, 67(5): 1223-1236. [[Python code](https://github.com/chloemnge/online_learning)]

  - Bing Yu, Haoteng Yin, Zhanxing Zhu, 2017. [*Spatio-temporal graph convolutional networks: a deep learning framework for traffic forecasting*](https://arxiv.org/pdf/1709.04875.pdf). arXiv. ([appear in IJCAI 2018](https://www.ijcai.org/proceedings/2018/0505.pdf))

  - Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, Yan Liu, 2018. [*Recurrent neural networks for multivariate time series with missing values*](https://doi.org/10.1038/s41598-018-24271-9). Scientific Reports, 8(6085).

  - Oren Anava, Elad Hazan, Assaf Zeevi, 2015. [*Online time series prediction with missing data*](http://proceedings.mlr.press/v37/anava15.pdf). Proceedings of the 32nd International Conference on Machine Learning (*ICML 2015*), 37: 2191-2199.

- ### **Matrix factorization**

  - Nikhil Rao, Hsiangfu Yu, Pradeep Ravikumar, Inderjit S Dhillon, 2015. [*Collaborative filtering with graph information: Consistency and scalable methods*](http://www.cs.utexas.edu/~rofuyu/papers/grmf-nips.pdf). Neural Information Processing Systems (*NIPS 2015*). [[Matlab code](http://bigdata.ices.utexas.edu/publication/collaborative-filtering-with-graph-information-consistency-and-scalable-methods/)]

  - Hsiang-Fu Yu, Nikhil Rao, Inderjit S. Dhillon, 2016. [*Temporal regularized matrix factorization for high-dimensional time series prediction*](http://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf). 30th Conference on Neural Information Processing Systems (*NIPS 2016*), Barcelona, Spain. [[Matlab code](https://github.com/rofuyu/exp-trmf-nips16)]


- ### **Bayesian matrix and tensor factorization**

  - Ruslan Salakhutdinov, Andriy Mnih, 2008. [*Bayesian probabilistic matrix factorization using Markov chain Monte Carlo*](https://www.cs.toronto.edu/~amnih/papers/bpmf.pdf). Proceedings of the 25th International Conference on Machine Learning (*ICML 2008*), Helsinki, Finland. [[Matlab code (official)](https://www.cs.toronto.edu/~rsalakhu/BPMF.html)] [[Python code](https://github.com/LoryPack/BPMF)] [[Julia and C++ code](https://github.com/ExaScience/bpmf)] [[Julia code](https://github.com/RottenFruits/BPMF.jl)]

  - Liang Xiong, Xi Chen, Tzu-Kuo Huang, Jeff Schneider, Jaime G. Carbonell, 2010. [*Temporal collaborative filtering with Bayesian probabilistic tensor factorization*](https://www.cs.cmu.edu/~jgc/publication/PublicationPDF/Temporal_Collaborative_Filtering_With_Bayesian_Probabilidtic_Tensor_Factorization.pdf). Proceedings of the 2010 SIAM International Conference on Data Mining. SIAM, pp. 211-222.

  - Qibin Zhao, Liqing Zhang, Andrzej Cichocki, 2015. [*Bayesian CP factorization of incomplete tensors with automatic rank determination*](https://doi.org/10.1109/TPAMI.2015.2392756). IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(9): 1751-1763.

  - Piyush Rai, Yingjian Wang, Shengbo Guo, Gary Chen, David B. Dunsun,	Lawrence Carin, 2014. [*Scalable Bayesian low-rank decomposition of incomplete multiway tensors*](http://people.ee.duke.edu/~lcarin/mpgcp.pdf). Proceedings of the 31st International Conference on Machine Learning (*ICML 2014*), Beijing, China.


- ### **Graph neural network**

  - [*How to do Deep Learning on Graphs with Graph Convolutional Networks (Part 1: A High-Level Introduction to Graph Convolutional Networks)*](https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780). blog post.

- ### **Missing data imputation**

Our blog posts (in Chinese)
--------------

  - [时序矩阵分解 | 时序数据修补与预测](https://zhuanlan.zhihu.com/p/56105537), by Jamie Yang (杨津铭).
