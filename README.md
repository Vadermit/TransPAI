***TransPAI***
--------------

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)
[![repo size](https://img.shields.io/github/repo-size/Vadermit/TransPAI.svg)](https://github.com/Vadermit/TransPAI/archive/master.zip)
[![GitHub stars](https://img.shields.io/github/stars/Vadermit/TransPAI.svg?logo=github&label=Stars&logoColor=white)](https://github.com/Vadermit/TransPAI)


> **Trans**portation data online **P**rediction **A**nd **I**mputation(***TransPAI***).

> This is the code repository for paper 'Real-time Spatiotemporal Prediction and Imputation of Traffic Status Based onLSTM and Graph Laplacian Regularized Matrix Factorization' which is submitted to Transportation Research Part C: Emerging Technologies

Contents
--------

-   [Strategic aim](#strategic-aim)
-   [Tasks and challenges](#tasks-and-challenges)
-   [Overview](#overview)
-   [Proposed method](#proposed-method)
-   [Selected References](#selected-references):

Strategic aim
--------------

>Minning the spatial temporal characteristics of transportation data to predict the future transportation status. And impute possible missing entries along the real-time data collection.

Tasks and challenges
--------------
> Tasks
- ### **Online traffic data prediction and imputation**

  - Online prediction **Predict traffic status in the next time step using real-time observation data**. 
  - Online imuputation **Impute incomplete observations with the real-time data collection**. 

> Challenges
- ### **Incomplete observations**
> The data we acquired may not be complete due to detector mailfunction, data transmission error and so on. We need to mine the data characteristic and make predictions with insufficient information. There are basically two forms of data missing:

  - **Point-wise missing (PM)**: Each sensor lost observations for individual time steps at completely random. 
  - **Continuous missing (CM)**: Each sensor lost observations for continuous periods e.g. a day. 

Overview
--------------

   >Accurate prediction of traffic status in real time is critical for advanced traffic management and travel navigation guidance. There are many attempts to predict short-term traffic flows using various deep learning algorithms. Most existing prediction models are only tested on spatiotemporal data assuming no missing data entries. However, this ideal situation rarely exists in real world due to sensor or network transmission failure. Missing data is an unnegligible problem.  Previous studies either remove time series with missing entries or impute missing data before building prediction models. The former may cause insufficient data for model training, while the latter adds extra computational burden and the imputation accuracy has direct impacts on the prediction performance. 


Proposed method
--------------
We propose a framework based on Matrix Factorization which is able to make spatiotemporal predictions using raw incomplete data and perform online data imputation simultaneously. We innovatively design a spatial and temporal regularized matrix factorization model, namely LSTM-GL-ReMF, as the key component of the framework.  

- ### **LSTM Graph Laplacian Regularized Matrix Factorization (LSTM-GL-ReMF)**
On the basis of TRMF, we propose a novel LSTM and Graph Laplacian regularized matrix factorization (LSTM-GL-ReMF). In LSTM-GL-ReMF, its temporal regularizer depends on the state-of-the-art Long Short-term Memory (LSTM) model, and the spatial regularizer is designed based on Graph Laplacian (GL) spatial regularization. These regularizers enable the incorporation of complex spatial and temporal dependence into matrix factorization process for more accurate prediction performance. The illustration of LSTM-GL-ReMF is presented as:

<p align="center">
<img align="middle" src="https://github.com/Vadermit/TransPOL/blob/master/images/lstmremf.png" width="650" />
</p>

The proposed MF model can be easily extended to LSTM Regularized Matrix Factorization (LSTM-ReMF) model by neglectng the Graph Laplacian spatial regularizer. LSTM-ReMF and LSTM-GL-ReMF can also be extended to there tensor deomcomposition version LSTM-ReTF and LSTM-GL-ReTF respectively by following the tensor Canonical Polyadic (CP) decomposition method.

- ### **An online prediction and imputation framework for spatiotemporal traffic status**
We propose a framework based on the aforementioned LSTM-(GL-)ReMF/TF models which is able to make spatiotemporal predictions using raw incomplete data and perform online data imputation simultaneously. As shown in the figure below, the framework basically consists of two steps: static training and dynamic prediction and imputation.

<p align="center">
<img align="middle" src="https://github.com/Vadermit/TransPOL/blob/master/images/framework_sub.png" width="850" />
</p>

Model Comparison
--------------
- ### **Our proposed models**
<table>
  <tr>
    <td>Proposed Models</td>
    <td colspan="2">Seattle Speed Data</td>
    <td colspan="2">Shanghai Pollutant Data</td>
    <td>Code Format</td>
  </tr>
  <tr>
    <td>Online Tasks:</td>
    <td>Prediction</td>
    <td>Imputation</td>
    <td>Prediction</td>
    <td>Imputation</td>
    <td> </td>
  </tr>
  <tr>
  <td><b>LSTM-ReMF</b></td>
  <td>✔</td>
  <td>✔</td>
  <td>✔</td>
  <td>✔</td>
  <td>Jupyter Notebook</td>
  </tr>
  <tr>
  <td><b>LSTM-GL-ReMF</b></td>
  <td>✔</td>
  <td>✔</td>
  <td>❌</td>
  <td>❌</td>
  <td>Jupyter Notebook</td>
  </tr>
  <tr>
  <td><b>LSTM-ReTF</b></td>
  <td>❌</td>
  <td>❌</td>
  <td>✔</td>
  <td>✔</td>
  <td>Jupyter Notebook</td>
  </tr>
  <tr>
  <td><b>LSTM-GL-ReTF</b></td>
  <td>❌</td>
  <td>❌</td>
  <td>✔</td>
  <td>✔</td>
  <td>Jupyter Notebook</td>
  </tr>
</table>

- ### **Baseline models**
<table>
  <tr>
    <td>Proposed Models</td>
    <td colspan="2">Seattle Speed Data</td>
    <td colspan="2">Shanghai Pollutant Data</td>
    <td>Code Format</td>
  </tr>
  <tr>
    <td>Online Tasks:</td>
    <td>Prediction</td>
    <td>Imputation</td>
    <td>Prediction</td>
    <td>Imputation</td>
    <td> </td>
  </tr>
  <tr>
  <td><b>TRMF</b></td>
  <td>✔</td>
  <td>✔</td>
  <td>✔</td>
  <td>✔</td>
  <td>Jupyter Notebook</td>
  </tr>
  <tr>
  <td><b>BTMF</b></td>
  <td>✔</td>
  <td>✔</td>
  <td>✔</td>
  <td>✔</td>
  <td>Jupyter Notebook</td>
  </tr>
  <tr>
  <td><b>LSTM</b></td>
  <td>✔</td>
  <td>❌</td>
  <td>✔</td>
  <td>❌</td>
  <td>Jupyter Notebook</td>
  </tr>
  <tr>
  <td><b>GRU-D</b></td>
  <td>✔</td>
  <td>❌</td>
  <td>✔</td>
  <td>❌</td>
  <td>Jupyter Notebook</td>
  </tr>
  <tr>
  <td><b>GCN-DDGF</b></td>
  <td>✔</td>
  <td>❌</td>
  <td>✔</td>
  <td>❌</td>
  <td>Python Code</td>
  </tr>
  <td><b>TGC-LSTM</b></td>
  <td>✔</td>
  <td>❌</td>
  <td>❌</td>
  <td>❌</td>
  <td>Jupyter Notebook</td>
  </tr>
</table>


Selected references
--------------

- ### **Spatio-temporal forecasting**

  - Bing Yu, Haoteng Yin, Zhanxing Zhu, 2017. [*Spatio-temporal graph convolutional networks: a deep learning framework for traffic forecasting*](https://arxiv.org/pdf/1709.04875.pdf). arXiv. ([appear in IJCAI 2018](https://www.ijcai.org/proceedings/2018/0505.pdf))

  - Cui, Zhiyong and Henrickson, Kristian and Ke, Ruimin and Wang, Yinhai, 2019. [*Traffic graph convolutional recurrent neural network: A deep learning framework for network-scale traffic learning and forecasting](https://www.researchgate.net/publication/323302472_Traffic_Graph_Convolutional_Recurrent_Neural_Network_A_Deep_Learning_Framework_for_Network-Scale_Traffic_Learning_and_Forecasting). IEEE Transactions on Intelligent Transportation Systems.
  
  - Lin, Lei and He, Zhengbing and Peeta, Srinivas, 2018. [*Predicting station-level hourly demand in a large-scale bike-sharing network: A graph convolutional neural network approach](https://www.sciencedirect.com/science/article/pii/S0968090X18300974). Transportation Research Part C: Emerging Technologies, 97: 258-276.
  
  - Geng, Xu and Li, Yaguang and Wang, Leye and Zhang, Lingyu and Yang, Qiang and Ye, Jieping and Liu, Yan, 2019. [*Spatiotemporal multi-graph convolution network for ride-hailing demand forecasting](https://www.aaai.org/ojs/index.php/AAAI/article/view/4247). Proceedings of the AAAI Conference on Artificial Intelligence, 33: 3656-3663.

  - Qi, Zhongang and Wang, Tianchun and Song, Guojie and Hu, Weisong and Li, Xi and Zhang, Zhongfei, 2018. [*Deep air learning: Interpolation, prediction, and feature analysis of fine-grained air quality](https://ieeexplore.ieee.org/abstract/document/8333777). IEEE Transactions on Knowledge and Data Engineering, 30: 2285-2297.

- ### **Prediction for dataset with missing values**
  - Hu, Jian and Xin, Xin and Guo, Ping, 2017. [*LSTM with Matrix Factorization for Road Speed Prediction](https://link.springer.com/chapter/10.1007/978-3-319-59072-1_29), 10.1007/978-3-319-59072-1_29.
  
  - Sridevi, S and Rajaram, S and Parthiban, C and SibiArasan, S and Swadhikar, C, 2011. [*Imputation for the analysis of missing values and prediction of time series data](https://ieeexplore.ieee.org/abstract/document/5972466/). 2011 International Conference on Recent Trends in Information Technology (ICRTIT), 1158-1163.
  
  - Purwar, Archana and Singh, Sandeep Kumar, 2015. [*Hybrid prediction model with missing value imputation for medical data](https://www.sciencedirect.com/science/article/abs/pii/S0957417415001578#:~:text=This%20paper%2C%20presents%20a%20novel,means%20clustering%20with%20Multilayer%20Perceptron.). Expert Systems with Applications, 42: 5621-5631.

  - Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, Yan Liu, 2018. [*Recurrent neural networks for multivariate time series with missing values*](https://doi.org/10.1038/s41598-018-24271-9). Scientific Reports, 8(6085).

  - Oren Anava, Elad Hazan, Assaf Zeevi, 2015. [*Online time series prediction with missing data*](http://proceedings.mlr.press/v37/anava15.pdf). Proceedings of the 32nd International Conference on Machine Learning (*ICML 2015*), 37: 2191-2199.

- ### **Matrix factorization**

 - Hsiang-Fu Yu, Nikhil Rao, Inderjit S. Dhillon, 2016. [*Temporal regularized matrix factorization for high-dimensional time series prediction*](http://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf). 30th Conference on Neural Information Processing Systems (*NIPS 2016*), Barcelona, Spain. [[Matlab code](https://github.com/rofuyu/exp-trmf-nips16)]

  - Lijun Sun and Xinyu Chen, 2019. [*Bayesian Temporal Factorization for Multidimensional Time Series Prediction](https://arxiv.org/abs/1910.06366). ArXiv, abs/1910.06366.

  - San Gultekin, John Paisley, 2019. [*Online Forecasting Matrix Factorization*](https://ieeexplore.ieee.org/document/8590686/). IEEE Transactions on Signal Processing, 67(5): 1223-1236. [[Python code](https://github.com/chloemnge/online_learning)]

  - Nikhil Rao, Hsiangfu Yu, Pradeep Ravikumar, Inderjit S Dhillon, 2015. [*Collaborative filtering with graph information: Consistency and scalable methods*](http://www.cs.utexas.edu/~rofuyu/papers/grmf-nips.pdf). Neural Information Processing Systems (*NIPS 2015*). [[Matlab code](http://bigdata.ices.utexas.edu/publication/collaborative-filtering-with-graph-information-consistency-and-scalable-methods/)]

  

- ### **Bayesian matrix and tensor factorization**

  - Ruslan Salakhutdinov, Andriy Mnih, 2008. [*Bayesian probabilistic matrix factorization using Markov chain Monte Carlo*](https://www.cs.toronto.edu/~amnih/papers/bpmf.pdf). Proceedings of the 25th International Conference on Machine Learning (*ICML 2008*), Helsinki, Finland. [[Matlab code (official)](https://www.cs.toronto.edu/~rsalakhu/BPMF.html)] [[Python code](https://github.com/LoryPack/BPMF)] [[Julia and C++ code](https://github.com/ExaScience/bpmf)] [[Julia code](https://github.com/RottenFruits/BPMF.jl)]

  - Liang Xiong, Xi Chen, Tzu-Kuo Huang, Jeff Schneider, Jaime G. Carbonell, 2010. [*Temporal collaborative filtering with Bayesian probabilistic tensor factorization*](https://www.cs.cmu.edu/~jgc/publication/PublicationPDF/Temporal_Collaborative_Filtering_With_Bayesian_Probabilidtic_Tensor_Factorization.pdf). Proceedings of the 2010 SIAM International Conference on Data Mining. SIAM, pp. 211-222.

  - Qibin Zhao, Liqing Zhang, Andrzej Cichocki, 2015. [*Bayesian CP factorization of incomplete tensors with automatic rank determination*](https://doi.org/10.1109/TPAMI.2015.2392756). IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(9): 1751-1763.

  - Piyush Rai, Yingjian Wang, Shengbo Guo, Gary Chen, David B. Dunsun,	Lawrence Carin, 2014. [*Scalable Bayesian low-rank decomposition of incomplete multiway tensors*](http://people.ee.duke.edu/~lcarin/mpgcp.pdf). Proceedings of the 31st International Conference on Machine Learning (*ICML 2014*), Beijing, China.


Our blog posts (in Chinese)
--------------

  - [时序矩阵分解 | 时序数据修补与预测](https://zhuanlan.zhihu.com/p/56105537), by Jamie Yang (杨津铭).
  

License
--------------

This work is released under the MIT license.
