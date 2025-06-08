<img src="./images/label.png" width="210px">

# UoMo: A Universal Model of Mobile Traffic Forecasting for Wireless Network Optimization
This is the official implementation of our foundation model for mobile traffic data, accepted by the KDD 2025 ADS track.


# Overall framework

<img src="./images/framework.png">
Our model adopts a three-stage paradigm consisting of <code>tokenization</code>,  `pre-training` , and fine-tuning. The tokenization stage transforms the data into an N-format representation. The pre-training stage learns the fundamental features of the data, while the fine-tuning stage incorporates the number of users and the distribution of POIs as conditional inputs.
