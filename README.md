# Tensorflow work for stock prediction
Use Tensorflow to run CNN for predict stock movement. Hope to find out which pattern will follow the price rising. Different implement codes are in separate folder.
 
Feature include daily close price, MA, KD, RSI, yearAvgPrice..... Detail described as below. This work is just an sample to demo deep learning. The result is not well estimated. 
 
All of the work are done by using the same stock(2330 in Taiwan stock) which are collected from yahoo.finance. Please notice that this stock perform a stable rise these years. But our result get a little better than just random guess.

For Brief tutorial slider please check ([Distributed Tensorflow & Stock prediction](https://github.com/kimber-chen/Tensorflow-for-stock-prediction/blob/master/graph/DistributedTensorFlow%26StockPred.pdf))

For Chinese outline slider please check HERE([中文簡介](https://github.com/kimber-chen/Tensorflow-for-stock-prediction/blob/master/graph/tensorflow%E8%82%A1%E7%A5%A8%E9%A0%90%E6%B8%AC_%E5%A4%A7%E7%B6%B1.pdf))


##requirement
- Tensorflow
- yahoo_finance
- numpy
If you run DQN_KD_value, DQN_cnn_image you'll need extra
- matplotlib
- cv2

##1.DQN_CNN_image
The major work of this project. We feed data(yearline,monthline, closePrice) as image and use CNN to recognize their pattern. 
Some resources really help a lot while building DQN. The main difference is how to set the reward function and way to train Q_network.
  * https://zhuanlan.zhihu.com/p/21477488
  * https://github.com/gliese581gg/DQN_tensorflow

File:
- DQN_draw_yearline.py  :use for making yearline img and closeprice img, and then build model.
- DQN_yearline_reward.py :to build model which should be train for about 24hr. //run DQN_draw_yearline.py first
- Test model by yearline.ipynb : There is one model exsit in saved_year_r. The code create some img to test on that.
- DQN_img_closePrice.py: build a model by closeprice img and do evaluation.

### Performance
#### exp 1
  * Training: 2011~2014 15-day image with only daily close price.
  * Testing : 2016/01 ~2016/08  
  * Trading strategy: Reward=(Tomorrow's close price)-(today's close price) if predict buy. Negate the number while predicting sell. 
  * Bad performance. 
 
![alt tag](https://github.com/kimber-chen/Tensorflow-for-stock-prediction/blob/master/graph/closePrice_rst.PNG)
#### exp 2
  * Trading strategy: sell while meet +10% profit or -5% loss.
  * Baseline: Considering of the rising stock price, the baseline is the average profit takes buying times into account.
  * We get more profit then baseline. And the testing file can be used for evaluating whether to believe in the model.
  
![alt tag](https://github.com/kimber-chen/Tensorflow-for-stock-prediction/blob/master/graph/yearline_rst.PNG)
  

##2.CNN_tsc
The model is originated from RobRomijnders 's work used for time series data (http://robromijnders.github.io/CNN_tsc/) 
 I've change some parameter and use different indicators to find out when to buy the stock is good.
### Performance
####Part 1. Indicators
Feature MA can drop the loss compare with RSI and ClosePrice at training step.

![alt tag](https://github.com/kimber-chen/Tensorflow-for-stock-prediction/blob/master/graph/cnn_tsc_rst.PNG)
####Part 2. When To Buy
With 10 days MA5 as an instance.
 * Training data (2330_train_15) : 2001~2014 2330.tw. Instance labeled as 1 when it rising dramatically up to 15% in 90 days. And mix with 4 times instance labeled as 0
 * Testing data (2330_test) : 2015/07 ~ 2016/08 MA5. 
 
 After running CNN_Classifier.ipynb, Result will be visualized.
 
![alt tag](https://github.com/kimber-chen/Tensorflow-for-stock-prediction/blob/master/graph/cnn_tsc_when.PNG)

##3.DQN_MLP_closePrice

My implement is under close price. This could be change to other features like RSI,KD,MA....Or, use all of them. There is CNN code that could be edit to meet the requirement (size of batch).  
 
### Performance
Not work in closePrice. Better with other feature.
##4.DQN_KD_value
Use KD value picture to predict.
```
python DQN_kd_pic.py //this call KD_draw.py and build model.
```
### Performance
  * Training: 2011~2014 15-day K value and D value image
  * Testing : 2015/07 ~2016/07  
![alt tag](https://github.com/kimber-chen/Tensorflow-for-stock-prediction/blob/master/graph/KD_rst.PNG)

