# Tensorflow work for stock prediction
Use Tensorflow to run CNN for predict stock movement. Hope to find out which pattern will follow the price rising. Different implement codes are in seperate folder.
 
Feature include daily close price, MA, KD, RSI, yearAvgPrice..... Detail described as below. This work is just an sample to demo deep learning. The result is not well estimated. 
 
All of the work are done by using the same stock(2330 in Taiwan stock) which are collected from yahoo.finance. Please notice that this stock perform a stable rise these years. But our result get a little better than just ramdon guess.

For Chinese version slider please check HERE([中文投影片](https://github.com/kimber-chen/Tensorflow-for-stock-prediction/blob/master/graph/tensorflow%E8%82%A1%E7%A5%A8%E9%A0%90%E6%B8%AC_%E5%A4%A7%E7%B6%B1.pdf))

##requirement
- Tensorflow
- yahoo_finance
- numpy
If you run DQN_KD_value, DQN_cnn_image you'll need extra
- matplotlib
- cv2

##DQN_cnn_image
The major work of this project. We feed the data(yearline,monthline, closePrice) as image and use CNN to recognize their pattern. 
File:
- DQN_draw_yearline.py  //use for make yearline img and closeprice img, and then build model.
- DQN_yearline_reward.py :to build model which should be train for about 24hr. //run DQN_draw_yearline.py first
- Test model by yearline.ipynb : There is one model exsit in saved_year_r. The code create some img to test on that.
- DQN_img_closePrice.py: build a model by closeprice img and do evaluation.

### performance
Use only daily close price. Bad performance. 
 
![alt tag](https://github.com/kimber-chen/Tensorflow-for-stock-prediction/blob/master/graph/closePrice_rst.PNG)
  * Training: 2011~2014 15-day image with avgYearLine,avgSeasonLine,avg20daysLine. 
  * Testing : 2016/01 ~2016/08  
  * Trading :stratage: sell while meet +10% profit or -5% loss.
  * baseline:Considering of the rising stock price, the baseline is the average profit takes buying times into account.
![alt tag](https://github.com/kimber-chen/Tensorflow-for-stock-prediction/blob/master/graph/yearline_rst.PNG)
  
 That we get more profit that baseline. And the test file can use for predict whether to buy without re-training the model.

##cnn_tsc
The model is original from RobRomijnders 's work used for time series data (http://robromijnders.github.io/CNN_tsc/) 
 I've change some parameter and use the daily close price as data
### performance
Feature MA can drop the loss compare with RSI and ClosePrice at training step.
 
![alt tag](https://github.com/kimber-chen/Tensorflow-for-stock-prediction/blob/master/graph/cnn_tsc_rst.PNG)
##DQN_MLP_closePrice
Some resources really help a lot while building DQN. The main different is how to set the reward function and way to train Q_network.
  * https://zhuanlan.zhihu.com/p/21477488
  * https://github.com/gliese581gg/DQN_tensorflow
 
My implement is under closeprice. This could be change to other features like RSI,KD,MA....Or, use all of them. There is CNN code that could be edit to meet the requirement (size of batch).  
 
### performance
Not work in closePrice. Better with other feature.
##DQN_KD_value
Use KD value picture to predict.
```
python DQN_kd_pic.py //this call KD_draw.py and build model.
```
### performance
  * Training: 2011~2014 15-day K value and D value image
  * Testing : 2015/07 ~2016/07  
![alt tag](https://github.com/kimber-chen/Tensorflow-for-stock-prediction/blob/master/graph/KD_rst.PNG)

