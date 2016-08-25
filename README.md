# Tensorflow work for stock prediction
Use Tensorflow to run CNN for predict stock movement. Hope to find out which pattern will follow the price rising. Different implement codes are in seperate folder.

Feature include daily close price, MA, KD, RSI, yearAvgPrice..... Detail described below. This work is just an sample to demo deep learning. The result is not well estimated.
All of the work are done by using the same stock(2330 in Taiwan stock) which are collected from yahoo.finance.

##requirement
-Tensorflow
-yahoo_finance
-numpy
if you run DQN_KD_value, DQN_cnn_image you'll need extra
-matplotlib
-cv2

##DQN_cnn_image
The major mork of this project. We feed the data(yearline,monthline, closePrice) as image and use CNN to recognize their pattern. 
file:
-DQN_draw_yearline.py  //use for make yearline img and closeprice img, and then build model.
-DQN_yearline_reward.py :to build model which should be train for about 24hr. //run DQN_draw_yearline.py first
-Test model by yearline.ipynb : There is one model exsit in saved_year_r. The code create some img to test on that.
-DQN_img_closePrice.py: build a model by closeprice img and do evaluation.

### performance


##cnn_tsc
The model is original from RobRomijnders 's work used for time series data.
http://robromijnders.github.io/CNN_tsc/
I change some parameter and use the daily close price as data
### performance

##DQN_MLP_closePrice
Some resources really help a lot while building DQN. The main different is how to set the reward function and way to train Q_network.
https://zhuanlan.zhihu.com/p/21477488
https://github.com/gliese581gg/DQN_tensorflow
My implement is under close price. This could be change to other features like RSI,KD,MA....Or, use all of them. There is CNN code that could be edit to meet the requirement (size of batch).  
### performance

##DQN_KD_value
Use KD value picture to predict.
python DQN_kd_pic.py //this call KD_draw.py and build model.

### performance


