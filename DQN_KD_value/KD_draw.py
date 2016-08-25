from __future__ import print_function
import datetime
from yahoo_finance import Share
import numpy as np
import matplotlib.pyplot as plt
START =  "2011-01-01" # Data start date
_ID = 2330 # By default, TSMC (2330)

stock = Share(str(_ID)+'.TW')
today = datetime.date.today()
stock_data = stock.get_historical(START, str(today))
print("Historical data since", START,": ", len(stock_data))
stock_data.reverse()

i = 0
while( i < len(stock_data)):
    if (int(stock_data[i].get('Volume')) <= 0):
        stock_data.remove(stock_data[i])
        i = -1
    i += 1

print("Remove the datas with zero volume, total data ",len(stock_data))

K = []
D = []
util = []
for i in xrange(len(stock_data)):
        util.append(float(stock_data[i].get('Close')))
        if i >= 8:
                assert len(util) == 9

                #----RSV----            
                if max(util) == min(util):
                        RSV = 0.0
                else:
                        RSV = (util[len(util)-1] - min(util))/(max(util)-min(util))
                #----RSV----

                #----K----
                if i == 8:
                        temp_K = 0.5*0.6667 + RSV*0.3333
                        K.append(temp_K)
                else:
                        temp_K = K[-1]*0.6667 + RSV*0.3333
                        K.append(temp_K)
                #----K----

                #----D----
                if i == 8:
                        D.append(0.5*0.6667 + temp_K*0.3333)
                else:
                        D.append(D[-1]*0.6667 + temp_K*0.3333)
                #----D----
                util.pop(0)
                assert len(util) == 8


def draw(arrlist,name):
    new = np.zeros((len(arrlist)-15), dtype=np.float)
    print(len(arrlist))
    for x in xrange(0,len(arrlist)-15):#save file
        for y in xrange(0,15):#save file
            print(arrlist[x+y])
            new[y]=arrlist[x+y]
            plt.plot(new,label='K',linewidth=5)
            plt.axis([0, 14, 0, 1])
            plt.savefig('/'+name+'/'+str(x)+'.png')
            plt.close()

def drawkd(k,d):
    newk = np.zeros((len(k)-15), dtype=np.float)
    newd = np.zeros((len(d)-15), dtype=np.float)
    for x in xrange(0,len(k)-15):#save file
        for y in xrange(0,15):#save file
            newk[y]=k[x+y]
            newd[y]=d[x+y]
            plt.plot(newk,label='K',linewidth=5,color=[0,0,1])
            plt.plot(newd,label='D',linewidth=5,color=[0,1,0])
            plt.axis([0, 14, 0, 1])
            plt.axis('off')
            plt.savefig('KD/'+str(x)+'.png')
            plt.close()            
#draw(K,"k")
#draw(D,"d")
drawkd(K,D)
