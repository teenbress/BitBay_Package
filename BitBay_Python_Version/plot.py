
import matplotlib.pyplot as plt
import numpy as np
def result_plots(prices, buy, sell, prob, cash, error): 

    n = len(prices)
    buyP = []
    sellP = []
    for i in buy:
        buyP.append(prices3[i])
    for i in sell:
        sellP.append(prices3[i])
    x = np.arange(0, len(prices3), 1)   
    print('Error of prediction, on average: {}'.format(error/n))
    print('Sell Accuracy:{} percent, Total profit:{}'.format( prob, cash))
    print('Percent profit(approx): {}'.format(cash*100/prices[-1]))
    # create plots of buy/sell points
    # note: cannot plot when running on -nojvm flag
    plt.figure(figsize = (200, 400))
    plt.figure(dpi=180)
    plt.plot(x, prices3,color="cornflowerblue", linestyle="-",linewidth=1,zorder=1)
    plt.scatter(buy, buyP,color="red",marker='o',s = 50,zorder=2)
    plt.scatter(sell, sellP,color="limegreen",marker='o',s = 50,zorder=2)
    plt.show()
