def brtrade(prices, theta0, theta, fee):
    """
    evaluation of performance:
    
    using our third set of prices, we estimate dp at each time interval,
    if dp > t and current position <= 0 , we buy
    if dp < -t and current position >= 0, we sell
    else, do:nothing

    trade using the above algorithm. returns expected profit

    given a list of prices
    assumes k-means clustered patterns have already been calculated
    assumes parameters w_i have already been calculated
    position is 0 or 1 (we have nothing, we have a bitcoin)
    bank is the amount of cash we have(# change bank to cash)
    threshold is the threshold for buying/sellingdefined in the paper    
    """
    # test caes
    # test caes
    if not prices:
        print("Lack of price data or other parameters, please enter it.")
        return
    if theta0 is None:
        theta0 =  -6.30563157181757e-05
    if theta is None:
        theta = [-0.032, 0.0138181772477096,-0.0904970244446115]
    position = 0
    cash = 0
    error = 0
    buy = []
    sell = []
    counttotal = 0
    counts = 0
    temp = 0
    start = 720
    fee = 0
    if fee == 0:
        buy_fee = 0.001
        sell_fee = 0.003
    else:
        buy_fee = fee*prices[0]/100
        sell_fee = buy_fee
        
    i = start 
    while i < len(prices):
        price180 = stats.zscore(prices[i-180:i], ddof=1)      
        price360 = stats.zscore(prices[i-360:i], ddof=1)      
        price720 = stats.zscore(prices[i-720:i], ddof=1)
       
        dp1 = bayesian(price180, kmeans180s)
        dp2 = bayesian(price360, kmeans360s)
        dp3 = bayesian(price720, kmeans720s)
        dp = theta0 +  theta[0]*dp1 + theta[1]*dp2 + theta[2]*dp3
        
    #  compare price at t+1 with predicted price jump     
        error = error + abs(prices[i]-prices[i-1]-dp)
        # trade coins strategy:
        # buy
        if dp > buy_fee and position == 0:
            position = 1
            temp = prices[i-1]
            print('Buying at {}'.format(temp))
            buy.append(i) # save the time for buying
        
        if dp < -sell_fee and position == 1:
            position = 0
            cash = cash + prices[i-1] - temp
            print('Selling at {}'.format(prices[i]))
            sell.append(i) # save the time for selling
            counttotal += 1
            if prices[i-1]-temp > 0:
                counts += 1

        i += 1
        
    if position == 1:
        cash += prices[-2] - temp
        print ('Final sale at {}'.format(prices[-2]))
        sell.append(len(prices)-1)
        counttotal += 1
        if prices[-2]-temp > 0:
            counts += 1
        
    prob = (counts/counttotal)*100
    res = [error, prob, sell, buy, cash]
    return res
    
    
        
    
    
    

