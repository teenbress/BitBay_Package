def bayesian(x, S):
    """
    Calculates the expected price change dp_j based on a given current price.
    
    :param x: The data set of current empirical prices, ending in our current price as a list of floats.
    :type x: list

    :param S: the former emiprical prices pattern.
    :type S: int

    :param r: Tolerance. Typically 0.1 or 0.2.Here is 0.2*std(U)
    :type r: float

    :return: List[(Int, Float/None, Float/None)...]

    The outputs are the sample entropies of the input, for all epoch lengths of 0 to a specified maximum length, m.
    :rtype: list
    """
    dpj = 0
    c = -1/4
    num = 0.0
    den = 0.0
    
    for i in range(20):
        cutS = S[i, 0:(len(x))]
        distance = math.exp(c*(np.linalg.norm(x-cutS)**2))
        num += S[i, -1]*distance
        den += distance
        
    if den!=0:
        dpj = num/den
    return dpj
    

    
    
