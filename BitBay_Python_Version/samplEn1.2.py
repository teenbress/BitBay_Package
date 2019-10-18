
import numpy as np

def samplEn(U, m, r):
    """
    Calculates an estimate of sample entropy and the variance of the estimate.

    :param U: The data set (time series) as a list of floats.
    :type U: list

    :param m: Maximum length of epoch (subseries).
    :type m: int

    :param r: Tolerance. Typically 0.1 or 0.2.Here is 0.2*std(U)
    :type r: float

    :return: List[(Int, Float/None, Float/None)...]

    The outputs are the sample entropies of the input, for all epoch lengths of 0 to a specified maximum length, m.
    :rtype: list
    """
    def _phi(m):
        x = [[U[j] for j in range(i, i + m)] for i in range(N - m + 1)]
        cnt = 0
        maxdist = np.zeros((len(x), len(x)-1))
        x = np.array(x)
        for i in range(len(x)):
            tmp = np.transpose(np.abs(x[i]-x).max(axis=1))
            maxdist[i] = tmp[tmp!= 0]
        
        for i in maxdist:
            for j in i:
                if j <= r:
                    cnt += 1
        res = cnt/((N-m)*(N-m+1))
        
        return res
    if r == 0:
        r = 0.02
    N = len(U)
    ans = -np.log(_phi(m+1) / _phi(m))      
    
    return ans
