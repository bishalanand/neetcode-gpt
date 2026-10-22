import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import math


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        pass
        z=np.dot(x,w)+b
        y_dash=1/(1+math.e**(-z))
        dlw=[]
        for i in (x):
            mis=(y_dash-y_true)*(1-y_dash)*y_dash*i
            dlw.append(np.round(mis,5))
        
        dlb=np.round((y_dash-y_true)*(1-y_dash)*y_dash,5)
        dlw_=np.array(dlw)
        return (dlw_,dlb)


