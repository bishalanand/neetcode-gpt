import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        pass
        n=len(y_true)
        wn2=len(W2)
        wn1=len(W1)
        z1=x@np.transpose(W1)+b1
        a1=np.maximum(0,z1)
        y_dash=a1@np.transpose(W2)+b2
        loss=np.mean((y_dash-y_true)**2)
        dw2=[]
        db2=(2/n)*(y_dash-y_true)
        for i in range(wn2):
            dw_2=(2/n)*(y_dash-y_true)*a1
            dw2.append(dw_2)

        dw1=[]
        db1=(np.transpose(W2)@db2)*(z1>0)
        for i in range(len(W1)):
            d_w1=db1[i]*np.array(x)
            dw1.append(d_w1)

        return {
            'loss': round(loss,4),
            'dW1': np.round(dw1,4),
            'db1': np.round(db1,4),
            'dW2': np.round(dw2,4),
            'db2': np.round(db2,4)
        }
