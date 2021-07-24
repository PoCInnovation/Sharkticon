import numpy as np

def computeAnomalieScore(reelPacket, predictPacket):
    anomalie = 0
    for i, j in zip(reelPacket, predictPacket):
        if i != j:
            anomalie += 1
    return anomalie