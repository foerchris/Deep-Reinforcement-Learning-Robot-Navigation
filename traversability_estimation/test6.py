import matplotlib.pyplot as plt
import numpy as np
import math as m
import torch
import cv2



array = np.zeros( (100, 100), np.uint8 )

cv2.imshow('image',array)
cv2.waitKey(2)
cv2.destroyAllWindows()
