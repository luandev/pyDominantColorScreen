from captureTest import evalScreen
import numpy as np

while True:
    unique, counts = evalScreen(False)
    print(unique[np.argmax(counts)])
    
