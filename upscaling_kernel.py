from utils import *
import numpy as np
from scipy import linalg



if __name__=='__main__':
    model = load_model('Models/Model_12KHz_97%.yaml', 'Models/Model_12KHz_97%.h5')

    # conv kernels of fisrt layer
    kernels = get_kernels(model.layers[0].get_weights())
    kernel = kernels[0]

    n = 6
    matrix = linalg.toeplitz(np.hstack([kernel, np.zeros(n-1)]), np.hstack([kernel[0], np.zeros(n-1)]))