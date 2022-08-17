import numpy as np


def create_averages_by_window(nparray, windowsize):
    # size of array
    arrsize = len(nparray)
    # iterations
    iterstodo = int(np.floor(arrsize / windowsize))
    # define new array to store data
    arrtoreturn = np.empty(iterstodo)
    # loop
    for i in range(iterstodo):
        idxtostart = i * windowsize
        arrtoreturn[i] = np.mean(nparray[idxtostart:(idxtostart + windowsize)])

    return arrtoreturn


def generate_shuffled_means(orinalarr, n_times):
    rows, cols = orinalarr.shape
    half_rows = int(rows / 2)
    shuffled_arr = np.empty([n_times, cols])
    for i in range(n_times):
        # shuffle and split in half
        np.random.shuffle(orinalarr)
        mean1 = np.mean(orinalarr[0:half_rows, ], axis=0)
        mean2 = np.mean(orinalarr[half_rows:, ], axis=0)
        shuffled_arr[i, :] = mean1 - mean2
    
    return shuffled_arr