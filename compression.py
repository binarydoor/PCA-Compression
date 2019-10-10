import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    data = np.genfromtxt('MNIST3.csv', delimiter=',')

    # making covariance matrix
    S = np.cov(np.transpose(data))

    # finding eigenvalue and eigenvector of the covar matrix
    egval, egvec = np.linalg.eig(S)

    # plot the first largest 100 eigenvalues
    largesthundred = egval[:100]

    plt.title("Largest 100 Eigenvalue")
    plt.plot(largesthundred)
    plt.show()

    # imshow the first 4 eigenvectors
    # (automatically rescale elements in 0-255 values
    firstfour = []
    for index in range(4):
        firstfour.append(np.array(np.transpose(egvec[:, index].reshape(28, 28)), dtype=float))

    fig = plt.figure()

    ax = []
    for index in range(4):
        ax.append(plt.subplot(2, 2, (index+1)))
        ax[index].set_title("{}-th eigenvector imshow".format(index+1))
        plt.imshow(firstfour[index], cmap='gray')

    plt.show()

    # find the mean vector x
    meanX = [0.0] * 784
    for i in range(400):
        for j in range(784):
            meanX[j] += data[i][j]
    for i in range(784):
        meanX[i] = meanX[i] / 400

    # PCA with M = 0
    newX = []
    newX.append(meanX)
    temp1 = np.dot(np.transpose(data[0]), egvec[:, 0])
    temp2 = np.dot(np.transpose(meanX), egvec[:, 0])
    temp3 = temp1 - temp2
    temp4 = np.dot(temp3, egvec[:, 0])
    newX[0] = np.add(newX[0], temp4)
    finishedX = []
    finishedX.append(np.array(np.transpose(newX[0].reshape(28, 28)), dtype=float))
    fig = plt.figure()
    newax = []
    newax.append(plt.subplot(2, 2, 1))
    newax[0].set_title("Compression with M = 1")
    plt.imshow(finishedX[0], cmap='gray')

    # do the same for M=10, 50, 250
    for i in range(1, 4):
        newX.append(meanX)
        for j in range((5**(i-1))*10):
            temp1 = np.dot(np.transpose(data[0]), egvec[:, j])
            temp2 = np.dot(np.transpose(meanX), egvec[:, j])
            temp3 = temp1 - temp2
            temp4 = np.dot(temp3, egvec[:, j])
            newX[i] = np.add(newX[i], temp4)
        finishedX.append(np.array(np.transpose(newX[i].reshape(28, 28)), dtype=float))
        newax.append(plt.subplot(2, 2, (i+1)))
        newax[i].set_title("M = {}".format((5**(i-1))*10))
        plt.imshow(finishedX[i], cmap='gray')

    plt.show()

    tempX = np.array(np.transpose(np.array(meanX).reshape(28, 28)), dtype=float)
    plt.imshow(tempX, cmap='gray')
    plt.title("M=0")
    plt.show()