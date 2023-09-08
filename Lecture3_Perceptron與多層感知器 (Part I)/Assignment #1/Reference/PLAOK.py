#PLAOK.py
import numpy as np
import matplotlib.pyplot as plt

xl1, yl1 = -3, -3
xl2, yl2  = 3, 3


def plot_data_and_line(w1, w2, j):
    w1, w2 = float(w1), float(w2)
    if w2 != 0 :
        y1, y2 = (-w1*(xl1))/w2, (-w1*(xl2))/w2
        vx1, vy1 = [xl1,xl2,xl2,xl1,xl1], [y1,y2,yl2,yl2,y1]
        vx2, vy2 = [xl1,xl2,xl2,xl1,xl1], [y1,y2,yl1,yl1,y1]
    elif w1 != 0:
        vx1, vy1 = [xl2,0,0,xl2,xl2], [yl1,yl1,yl2,yl2,yl1]
        vx2, vy2 = [xl1,0,0,xl1,xl1], [yl1,yl1,yl2,yl2,yl1]
    if  w2 > 0 or ( w2 == 0 and w1 > 0):
        c1, c2 = 'b','r'
    else:
        c1, c2 = 'r','b'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(X1[Y1 > 0], X2[Y1 > 0], s = 80, c = 'b', marker = "o")
    plt.scatter(X1[Y1<= 0], X2[Y1<= 0], s = 80, c = 'r', marker = "^")
    plt.scatter(w1, w2, s=20, c='g', marker="x")
    plt.text(w1 + 0.1, w2, ("w%s = (%s, %s)")%(j+1, w1, w2), color='k', size=10, fontweight='bold')
    plt.fill(vx1, vy1, c1, alpha = 0.25)
    plt.fill(vx2, vy2, c2, alpha = 0.25)
    x = np.linspace(0, w1)
    y = np.linspace(0, w2)
    plt.plot(x, y, 'g-')
    plt.grid(True)
    plt.title("iteration " + str(j+1))
    ax.set_xlim(xl1, xl2)
    ax.set_ylim(yl1, yl2)
    fig.set_size_inches(6, 6)
    plt.show()

def getDataSet(filename):
    dataSet = open(filename, 'r')
    dataSet = dataSet.readlines()
    num = len(dataSet)
    Y1 = np.zeros((num, 1))
    X1 = np.zeros((num, 1))
    X2 = np.zeros((num, 1))
    for i in range(num):
        data = dataSet[i].strip().split()
        X1[i] = np.float(data[0])
        X2[i] = np.float(data[1])
        Y1[i] = np.float(data[2])
    return num, X1, X2, Y1

def sign(x, w):
    if np.dot(x, w) >= 0:
        return 1
    else:
        return -1

def PLA_Naive(X1,X2, w):
    iterations = 0
    X = np.zeros((num, 2))
    flag = True
    for i in range(80):
        flag = True
        for j in range(num):
            X[j,0] = X1[j]
            X[j,1] = X2[j]
            if sign(X[j], w) != Y1[j]:
                flag = False
                w = w + Y1[j] * (X1[j], X2[j])
                # print( 'w= ', w,'x= ', X)
                plot_data_and_line(w[0], w[1], i)
                break
            else:
                continue
        if flag == True:
            iterations = i
            break
    # plt.show()
    return w, flag, iterations

filename = r"data/PLA.dat"
num, X1, X2, Y1 = getDataSet(filename)
w0 = np.zeros((2, 1))
w, flag, iterations = PLA_Naive(X1,X2, w0)
# print(flag)
# print(iterations)
# print(w)