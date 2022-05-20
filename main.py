# This is a sample Python script.
import numpy
import numpy as np
import matplotlib.pyplot as plt

import itertools


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def G(row_s: np.array, Temp: int):
    return np.exp((1 / Temp) * np.sum(np.diag(np.einsum('i,j', row_s, np.delete(row_s, 0)))))


def F(row_s, row_t, Temp):
    return np.exp((1 / Temp) * np.sum(np.diag(np.einsum('i,j', row_s, row_t))))


def calc_Ztemp2(Temp):
    sum = 0
    tuple = np.array([-1, 1])
    for i in tuple:
        for j in tuple:
            for k in tuple:
                for l in tuple:
                    sum += np.exp((i * j + k * l + i * k + j * l) / Temp)
    return sum


def ex3():
    temps = np.array([1, 1.5, 2])
    for temp in temps:
        print(calc_Ztemp2(temp))


def calc_Ztemp3(Temp):
    sum = 0
    tuple = np.array([-1, 1])
    for a in tuple:
        for b in tuple:
            for c in tuple:
                for d in tuple:
                    for e in tuple:
                        for f in tuple:
                            for g in tuple:
                                for h in tuple:
                                    for i in tuple:
                                        sum += np.exp((
                                                              a * b + a * d + b * c + b * e + c * f + e * f + d * g + d * e + e * h + g * h + f * i + h * i) / Temp)
    return sum


def ex4():
    temps = np.array([1, 1.5, 2])
    for temp in temps:
        print(calc_Ztemp3(temp))


def y2row(y, width=8):
    """ y: an integer in (0,...,(2**width)-1) """
    if not 0 <= y <= (2 ** width) - 1:
        raise ValueError(y)
    my_str = np.binary_repr(y,width=width)
    my_list = list(map(int,my_str)) # Python 3
    my_array = np.asarray(my_list)
    my_array[my_array==0]=-1
    row=my_array
    return row


def calc_Ztemp_ex5(Temp):
    sum = 0
    for i in range(0, 4):
        for j in range(0, 4):
            y1 = y2row(i, 2)
            y2 = y2row(j, 2)
            sum += G(y1, Temp) * G(y2, Temp) * F(y1, y2, Temp)
    return sum


def ex5():
    temps = np.array([1, 1.5, 2])
    for temp in temps:
        Ztemp = calc_Ztemp_ex5(temp)
        for i in range(0,4):
            for j in range(0,4):
                y1 = y2row(i, 2)
                y2 = y2row(j, 2)
                py = (1 / Ztemp) * G(y1, temp) * G(y2, temp) * F(y1, y2, temp)
                print(f"temp: {temp} i: {i}, j: {j} py: {py}")


def calc_Ztemp_ex6(Temp):
    sum = 0
    for i in range(0, 8):
        for j in range(0, 8):
            for k in range(0, 8):
                y1 = y2row(i, 3)
                y2 = y2row(j, 3)
                y3 = y2row(k, 3)
                sum += G(y1, Temp) * G(y2, Temp) * G(y3, Temp) * F(y1, y2, Temp) * F(y2, y3, Temp)
    return sum


def ex6():
    temps = np.array([1, 1.5, 2])
    for temp in temps:
        Ztemp = calc_Ztemp_ex6(temp)
        if temp == 1:
            print(Ztemp)
        for i in range(0,8):
            for j in range(0,8):
                for k in range(0,8):
                    y1 = y2row(i, 3)
                    y2 = y2row(j, 3)
                    y3 = y2row(k, 3)
                    py = (1 / Ztemp) * G(y1, temp) * G(y2, temp) * G(y3, temp) * F(y1, y2, temp) * F(y2, y3, temp)
                    print(f"temp: {temp} i: {i}, j: {j} k: {k} py: {py}")


def inner_multiply(base, y1, y2, temp):
    y1tag = y2row(y1, base)
    y2tag = y2row(y2, base)
    return G(y1tag, temp) * F(y1tag, y2tag, temp)


def calc_T1(base, temp):
    upper_bound = np.power(2,base)
    T1 = np.zeros(upper_bound)  # 2^2, 2^3...
    for y2 in range(upper_bound):
        entry_sum = 0
        for y1 in range(upper_bound):
            entry_sum += inner_multiply(base, y1, y2, temp)
        T1[y2] = entry_sum

    return T1


def calc_Tk(base, Tprev, temp):
    upper_bound = np.power(2, base)
    Tk = np.zeros(upper_bound)  # 2^2, 2^3...
    for yk1 in range(upper_bound):  # yk+1
        entry_sum = 0
        for yk in range(upper_bound):
            yk_tag = y2row(yk, base)
            yk1_tag = y2row(yk1, base)
            entry_sum += Tprev[yk] * G(yk_tag, temp) * F(yk_tag, yk1_tag, temp)
        Tk[yk1] = entry_sum
    return Tk


def calc_Tn(base, Tprev, temp):
    upper_bound = np.power(2,base)
    sum = 0
    for yn in range(upper_bound):
        yntag = y2row(yn, base)
        sum += Tprev[yn] * G(yntag, temp)
    return sum


def calc_T(temp):
    base = 8
    T = {}
    for k in range(1, base+1):
        if k == 1:
            T["T{0}".format(k)] = calc_T1(base, temp)
        elif k == base:
            Tprev = T['T{0}'.format(k-1)]
            T["T{0}".format(k)] = calc_Tn(base, Tprev, temp)
        else:
            Tprev = T['T{0}'.format(k-1)]
            T["T{0}".format(k)] = calc_Tk(base, Tprev, temp)
    return T


def calc_p12(base, temp, T1):
    upper_bound = np.power(2, base)
    p12 = np.zeros((upper_bound, upper_bound))
    for y1 in range(upper_bound):
        for y2 in range(upper_bound):
            y1tag = y2row(y1, base)
            y2tag = y2row(y2, base)
            p12[y1][y2] = G(y1tag, temp) * F(y1tag, y2tag, temp) / T1[y2]
    return p12


def calc_pkk1(base, Tprev, Tk, temp):
    upper_bound = np.power(2, base)
    pkk1 = np.zeros((upper_bound, upper_bound))
    for yk in range(upper_bound):
        for yk1 in range(upper_bound): # yk1 = Yk+1
            yktag = y2row(yk, base)
            yk1tag = y2row(yk1, base)
            pkk1[yk][yk1] = Tprev[yk] * G(yktag, temp) * F(yktag, yk1tag, temp) / Tk[yk1]
    return pkk1


def calc_p8(temp, base, T):
    upper_bound = np.power(2, base)
    T7 = T["T7"]
    T8 = T["T8"]
    p8 = np.zeros(upper_bound)
    for y8 in range(upper_bound):
        y8tag = y2row(y8, base)
        p8[y8] = T7[y8] * G(y8tag, temp) / T8
    return p8


def calc_p(temp, T):
    base = 8
    p = {}
    for k in range(1, base+1):
        if k == 1:
            p["p{0}|{1}".format(k, k + 1)] = calc_p12(base, temp, T["T1"])
        elif k == base:  # Last p to calc, p8
            p["p{0}".format(k)] = calc_p8(temp, base, T)
        else:
            Tprev = T['T{0}'.format(k - 1)]
            Tk = T['T{0}'.format(k)]
            p["p{0}|{1}".format(k, k + 1)] = calc_pkk1(base, Tprev, Tk, temp)
    return p


def create_sample(p):
    sample = np.zeros((8, 8))
    rands = np.zeros(8)

    for j in range(8):
        rands[j] = numpy.random.choice(numpy.arange(256), p=p["p8"])
    rands = rands.astype(int)

    # ------------build the sample------------
    rand8 = rands[7]
    # we pick the column that corresponds to that y8 we drew, and use it to sample y7|8. y7 => rows y8 => columns
    # each column of pi|j is a distribution
    rand7 = numpy.random.choice(numpy.arange(256), p=p["p7|8"][:, rand8])
    rand6 = numpy.random.choice(numpy.arange(256), p=p["p6|7"][:, rand7])
    rand5 = numpy.random.choice(numpy.arange(256), p=p["p5|6"][:, rand6])
    rand4 = numpy.random.choice(numpy.arange(256), p=p["p4|5"][:, rand5])
    rand3 = numpy.random.choice(numpy.arange(256), p=p["p3|4"][:, rand4])
    rand2 = numpy.random.choice(numpy.arange(256), p=p["p2|3"][:, rand3])
    rand1 = numpy.random.choice(numpy.arange(256), p=p["p1|2"][:, rand2])

    # create the sample rows in {-1, 1} from the random numbers
    sample[7] = y2row(int(rand8))
    sample[6] = y2row(int(rand7))
    sample[5] = y2row(int(rand6))
    sample[4] = y2row(int(rand5))
    sample[3] = y2row(int(rand4))
    sample[2] = y2row(int(rand3))
    sample[1] = y2row(int(rand2))
    sample[0] = y2row(int(rand1))

    return sample #sample of size 8x8


def ex7():
    Temps = np.asarray([1, 1.5, 2])
    fig, axs = plt.subplots(3, 10)
    axs[0][0].set_title('Temp = 1')
    axs[1][0].set_title('Temp = 1.5')
    axs[2][0].set_title('Temp = 2')
    for i in range(len(Temps)):
        T = calc_T(Temps[i])
        p = calc_p(Temps[i], T)
        for j in range(10):
            sample = create_sample(p)
            axs[i][j].imshow(sample, interpolation='None')
            axs[i][j].axis('off')
    plt.show()


# Press the green button in the gutter to run the script.

def ex8():
    X = np.zeros((10000, 8, 8))
    Temps = np.asarray([1, 1.5, 2])
    for temp in Temps:
        T = calc_T(temp)
        p = calc_p(temp, T)
        for i in range(10000):
            X[i] = create_sample(p)

        Etemp12 = 0
        Etemp18 = 0
        for n in range(1000):
            Etemp12 += X[n][0][0] * X[n][1][1]
            Etemp18 += X[n][0][0] * X[n][7][7]
        Etemp12 /= 1000
        Etemp18 /= 1000

        print(f"----------Temp {temp}----------")
        print(f"Etemp(X11, X22): {Etemp12}")
        print(f"Etemp(X11, X88): {Etemp18}")


if __name__ == '__main__':
    # ex3()
    # ex4()
    # ex5()
    # ex6()
    # ex7()
    ex8()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
