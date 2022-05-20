# This is a sample Python script.
import numpy
import numpy as np
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
            p["p{0}{1}".format(k, k + 1)] = calc_p12(base, temp, T["T1"])
        elif k == base:  # Last p to calc, p8
            p["p{0}".format(k)] = calc_p8(temp, base, T)
        else:
            Tprev = T['T{0}'.format(k - 1)]
            Tk = T['T{0}'.format(k)]
            p["p{0}{1}".format(k, k + 1)] = calc_pkk1(base, Tprev, Tk, temp)
    return p


def ex7():
    T = calc_T(1)
    # p8 = calc_p8(1, 1, 8, T)
    # print(p8)
    # test = calc_p8(1, 8, T)
    # print(test)
    # print(len(test))
    p = calc_p(1, T)
    print(p)
    print(len(p))
    # calc_p(1, T)
    # T1 = T["T1"]
    # test = calc_p12(8, 1, T1)
    # print(test)
    # print(len(test))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # arr1 = np.array([1, 2])
    # arr2 = np.array([1, 2])
    # # Temp = 1
    # # a = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    # # ex3()
    # # ex4()
    # print(y2row(3, 2))
    # # ex5()
    # ex6()
    # calc_T()
    ex7()
    # print(G(arr, Temp))
    # print(G(arr1, Temp))
    # print(np.exp([5]))
    # print(F(arr1, arr2, Temp))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
