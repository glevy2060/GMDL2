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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    arr1 = np.array([1, 2])
    arr2 = np.array([1, 2])
    # Temp = 1
    # a = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    # ex3()
    # ex4()
    print(y2row(3, 2))
    # ex5()
    ex6()
    # print(G(arr, Temp))
    # print(G(arr1, Temp))
    # print(np.exp([5]))
    # print(F(arr1, arr2, Temp))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
