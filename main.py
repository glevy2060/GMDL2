# This is a sample Python script.
import numpy
import numpy as np
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def G(row_s: np.array, Temp: int):
    return np.exp((1/Temp) * np.diag(np.einsum('i,j', row_s, np.delete(row_s, 0))))
    # sum = 0;
    # for i in range(len(row_s)-1):
    #     sum += row_s[i] * row_s[i+1]
    # sum /= Temp
    # return np.exp(sum)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    arr = np.array([1, 2])
    Temp = 1
    print(G(arr, Temp))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
