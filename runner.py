import math
import matplotlib.pyplot as plt
import numpy as np


fields = ['year', 'jan', 'feb', 'march', 'april', 'may', 'june', 'july','august', 'sept', 'oct', 'nov', 'dec', 'year-middle']


def splitter(fieldnames, file):
    """special function for a specific datat set of mine
    returns list of operable values in list object + errors(also specific) count in data set"""
    middle_year_list = []
    errors = 0
    for line in file:
        elem = line.split('\t')[len(line.split('\t')) - 1]
        elem = float(elem[:len(elem)-2])

        if elem != 999.9:
            middle_year_list.append(elem)
        else:
            errors += 1
    return middle_year_list, errors




def lesser_then_lists_elements(list, pivot):

    #pivot's index should be the same as list's slice end index

    for elem in list:
        if pivot < elem:
            continue
        else:
            return 0
    return 1


def greater_then_lists_elements(list, pivot):

    #pivot's index should be the same as list's slice end index

    for elem in list:
        if pivot > elem:
            continue
        else:
            return 0
    return 1


def leveller(list):
    k = []
    l = []
    for i in range(len(list)):
        if i ==0:
            k.append('start')
            l.append('start')
        else:
            k.append(greater_then_lists_elements(list[:i], list[i]))
            l.append(lesser_then_lists_elements(list[:i],list[i]))
    return k, l


def summ_of_side_values(ks, ls):
    if len(ks) == len(ls):
        s = 0
        for i in range(len(ks)):
            if i == 0:
                continue
            s += ks[i] - ls[i]      #should check values before summing them todo
        return s
    else:
        print('lists length not equal')


def summ_of_list(list):
    s = 0
    for value in list:
        s += value
    return s


def mid_arithmethic(list):
    return summ_of_list(list)/len(list)


def mid_square_desp(list_orig, pivot):      #pivot's value of mid arithmetic value
    semi_list = [(value - pivot) ** 2 for value in list_orig]
    return math.sqrt(mid_arithmethic(semi_list))


def draw_graph(x, y):
    pass
    #todo deligated to borsyakov


def filler(your_list, length):
    """function that fills not calculated values of list"""
    for i in range(int(length/2)):
        your_list.append(None)


def smooth_summ(l):
    s = 0
    main_devider = 143
    weights = [-11, 0, 9, 16, 21, 24, 25, 24, 21, 16, 9, 0, -11]
    if len(l) == len(weights):
        for i in range(len(weights)):
            s += l[i]*weights[i]
    else:
        print('There is not equal length of weights list and slices length. Maybe something wrong with slices definition')
        return
    return s/main_devider


def smooth(list):
    print('here starts smooth\n')
    n = 13 # len of interval
    smoothed_list_value = []
    filler(smoothed_list_value, n)
    for i in range(len(list) - (n-1)):
        smoothed_list_value.append(smooth_summ(list[i:n+i]))
    filler(smoothed_list_value, n)
    return smoothed_list_value

def stepener(mat):
    shape = getattr(mat, 'shape')
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            mat[i][j] **= i+1
    print(mat)


"""    for i in range(shape[1]):
        for row in mat:
            for element in row:
                element *= getattr(element, 'shape')
 """   #if hasattr(li, 'length'):



if __name__ == "__main__":
    csv_path = "Weather.csv"

    with open(csv_path,'r') as f_obj:
        middle_list, err = splitter(fields, f_obj)
    f_obj.close()
    gr, ls = leveller(middle_list)
    d = summ_of_side_values(gr, ls)
    print(d)
    print('errors counter: ' + err.__str__())
    sig = mid_square_desp(middle_list, mid_arithmethic(middle_list))
    t = d/sig
    print('значение t-критерия: '+ t.__str__())
    #draw_graph(len(middle_list), middle_list)
    smoothed_values = smooth(middle_list)
    if len(smoothed_values) == len(middle_list):
        print('smoothed list is relevant')
    print('here starts matrix:')
    matrix = np.ones((3, len(middle_list)))
    for row in matrix:
        for i in range(len(row)):
            row[i] = i+1
    stepener(matrix)
    #place for regress todo



    """for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] *= 2"""

