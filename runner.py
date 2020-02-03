import math
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from scipy.linalg import *
from scipy.stats import linregress


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


def filler(your_list, length):
    """function that fills not calculated values of list"""
    for i in range(int(length/2)):
        your_list.append('None')


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


def smooth(list, n=13):
    print('here starts smooth\n')
    #n = 13 # len of interval
    smoothed_list_value = []
    filler(smoothed_list_value, n)
    for i in range(len(list) - (n-1)):
        smoothed_list_value.append(smooth_summ(list[i:n+i]))
    filler(smoothed_list_value, n)
    for el in smoothed_list_value: #this part inverses filler with original's list values
        if el == 'None':
            smoothed_list_value[smoothed_list_value.index(el)] = list[smoothed_list_value.index(el)]
    return smoothed_list_value


def stepener(row, power):
    powered_row = [el**power for el in row]
    return powered_row


def checker(mnk_row, orig_row):
    p = 0
    if len(mnk_row) == len(orig_row):
        for i in range(len(mnk_row)):
            if abs(mnk_row[i] - orig_row[i]) < 1:
                p+=1
        q = len(mnk_row) - p
        u = p/(p+q)
        mid = []
        for i in range(len(orig_row)):
            mid.append(abs(mnk_row[i] - orig_row[i]))
        mid = sum(mid)/len(orig_row)
    else:
        print('not equal length of parametrs in checker')
        return
    return {'mid_value':mid,
            'u_value': u}


if __name__ == "__main__":
    csv_path = "Weather.csv"
    with open(csv_path, 'r') as f_obj:
        middle_list, err = splitter(fields, f_obj)
    f_obj.close()
    print('data collected')
    gr, ls = leveller(middle_list)
    d = summ_of_side_values(gr, ls)
    print('value of side summ:' + d.__str__())
    print('errors counter: ' + err.__str__())
    sig = mid_square_desp(middle_list, mid_arithmethic(middle_list))
    t = d/sig
    print('значение t-критерия: '+ t.__str__())
    smoothed_values = smooth(middle_list)
    if len(smoothed_values) == len(middle_list):
        print('smoothed list is relevant')
        #for i in range(len(smoothed_values)):
            #print('The disstance is {}'.format(abs(middle_list[i] - smoothed_values[i])))
    print('here starts matrix:')
    x = [i for i in range(len(middle_list))]
    x2 = stepener(x, 2)
    x3 = stepener(x, 3)
    x4 = stepener(x, 4)
    x5 = stepener(x, 5)
    x6 = stepener(x, 6)
    x7 = stepener(x, 7)
    y = smoothed_values
    m = vstack((x7, x6, x5, x4, x3, x2, x, ones(len(middle_list)))).T #определение вектора-функции, ее вида
    s = lstsq(m, y)[0]# MNK
    x_prec = linspace(0, len(middle_list), 100) # def of the segment
    func = s[0]*x_prec**7 + s[1]*x_prec**6 + s[2]*x_prec**5 + s[3]*x_prec**4 + s[4]*x_prec**3 + s[5]*x_prec**2 + s[6]*x_prec + s[7]
    print('Checker: {}'.format(checker(func, y)))

    x_sin = [math.sin(i) for i in x]
    x_cos = [math.cos(i) for i in x]
    vec = vstack((x_cos, x_sin, x, ones(len(middle_list)))).T
    coef = lstsq(vec, smoothed_values)[0]
    print(len(coef))
    func1 = coef[0]*cos(x_prec) + coef[1]*sin(x_prec) + coef[2]*x_prec + coef[3]
    row = np.ones((1, len(middle_list)))
    for i in range(len(row[0])):
        row[0][i] = i
    A = np.vstack([row[0], np.ones(len(row[0]))]).T
    m, c = np.linalg.lstsq(A, smoothed_values, rcond=None)[0]
    print('m: {}; c: {}'.format(m, c))
    plt.plot(x_prec, func1, 'g', label='trigonometrical fit', lw=2)
    plt.plot(x_prec, func, 'b', label='Least square fit power of: {}'.format(len(s) - 1), lw=2)
    plt.plot(row[0], smoothed_values, 'o', label='Orig data', markersize=10)
    plt.plot(row[0], m*row[0] + c, 'r', label='Fitted line')
    plt.legend()
    plt.show()







    """matrix = np.ones((3, len(middle_list)))
    for row in matrix:
        for i in range(len(row)):
            row[i] = i+1
    stepener(matrix)
    sl, intercept, rval, pval, err = linregress(smoothed_values, matrix[0])
    print('possible coeff\'s: ', sl, 'intercept:', intercept,'r-squared: ', rval**2)
    row1 = np.ones((1, len(middle_list)))
    sl, intercept, rval, pval, err = linregress(smoothed_values, row1)
    print('sl, intercept, r-squared: ', sl, intercept, rval**2)
    print('row:', row1)
    matrix = np.concatenate((matrix, row1))"""
   # print(matrix)




    """for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] *= 2"""

