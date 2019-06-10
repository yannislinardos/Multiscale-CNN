import numpy as np
import utils
import sympy as sp
from sympy import symbols
from scipy import linalg
from tqdm import tqdm


def get_prolongation(__method: str, __coarse_scale: int, __fine_scale: int, __zero_padding: bool, __kernel_size=0) -> np.ndarray:

    __rows, __columns = __fine_scale, __coarse_scale
    __matrix = np.zeros([__rows, __columns])
    __ratio = int(__fine_scale / __coarse_scale)

    if __method == 'nearest_neighbor':

        # in every column, put 1 in the cells [2xcolumn][column], [2xcolumn+1][column] ... [2xcolumn + ratio-1][column]
        for __column in range(__columns):
            for __i in range(__ratio):
                __matrix[2*__column+__i][__column] = 1

    elif __method == 'distance_weighting':


            ### if upscale for twice the rate
            ### if more than twice, call recursively
            ### only works for ratio power of two
            if __ratio == 2:
                # interpolate between two samples their average
                # if it is in even position leave it the same, if in odd interpolate with the previous
                __rows, __columns = 2 * __coarse_scale, __coarse_scale

                __matrix[0][0] = 5 / 6
                __matrix[0][1] = 1 / 6

                __matrix[__rows - 1][__columns - 2] = 1 / 6
                __matrix [__rows - 1][__columns - 1] = 5 / 6

                __counter = 1
                for __row in range(1, __rows - 1):

                        if __row % 2 == 0:
                            __matrix[__row][__counter - 1] = 1 / 4
                            __matrix[__row][__counter] = 3 / 4
                            __counter += 1

                        else:
                            __matrix[__row][__counter - 1] = 3 / 4
                            __matrix[__row][__counter] = 1 / 4
            elif __ratio != 2:

                __matrix = np.matmul(get_prolongation('distance_weighting', 2 * __coarse_scale, __fine_scale, False),
                                     get_prolongation('distance_weighting', __coarse_scale, 2 * __coarse_scale, False))

            __matrix = np.delete(__matrix, 0, 0)
            __matrix = np.delete(__matrix, -1, 0)
            __matrix = np.delete(__matrix, 0, 1)
            __matrix = np.delete(__matrix, -1, 1)

    elif __method == 'linear':

        # if upscale for twice the rate
        # if more than twice, call recursively
        # only works for ratio power of two
        if __ratio == 2:
            # interpolate between two samples their average
            # if it is in even position leave it the same, if in odd interpolate with the previous
            __rows, __columns = 2 * __coarse_scale, __coarse_scale
            __column = 0
            for __row in range(__rows):

                # if even leave it the same
                if __row % 2 == 0:
                    __matrix[__row][__column] = 1

                # if odd, take half
                elif __row % 2 == 1:
                    __matrix[__row][__column] = 1 / 2

                    # to deal with the last one
                    if __column < __columns-1:
                        __matrix[__row][__column + 1] = 1 / 2
                    __column += 1

        elif __ratio != 2:

            __matrix = np.matmul(get_prolongation(
                'linear', 2 * __coarse_scale, __fine_scale, False), get_prolongation('linear', __coarse_scale, 2 * __coarse_scale, False))

    # elif __method == 'fourier':
    #
    #     __dft = linalg.dft(__coarse_scale)
    #     __idft = np.linalg.inv(linalg.dft(__fine_scale))
    #     __m = np.zeros(shape=(__rows, __columns))
    #
    #     __row = 0
    #     for __column in range(int(__columns / 2)):
    #         __m[__row][__column] = 1
    #         __m[-(__row + 1)][-(__column + 1)] = 1
    #         __row += 1
    #
    #     if __columns % 2 == 1:
    #         __m[__row][int(__columns / 2)] = 1
    #     print(__m)
    #     __matrix = np.matmul(__idft, np.matmul(__m, __dft)).real *__coarse_scale / __fine_scale


    # ZERO PADDING
    if __zero_padding is True:
        __zero_rows = int(__kernel_size / 2)  #how many zero rows before and after
        __zeros = np.zeros([__zero_rows, __columns]) # zero rows of appropriate size
        __matrix = np.vstack((__zeros, __matrix, __zeros)) # add zeros before and after

    return __matrix


def get_restriction(__method: str, __fine_scale: int, __coarse_scale: int, __zero_padding: bool, __kernel_size=0) -> np.ndarray:

    __rows, __columns = __coarse_scale, __fine_scale
    __matrix = np.zeros([__rows, __columns])
    __ratio = int(__fine_scale / __coarse_scale)

    if __method == 'nearest_neighbor':

        # in every column, put 1 in the cells [i*ratio][column], where ratio is integer
        __i = 0
        for __row in range(__rows):
            for __j in range(__ratio):
                __matrix[__row][__i + __j] = 1 / 2
            __i += __ratio


    elif __method == 'linear':

        ## MY METHOD ################
        # in every column, put 1 in the cells [i*ratio][column], where ratio is integer
        # same as before
        __column = 0
        for __row in range(__rows):
            __column += 1
            __matrix[__row][__column] = 2
            if __column == __columns - 1:
                break
            __column += 1
            __matrix[__row][__column] = -1

    elif __method == 'distance_weighting':
        R = np.zeros((coarse_scale - 2, fine_scale - 2))
        col = 0
        for row in range(coarse_scale - 2):
            R[row][col] = -1 / 2
            col += 1
            R[row][col] = 3 / 2
            col += 1

    # elif __method == 'fourier':
    #
    #     __dft = linalg.dft(__fine_scale)
    #     __idft = np.linalg.inv(np.conj(linalg.dft(__coarse_scale)))
    #     __m = np.zeros(shape=(__rows, __columns))
    #
    #     __column = 0
    #     for __row in range(int(__rows / 2)):
    #         __m[__row][__column] = 1
    #         __m[-(__row + 1)][-(__column + 1)] = 1
    #         __column += 1
    #
    #     if __rows % 2 == 1:
    #         __m[__row][int(__columns / 2)] = 1
    #     print(__m)
    #     __matrix = np.matmul(__idft, np.matmul(__m, __dft)).real * __fine_scale /__coarse_scale

    # remove the zero padding, add zero columns left and right
    if __zero_padding is True:
        __zero_rows = int(__kernel_size / 2)  #how many zero rows before and after
        __zeros = np.zeros([__zero_rows, __columns]) # zero rows of appropriate size
        __matrix = np.vstack((__zeros, __matrix, __zeros)) # add zeros before and after

    return __matrix


def downscale_kernel(__method: str,  __kernel: np.ndarray, __fine_scale: int, __coarse_scale: int, __zero_padding=True) -> np.ndarray:

    __P = get_prolongation(__method, __coarse_scale, __fine_scale, __zero_padding=False, __kernel_size=len(__kernel))
    __R = get_restriction(__method, __fine_scale, __coarse_scale, __zero_padding=False, __kernel_size=len(__kernel))
    __K_h = utils.get_toeplitz(__kernel, __fine_scale, zero_padding=__zero_padding)

    __K_H_trans = np.matmul(__R, np.matmul(__K_h, __P))

    # # remove zero padding ##############
    # if __padding is True:
    #     __zero_rows = int(len(__kernel) / 2)
    #     __K_H_trans = np.delete(__K_H_trans, slice(__zero_rows), 0)

    __K_H = np.transpose(__K_H_trans)
    __new_kernel = utils.get_kernel_from_toeplitz(__K_H, len(__kernel), __remove_zero_padding=True)

    return __new_kernel


def upscale_kernel(__method: str,  __kernel: np.ndarray, __coarse_scale: int, __fine_scale: int, __zero_padding=True) -> np.ndarray:

    # create a kernel with unknown variables
    __num_of_unknowns = len(__kernel)
    __unknowns = symbols('x0:%d' % __num_of_unknowns)
    __K_h = sp.Matrix(utils.get_toeplitz(np.array(sp.Array(__unknowns)), __fine_scale, zero_padding=True))

    __P = sp.Matrix(get_prolongation(__method, __coarse_scale, __fine_scale, __zero_padding=False, __kernel_size=len(__kernel)))
    __R = sp.Matrix(get_restriction(__method, __fine_scale, __coarse_scale, __zero_padding=False, __kernel_size=len(__kernel)))
    __K_H = sp.Matrix(np.transpose(utils.get_toeplitz(__kernel, __coarse_scale, zero_padding=__zero_padding)))

    __B = __R * __K_h * __P

    if __method == 'linear':
        __K_H.col_del(0)
        __K_H.row_del(-1)
        __B.col_del(0)
        __B.row_del(-1)




    sp.pprint(round_expr(__K_H, 3))
    sp.pprint(round_expr(__B,3))
######################################
    # __M = __K_H - __B
    # __equations = []
    #
    # for __r in range(__M.shape[0]):
    #     for __c in range(__M.shape[1]):
    #         __equation = sp.Eq(__M[__r, __c])
    #
    #         if __equation not in __equations and type(__equation) == sp.Eq:
    #             __equations.append(__equation)

###############################################
    # define the matrix equation
    __eq = sp.Eq(__K_H, __B)

    __solution = sp.solve(__eq, __unknowns)

    __new_kernel = []

    try:
        # if there is a solution
        if __solution != []:
            i = 0
            for x in __unknowns:

                __new_kernel.append(np.float32(__solution[x]))
                i += 1
            return np.array(__new_kernel)
        else:
            print("We cannot upscale this kernel")

        __new_kernel = __solution
        return __new_kernel
    except KeyError:
        print('Not unique solution')
        return __solution
    except TypeError:
        print('Not unique solution')
        return __solution


# take an array that contains all the kernels of a layer, the output of utils.get_kernels
# and up- or down-scales it with the given method
def multiscale_layer(__up_or_down: str, __method: str, __layer: np.ndarray, __coarse_scale: int, __fine_scale: int) -> np.ndarray:

    # upscaling
    if __up_or_down == 'up':

        __new_layer = np.empty(shape=__layer.shape)
        for i in tqdm(range(len(__layer))):
            __new_layer[i] = upscale_kernel(__method, __layer[i] ,__coarse_scale, __fine_scale)

        return __new_layer

    elif __up_or_down == 'down':

        __new_layer = np.empty(shape=__layer.shape)

        __kernel_size = len(__layer[0])
        __P = get_prolongation(__method, __coarse_scale, __fine_scale, __zero_padding=True,
                               __kernel_size=__kernel_size)
        __R = get_restriction(__method, __fine_scale, __coarse_scale, __zero_padding=True,
                              __kernel_size=__kernel_size)

        for i in tqdm(range((len(__layer)))):

            #__new_layer[i] = downscale_kernel(__method, __layer[i], __coarse_scale, __fine_scale)
            __K_h = utils.get_toeplitz(__layer[i], __fine_scale, zero_padding=True)
            __K_H = np.transpose(np.matmul(__R, np.matmul(__K_h, __P)))
            __new_layer[i] = utils.get_kernel_from_toeplitz(__K_H, __kernel_size)

        return __new_layer


def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sp.Number)})



if __name__ == '__main__':

    method = 'distance_weighting'
    coarse_scale = 8
    fine_scale = 16
    # P = get_prolongation(method, coarse_scale, fine_scale, __zero_padding=False)
    # R = get_restriction(method, fine_scale, coarse_scale, __zero_padding=False)


    high_kernel = symbols('x1:%d' % 6)
    low_kernel = symbols('y1:%d' % 6)
    K_h = sp.Matrix(utils.get_toeplitz(np.array(sp.Array(high_kernel)), fine_scale, zero_padding=True))

    P = sp.Matrix(get_prolongation(method, coarse_scale, fine_scale, __zero_padding=False, __kernel_size=len(low_kernel)))
    R = sp.Matrix(get_restriction(method, fine_scale, coarse_scale, __zero_padding=False, __kernel_size=len(high_kernel)))

    K_H = sp.Matrix(np.transpose(utils.get_toeplitz(low_kernel, coarse_scale, zero_padding=True)))

    B = R * K_h * P

    M = K_H - B
    equations = []

    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            equation = sp.Eq(M[r, c])

            if equation not in equations and type(equation) == sp.Eq:
                equations.append(equation)

    solution = sp.solve(equations, low_kernel)

    coefficient_matrix = sp.linear_eq_to_matrix(equations, high_kernel)[0]


    # kernel_original = np.array([0.5,0.5], dtype=np.float32)
    # print('kernel original: ', kernel_original)
    #
    # kernel_down = downscale_kernel(method, kernel_original, fine_scale, coarse_scale, __zero_padding=False)
    # print('kernel downscale: ', kernel_down)
    # kernel_up = upscale_kernel(method, kernel_down, coarse_scale, fine_scale, __zero_padding=False)
    # print('kernel upscale: ', kernel_up)
