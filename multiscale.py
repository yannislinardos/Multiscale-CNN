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
                __rows, __columns = 2 * __coarse_scale+2, __coarse_scale+2
                __matrix = np.zeros([__rows, __columns])

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
        __matrix = np.zeros((__coarse_scale, __fine_scale))
        __col = 0
        for row in range(__coarse_scale):
            __matrix[row][__col] = -1 / 2
            __col += 1
            __matrix[row][__col] = 3 / 2
            __col += 1

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



def get_avg_pooling(signal_size: int) -> np.ndarray:

    __K_p = np.zeros(shape=(int(signal_size/2), signal_size))

    __c = 0
    for __r in range(int(signal_size/2)):
        __K_p[__r][__c] = 1 / 2
        __c += 1
        __K_p[__r][__c] = 1 / 2
        __c += 1

    return __K_p


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


def upscale_kernel(__method: str,  __kernel: np.ndarray, __coarse_scale: int, __fine_scale: int, __zero_padding=True,
                   P=None, R=None) -> np.ndarray:


    if P is None and R is None:
        __P = sp.Matrix(
            get_prolongation(__method, __coarse_scale, __fine_scale, __zero_padding=False, __kernel_size=len(__kernel)))
        __R = sp.Matrix(
            get_restriction(__method, __fine_scale, __coarse_scale, __zero_padding=False, __kernel_size=len(__kernel)))

    else:
        __P, __R = P, R


    # create a kernel with unknown variables
    __num_of_unknowns = len(__kernel)
    __unknowns = symbols('x0:%d' % __num_of_unknowns)
    __K_h = sp.Matrix(utils.get_toeplitz(np.array(sp.Array(__unknowns)), __fine_scale, zero_padding=True))

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


def downscale_avg_pooling(__method, __fine_scale, __coarse_scale):

    if __method == 'nearest_neighbor':
        __P = get_prolongation(__method, __coarse_scale, __fine_scale, __zero_padding=False)
        __R = get_restriction(__method, __fine_scale // 2, __coarse_scale // 2, __zero_padding=False)
        __K_p = get_avg_pooling(__fine_scale)

        return __R@__K_p@__P

    elif __method == 'linear':
        __P = get_prolongation(__method, __coarse_scale, __fine_scale, __zero_padding=False)
        __R = get_restriction(__method, __fine_scale // 2, __coarse_scale // 2, __zero_padding=False)
        __K_p = get_avg_pooling(__fine_scale)

        __m = __R@__K_p@__P
        # delete borders because they do not follow the pattern
        # __m = np.delete(__m, -1, 0)
        # __m = np.delete(__m, 0, 1)

        return __m

    elif __method == 'distance_weighting':

        __P = get_prolongation(__method, __coarse_scale, __fine_scale, __zero_padding=False)
        __R = get_restriction(__method, __fine_scale // 2, __coarse_scale // 2, __zero_padding=False)
        __K_p = get_avg_pooling(__fine_scale)

        return __R @ __K_p @ __P


def upscale_avg_pooling(__method, __coarse_scale, __fine_scale):

    if __method == 'nearest_neighbor':

        __K_P = sp.Matrix(get_avg_pooling(__coarse_scale))

        __P = sp.Matrix(get_prolongation(__method, __coarse_scale, __fine_scale, __zero_padding=False))
        __R = sp.Matrix(get_restriction(__method, __fine_scale//2, __coarse_scale//2, __zero_padding=False))

        __K_p, __unknowns = utils.get_symbol_matrix( __fine_scale//2, __fine_scale)

        __eq = sp.Eq(__K_P, __R*__K_p*__P)
        sp.pprint(__eq)
        __solutions = sp.solve(__eq, __unknowns, particular=True, quick=True)

        sp.pprint(__solutions)

        return utils.get_matrix_from_symbols(__K_p.shape[0], __K_p.shape[1], __solutions)

    elif __method == 'linear':

        __K_P = sp.Matrix(get_avg_pooling(__coarse_scale))

        __P = sp.Matrix(get_prolongation(__method, __coarse_scale, __fine_scale, __zero_padding=False))
        __R = sp.Matrix(get_restriction(__method, __fine_scale//2, __coarse_scale//2, __zero_padding=False))

        __K_p, __unknowns = utils.get_symbol_matrix( __fine_scale//2 , __fine_scale )

        __B = __R * __K_p * __P

        __B.col_del(0)
        __B.row_del(-1)

        __eq = sp.Eq(__K_P, __B)

        __solutions = sp.solve(__eq, __unknowns)

        return utils.get_matrix_from_symbols(__K_P.shape[0], __K_P.shape[1], __solutions)

    elif __method == 'distance_weighting':

        print()


def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sp.Number)})


if __name__ == '__main__':


###################### Get formulae #########################
    method = 'distance_weighting'
    coarse_scale = 18
    fine_scale = 36

    high_kernel = symbols('x1:%d' % 17)
    low_kernel = symbols('y1:%d' % 17)
    K_h = sp.Matrix(utils.get_toeplitz(high_kernel, fine_scale, zero_padding=True))

    P = sp.Matrix(get_prolongation(method, coarse_scale, fine_scale, __zero_padding=False))
    R = sp.Matrix(get_restriction(method, fine_scale, coarse_scale, __zero_padding=False))

    K_H = sp.Matrix(utils.get_toeplitz(low_kernel, coarse_scale, zero_padding=True))
    # K_H = utils.get_symbol_matrix(4,9)[0]
    # unknowns = utils.get_symbol_matrix(5,10)[1]

    B = R * K_h * P

    if method == 'linear':
        K_H.col_del(0)
        K_H.row_del(-1)
        B.col_del(0)
        B.row_del(-1)

    if method == 'distance_weighting':
        K_H.col_del(-1)
        B.col_del(-1)

    M = K_H - B
    equations = []

    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            equation = sp.Eq(M[r, c])

            if equation not in equations and type(equation) == sp.Eq:
                equations.append(equation)

    solution = sp.solve(equations, low_kernel)

    ## coefficient matrix to find scaled up kernel
    coefficient_matrix = sp.linear_eq_to_matrix(equations, high_kernel)[0]


###################################################################



####################### find downscaled avg pooling #########################

    # method = 'linear'
    # coarse_scale = 10
    # fine_scale = 20
    #
    # P = get_prolongation(method, coarse_scale, fine_scale, __zero_padding=False)
    # R = get_restriction(method, fine_scale // 2, coarse_scale // 2, __zero_padding=False)
    #
    # K_p = get_avg_pooling(fine_scale)
    #
    # K_P = R@K_p@P
#################################################################



####################### Upscaling avg pooling####################################3
    # kernel_original = np.array([0.5,0.5], dtype=np.float32)
    # print('kernel original: ', kernel_original)
    #
    # kernel_down = downscale_kernel(method, kernel_original, fine_scale, coarse_scale, __zero_padding=False)
    # print('kernel downscale: ', kernel_down)
    # kernel_up = upscale_kernel(method, kernel_down, coarse_scale, fine_scale, __zero_padding=False)
    # print('kernel upscale: ', kernel_up)
    #
    #
    # kernel = symbols('x1:%d' % 4)
    # # avg_pool = symbols('y1:%d' % 4)
    #
    # # K_P = sp.Matrix(get_avg_pooling(coarse_scale))
    # # K_P = sp.Matrix(utils.strided_toeplitz(avg_pool, coarse_scale))
    #
    # P = sp.Matrix(get_prolongation(method, coarse_scale, fine_scale, __zero_padding=False))
    # R = sp.Matrix(get_restriction(method, fine_scale // 2, coarse_scale // 2, __zero_padding=False))
    #
    # # K_p, unknowns = utils.get_symbol_matrix(fine_scale//2, fine_scale)
    #
    # K_p = sp.Matrix(utils.strided_toeplitz(kernel, fine_scale))
    #
    # B = R * K_p * P
    #
    # K_P.col_del(-1)
    # B.col_del(0)
    #
    # M = K_P - B
    #
    # eq = sp.Eq(K_P, B)
    #
    # solutions = sp.solve(eq, kernel)
    #
    # equations = []
    #
    # for r in range(M.shape[0]-2):
    #     for c in range(M.shape[1]):
    #         equation = sp.Eq(M[r, c])
    #
    #         if equation not in equations and type(equation) == sp.Eq:
    #             equations.append(equation)
    #
    #
    #
    #
    # M.col_del(0)
    # K_P.col_del(0)
    #
    # eq = sp.Eq(K_P, B)
    #
    #
    # equations = []
    #
    # for r in range(M.shape[0]):
    #     for c in range(M.shape[1]):
    #         equation = sp.Eq(M[r, c])
    #
    #         if equation not in equations and type(equation) == sp.Eq:
    #             equations.append(equation)
    #
    #
    # solutions = sp.solve(eq, kernel)