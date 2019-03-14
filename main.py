import random

from math import exp
from constants import *

random.seed(0)

# global variables: weight, out and delta arrays
delta = []
f_value = []
w = []


def tanh(x):
    return (exp(x) - exp(-x))/(exp(x) + exp(-x))


def fermi(x):
    return 1.0/(1.0 + exp(-x))


def ident(x):
    return x


def der_from_tanh(f):
    """
    calculate derivative of the tanh function
    using function value in point x itself
    """
    return 1.0 - f * f


def der_from_fermi(f):
    """
    calculate derivative of the fermi function
    using function value in point x itself
    """
    return f*(1.0 - f)


def der_from_ident(f):
    """
    calculate derivative of the ident function
    using function value in point x itself
    """
    return 1.0


# transfer function by name
transfer_fs = {F_TAN: tanh, F_FERMI: fermi, F_IDENT: ident}

# derivative function by name
der_fs = {F_TAN: der_from_tanh, F_FERMI: der_from_fermi, F_IDENT: der_from_ident}


def delta_f(neuron, layer, y_teach):
    """
    calculate delta
    """
    if layer == LAYERS_NUM - 1:
        return delta_out(neuron, layer, y_teach)
    else:
        return delta_hidden(neuron, layer)


def delta_out(neuron, layer, y_teach):
    """
    calculate delta for output layer
    """
    der_func = der_fs[FS_FOR_LAYERS[layer]]
    out = f_value[layer][neuron]
    return (y_teach[neuron] - out) * der_func(out)


def delta_hidden(neuron, layer):
    """
    calculate delta for hidden layer
    """
    der_func = der_fs[FS_FOR_LAYERS[layer]]
    out = f_value[layer][neuron]
    coef = 0.0
    for next_neuron in range(NEUR_ON_LAYER[layer + 1]):
        coef += w[layer + 1][next_neuron][neuron] * delta[layer + 1][next_neuron]
    return coef * der_func(out)


def prepare():
    """
    Prepare data structures
    """
    for i in range(LAYERS_NUM):
        f_value.append([])
        delta.append([])
        w.append([])
        for j in range(NEUR_ON_LAYER[i]):
            f_value[i].append(0.0)
            delta[i].append(0.0)
            if i == 0:
                continue
            w[i].append([])
            for k in range(NEUR_ON_LAYER[i - 1] + 1):  # + 1 for bias
                w[i][j].append(random.uniform(-2.0, 2.0))
        f_value[i].append(1.0)  # for bias weight update


def error(teacher, output):
    """
    Calculate error as sum over quadratic differences
    """
    sum = 0
    for i in range(len(teacher)):
        diff = teacher[i] - output[i]
        sum += diff*diff
    return sum / 2.0


def backprop(teacher):
    """
    Backpropagation step
    """
    for layer in range(LAYERS_NUM - 1, 0, -1):  # Go from the last layer backwards
        cur_rate = LEARN_RATE[layer]
        for neuron in range(0, NEUR_ON_LAYER[layer]):
            cur_delta = delta_f(neuron, layer, teacher)
            delta[layer][neuron] = cur_delta
            for prev_neuron in range(NEUR_ON_LAYER[layer - 1] + 1):  # Update bias weight too
                w_change = cur_rate * cur_delta * f_value[layer - 1][prev_neuron]
                w[layer][neuron][prev_neuron] += w_change


def update_f_values(input):
    """
    Propagate input values towards output neurons
    Update arrays of output values
    """
    # first layer just transmits values as they are
    for neuron in range(NEUR_ON_LAYER[0]):
        f_value[0][neuron] = input[neuron]
    for layer in range(1, LAYERS_NUM):
        for neuron in range(0, NEUR_ON_LAYER[layer]):
            # weight from neuron h to m is stored as w[layer][m][h]
            # this way we can store bias weight more logically in last element of subarray
            net = w[layer][neuron][-1]  # bias
            for prev_neuron in range(NEUR_ON_LAYER[layer - 1]):
                net += f_value[layer - 1][prev_neuron] * w[layer][neuron][prev_neuron]
            f_name = FS_FOR_LAYERS[layer]
            f_net = transfer_fs[f_name](net)
            f_value[layer][neuron] = f_net


if __name__ == "__main__":
    file = open("training.dat", 'r')
    patterns = file.readlines()
    # Try to find information about N abd M
    for pattern in patterns:
        try:
            if "N=" in pattern:
                data = pattern.split()
                new_n = int(data[2][2:])
                new_m = int(data[3][2:])
                NEUR_ON_LAYER[0] = new_n
                NEUR_ON_LAYER[-1]= new_m
                break
        except:
            print("NO N and M values found, using default")

    M = NEUR_ON_LAYER[0]
    N = NEUR_ON_LAYER[-1]
    prepare()
    error_file = open('learning.curve', 'w')
    iter = 0
    for pattern in patterns:
        if pattern.startswith('#'):
            continue
        values = pattern.split()
        values = [float(i) for i in values]
        update_f_values(values[0:M])
        new_teacher = values[M:]
        new_error = error(new_teacher, f_value[LAYERS_NUM - 1])
        str_res = str(iter) + " " + str(new_error) + '\n'
        error_file.write(str_res)
        backprop(new_teacher)
        iter += 1
        if iter > MAX_ITER_NUM:
            break

    file = open("test.dat", 'r')
    patterns = file.readlines()
    for pattern in patterns:
        if pattern.startswith('#'):
            continue
        values = pattern.split()
        values = [float(i) for i in values]
        update_f_values(values[0:M])
        expected_out = values[M:]
        new_error = error(expected_out, f_value[LAYERS_NUM - 1])
        print("Error: " + str(new_error))
