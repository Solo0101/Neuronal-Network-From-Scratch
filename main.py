from src.enviroment.constants import output_layer_neurons_number
from src.framework.neuronal_network import Network
from src.read_data.read import *
import cupy as cp


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



def main():
    print('Test')
    num_neurons = cp.array([15, 15, output_layer_neurons_number])
    n1 = Network(3, num_neurons)
    n1.train(XTrainArr, yTrainArr, 500, 0.1)
    # Press Ctrl+F8 to toggle the breakpoint.
    return

if __name__ == '__main__':
    main()


