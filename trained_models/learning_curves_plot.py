import matplotlib 
matplotlib.use("agg")
import matplotlib.pyplot as plt

import numpy as np
import torch

import sys

#User should call this function with the name of direcatory and  the name of the output file as argument.
#Example : python learning_curves_print.py "8dim_lr01/" "plot1.pdf" 
def main():
    directory=sys.argv[1:][0]
    train=torch.load(directory+"train_history.pt")
    val=torch.load(directory+"validation_history.pt")

    plt.plot(train,label="Training Loss")
    plt.plot(val,label="Validation Loss")

    plt.legend()
    plt.title("Learning Curves")
    plt.ylabel("RMSE")

    plt.ylim((0,1.2))

    plt.savefig(directory+sys.argv[1:][1])

if __name__=="__main__":
    main()
