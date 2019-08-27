import matplotlib.pyplot as plt
import numpy as np

d = np.load("normal_compression.dat", allow_pickle=True)
b = np.load("normal_bestaccuracy.dat", allow_pickle=True)
c = np.load("reinit_bestaccuracy.dat", allow_pickle=True)



a = np.arange(25)
plt.plot(a, b, c="blue", label="winning tickets") 
plt.plot(a, c, c="red", label="random reinit") 
plt.title("Test Accuracy vs Pruning Rate (mnist)") 
plt.xlabel("Pruning rate") 
plt.ylabel("Test accuracy") 
plt.xticks(a, d, rotation ="vertical") 
plt.ylim(90,100)
plt.legend() 
plt.grid(color="gray") 

#NOTE Adjust Image Quality Here
plt.savefig("combined_figs/combined_fig1.png", dpi=1200) 
plt.close()