from matplotlib import pyplot as plt
from swarmlib import FUNCTIONS
from firefly import Firefly
FA = Firefly()
bestvals,best = FA.run(function=FUNCTIONS['rastrigin'], dim=1, lowerbound=-5.12, upperbound=5.12, max_evals=10000, pop_size=100)
lenbestval = len(bestvals)
arr = []
for i in range(0,lenbestval):
    arr.append(i)
plt.plot(bestvals)
plt.title("Function 1")
plt.xlabel("Iterations")
plt.ylabel("Min function(x)")
plt.show()
plt.close()

print(best)