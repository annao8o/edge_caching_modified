import matplotlib.pyplot as plt

x_range = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

for i in range(len(x_range)):
    x_range[i] = x_range[i] / 2000

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xticks(x_range)
#non-clustered
algo_0 = [0.        , 0.06824926, 0.14836795, 0.29673591, 0.33827893,
       0.39169139, 0.48367953, 0.54896142, 0.59050445, 0.72106825,
       1.   ]

#clustered / LCU_num = 1
algo_1 = [0.        , 0.06824926, 0.17507418, 0.30267062,
       0.38872404, 0.51335312, 0.6231454 , 0.67655786, 0.75667656,
       0.83976261, 1.       ]

#clustered / LCU_num = 2
algo_2 = [0.        , 0.06824926, 0.13946588,
       0.26409496, 0.34421365, 0.46884273, 0.56083086, 0.70623145,
       0.8041543 , 0.884273  , 1.        ]

ax.grid(True)
plt.xlabel("Normalized cahce size")
plt.ylabel("Hit ratio")
ax.plot(x_range, algo_0, 'rs--', label='UC')
ax.plot(x_range, algo_1, 'b*-', label='C/num_LCU=1')
ax.plot(x_range, algo_2, 'gv:', label='C/num_LCU=2')

ax.legend(loc=2)
plt.show()