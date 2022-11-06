import os
import utils

scores = []
filename = 'sac_average_scores.png'
figure_file = os.path.join('Mario/SAC', filename)

with open("Mario/SAC/gg", 'r') as f:
    lines = f.readlines()
    lines = lines[53:100] + lines[100:200] + lines[150:250] + lines[200:]
    for i in lines:
        scores.append(int(float(i.split('=')[1].split()[0])))

x = [i + 1 for i in range(len(scores))]
utils.plot_learning_curve(x, scores, figure_file)
