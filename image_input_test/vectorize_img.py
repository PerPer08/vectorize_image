import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

img = cv2.imread(r'C:\projects\python\vectorize_img\img10.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
r, g, b = cv2.split(img)
r = r.flatten()
g = g.flatten()
b = b.flatten()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # add a 3D subplot to the figure
ax.scatter(r, g, b, c='r', marker='.')  # plot the scatter points

ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

plt.show()