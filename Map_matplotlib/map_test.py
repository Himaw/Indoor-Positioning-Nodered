import matplotlib.pyplot as plt
import pandas as pd
img = plt.imread("room12.jpg")
fig, ax = plt.subplots()
ax.imshow(img, extent=[0, 835, 0, 665])
# ax.scatter(TMIN, PRCP, color="#ebb734")
plt.show()