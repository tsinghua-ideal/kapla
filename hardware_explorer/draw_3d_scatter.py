import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# x = np.random.standard_normal(100)
# y = np.random.standard_normal(100)
# z = np.random.standard_normal(100)
# c = np.random.standard_normal(100)

# img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
# fig.colorbar(img)

# plt.savefig('temp.pdf', format='pdf')


if __name__ == '__main__':
    explr_log_dir = 'hardware_explorer'
    pickle_dir = 'design_explr_pickle'
    pickle_name = 'alex_net_bp'
    pickle_path = os.path.join(explr_log_dir, pickle_dir, pickle_name)
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = fig.add_subplot(projection='3d')
    # for result in results:
        # print(result[0], result[1], result[2], result[3])
        # img = ax.scatter(result[0], result[1], result[2], c=-result[3], cmap=plt.hot())
    # ax.set_xlim3d(0, 35)
    # ax.set_ylim3d(0, 70000)
    # ax.set_zlim3d(0, 35)

    print(results)
    img = ax.scatter(results[:, 0], results[:, 1], results[:, 2], c=results[:, 3], cmap=plt.hot())
    fig.colorbar(img)
    plt.savefig(os.path.join(explr_log_dir, f'{pickle_name}.pdf'), format='pdf')