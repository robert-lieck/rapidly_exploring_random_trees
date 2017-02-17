import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
sns.set_style('white')
import numpy as np
from sklearn.neighbors import NearestNeighbors


def plot_tree(locations, parents, plot_limits=None, file_name=None):
    """Plot the RRT with given node locations and parents"""

    # define size from 1.0 down to 0.0
    unit_sizes = 1 - np.array(range(len(parents))) / len(parents)

    # define arcs as pairs of start location and end location
    arcs = np.array([[loc, locations[par]] for loc, par in zip(locations, parents)])

    # create the figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # plot the arcs as separate lines
    ax.add_collection(LineCollection(arcs, linewidths=0.5 * unit_sizes + 0.1, color='blue'))
    # plot the nodes
    ax.scatter(*list(np.array(locations).transpose()), s=0.5 * unit_sizes + 0.5, c=(0, 0, 0))
    # remove ticks, set limits
    ax.set_xticks([])
    ax.set_yticks([])
    if plot_limits is None:
        plot_limits = np.abs(np.array(locations).flatten()).max() * 1.05
    ax.set_xlim(-plot_limits, plot_limits)
    ax.set_ylim(-plot_limits, plot_limits)
    plt.tight_layout()
    # save figure or show
    if file_name is not None:
        fig.savefig(file_name, dpi=300)
        plt.close()
        fig.clear()
    else:
        plt.show()

# create RRT
node_locations = [[0, 0]]
parents = [0]
remember = []
for iteration in range(10000):  # this many nodes will be created
    # nodes should spread out with increasing iteration (size goes from 0 to 1)
    size = 1 - np.exp(-(iteration + 1) / 500)

    # sample new location from normal distribution with standard deviation 'size'
    new_node_location = np.random.multivariate_normal([0, 0], [[size ** 2, 0], [0, size ** 2]])

    # find nearest neighbor
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto')  # algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']
    nbrs.fit(node_locations)

    # parent of new node is its nearest neighbor
    new_parent = nbrs.kneighbors([new_node_location], n_neighbors=1, return_distance=False)[0][0]

    # store new node and its parent
    node_locations.append(new_node_location)
    parents.append(new_parent)

    # remember this iteration for plotting
    remember.append((list(node_locations), parents))

    # plot progress in steps of 1000
    if iteration % 1000 == 0:
        print("created {} nodes".format(iteration))

# plot growth of RRT
for idx, (loc, par) in enumerate(remember):
    if idx % 10 == 0:  # only plot each 10th iteration for speedup
        print("plot iteration {}".format(idx))
        plot_tree(locations=loc,
                  parents=par,
                  plot_limits=np.abs(np.array(node_locations).flatten()).max() * 1.05,
                  file_name='iteration_{}.jpg'.format(str(idx).zfill(6)))
