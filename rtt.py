import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
sns.set_style('white')
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image


def plot_tree(node_locations, parent_indices, plot_limits=None, file_name=None):
    """Plot the RRT with given nodes and parents"""

    # define size from 1.0 down to 0.0

    # unit_sizes = 1 - np.array(range(len(parent_indices))) / len(parent_indices)

    # unit_sizes = 1/((np.array(range(len(parent_indices))) / 100) + 1)
    # unit_sizes -= unit_sizes.min()
    # unit_sizes /= unit_sizes.max()

    unit_sizes = np.zeros(len(parent_indices))

    # line widths
    # linewidths = 0.5 * unit_sizes + 0.1
    linewidths = 5 * unit_sizes + 0.1

    # node sizes
    # node_sizes = 0.5 * unit_sizes + 0.5
    node_sizes = linewidths ** 2

    # define arcs as pairs of start node and end node
    arcs = np.array([[loc, node_locations[par]] for loc, par in zip(node_locations, parent_indices)])

    # create the figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # plot the arcs as separate lines
    ax.add_collection(LineCollection(arcs, linewidths=linewidths, color=(0, 0, 0)))
    # plot the nodes
    ax.scatter(*list(np.array(node_locations).transpose()), s=node_sizes, c=(0, 0, 0))
    # remove ticks, set limits
    ax.set_xticks([])
    ax.set_yticks([])
    if plot_limits is None:
        plot_limits = np.abs(np.array(node_locations).flatten()).max() * 1.05
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

img = Image.open('image.jpg')
img = img.convert('LA')

def choose_sample(iteration, strategy, **params):
    if strategy == 'square':
        return np.random.uniform(params['min'], params['max'], 2)
    elif strategy == 'gauss':
        # nodes should spread out with increasing iteration (size goes from 0 to 1)
        size = 1 - np.exp(-(iteration + 1) / 500)
        # sample new node from normal distribution with standard deviation 'size'
        return np.random.multivariate_normal([0, 0], [[size ** 2, 0], [0, size ** 2]])
    elif strategy == 'image':
        (x, y) = (0, 0)
        for idx in range(100000):  # max number of samples
            x = np.random.randint(0, img.width)
            y = np.random.randint(0, img.height)
            if np.random.uniform() < 1 - img.getpixel((x, img.height - y - 1))[0] / 255:
                break
        point = np.array([x, y], dtype=float) + np.random.uniform(-0.5, 0.5, 2)
        point /= max(img.width, img.height)
        point -= np.array([img.width, img.height]) / max(img.width, img.height) / 2
        return point
    else:
        raise UserWarning("Unknown strategy '{}'".format(strategy))


def find_neighbor(node_locations, target_location):
    # find nearest neighbor
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto')  # algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']
    nbrs.fit(node_locations)
    # return nearest neighbor
    return nbrs.kneighbors([target_location], n_neighbors=1, return_distance=False)[0][0]


def choose_location(target_location, tree_node_location, strategy, **params):
    if strategy == 'simple':
        return target_location
    elif 'maxstep':
        diff = np.array(target_location) - np.array(tree_node_location)
        norm = np.linalg.norm(diff)
        if norm <= params['max']:
            return target_location
        else:
            return np.array(tree_node_location) + diff / norm * params['max']
    else:
        raise UserWarning("Unknown strategy '{}'".format(strategy))

# create RRT
node_locations = [[0, 0]]
parent_indices = [0]
remember = []
for iteration in range(100000):  # this many nodes will be created

    # samples target location
    target_location = choose_sample(iteration=iteration,
                                    strategy='image', min=-1, max=1)

    # parent tree node is nearest neighbor
    tree_node_idx = find_neighbor(node_locations=node_locations,
                                  target_location=target_location)

    new_location = choose_location(target_location=target_location,
                                   tree_node_location=node_locations[tree_node_idx],
                                   strategy='maxstep', max=0.1)

    # store new node and its parent
    node_locations.append(new_location)
    parent_indices.append(tree_node_idx)

    # remember this iteration for plotting
    remember.append((list(node_locations), parent_indices))

    # plot progress in steps of 1000
    if iteration % 1000 == 0:
        print("created {} nodes".format(iteration))
        plot_tree(node_locations=node_locations,
                  parent_indices=parent_indices,
                  plot_limits=np.abs(np.array(node_locations).flatten()).max() * 1.05,
                  file_name='iteration_{}.jpg'.format(str(iteration).zfill(6)))

# # plot growth of RRT
# for idx, (loc, par) in enumerate(remember):
#     if idx % 100 == 0:  # only plot each 10th iteration for speedup
#         print("plot iteration {}".format(idx))
#         plot_tree(nodes=loc,
#                   parents=par,
#                   plot_limits=np.abs(np.array(nodes).flatten()).max() * 1.05,
#                   file_name='iteration_{}.jpg'.format(str(idx).zfill(6)))
