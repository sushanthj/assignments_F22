import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

A = np.array([[1,0,4],[0,1,0],[0,0,1]])
TEST = np.array([[0,-1,0],[1,0,0],[0,0,1]]) # 90 degree counter clock rotation
B = np.array([[0.866,0.5,0],[-0.5,0.866,0],[0,0,1]])

def main():
    # define the initial points in matlab
    pts = np.array([[-1,1], [0,1], [1,0], [0,-1], [-1,-1]])

    # plot initial shape to check
    plot_shape(pts, 'original_plot')
    
    # a.) A, relative to the fixed frame
    pts_a = transform_mat(pts, A)
    plot_shape(pts_a, 'q2.a')

    # b.) A, relative to the fixed frame, followed by B, relative to the current frame
    #? since it's relative to current frame, we right mulitply
    pts_b = transform_mat(pts, A @ B)
    plot_shape(pts_b, 'q2.b')

    # c.) A, relative to the fixed frame, followed by B, relative to the fixed frame
    #? since it's relative to fixed frame, we left mulitply
    pts_c = transform_mat(pts, B @ A)
    plot_shape(pts_c, 'q2c')

    # d.) B, relative to the fixed frame
    pts_d = transform_mat(pts, B)
    plot_shape(pts_d, 'q2d')

    # e.) B, relative to the fixed frame, followed by A, relative to the fixed frame
    #? since it's relative to fixed frame, we  left mulitply
    pts_e = transform_mat(pts, A @ B)
    plot_shape(pts_e, 'q2e')

    # f.) B, relative to the fixed frame, followed by A, relative to the current
    #? since it's relative to current frame, we right mulitply
    pts_f = transform_mat(pts, B @ A)
    plot_shape(pts_f, 'q2f')
    

# transforms coords according to given transformation matrix
def transform_mat(given_pts, transf_matrix):
    end_row = np.ones((1,5))
    given_pts = np.vstack((given_pts.T, end_row))
    transformed_coords = transf_matrix @ given_pts
    
    return transformed_coords[0:2,:].T


def plot_shape(pts, fig_name='figure_1'):
    p = Polygon(pts, closed=False)
    ax = plt.gca()
    ax.add_patch(p)
    ax.set_xlim(-5,8)
    ax.set_ylim(-5,8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(fig_name)
    plt.show()

if __name__ == '__main__':
    main()