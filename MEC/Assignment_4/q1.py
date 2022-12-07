import numpy as np

def main():
    R1 = np.array([[1,0,0], [0,0,-1],[0,1,0]])
    R2 = np.array([[0.5, 0.866, 0],[-0.866, 0.5, 0],[0,0,1]])

    R = R1 @ R2
    print("R is \n", R)

    # R = np.array([[0.5, 0.866, 0, 0], [0, 0, -1, 5], [-0.866, 0.5, 0, 6],[0,0,0,1]])
    # print("R is \n", R)

    # R_inv = np.linalg.inv(R)
    # print("inverse R is \n", R_inv)

if __name__ == '__main__':
    main()

