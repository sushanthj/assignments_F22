#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

/*
int main()
{
    // define 3x3 matrix -explicit declaration
    Matrix <float, 3, 3> matrixA;
    matrixA.setZero();
    cout << matrixA <<endl;

    // define 3x3 matrix -typedef declaration
    Matrix3f matrixA1;
    matrixA1.setZero();
    cout <<"\n"<<matrixA1<<endl;

    // Dynamic Allocation -explicit declaration
    Matrix <float, Dynamic, Dynamic> matrixB;

    // Dynamic Allocation -typedef declaration
    // 'X' denotes that the memory is to be dynamic
    MatrixXf matrixB1;

    // constructor method to declare matrix
    MatrixXd matrixC(10,10);

    // print any matrix in eigen is just piping to cout
    cout << endl << matrixC << endl;

    // resize any dynamic matrix
    MatrixXd matrixD1;
    matrixD1.resize(3, 3);
    matrixD1.setZero();
    cout << endl << matrixD1 << endl;

    return 0;
}
*/

/*
int main()
{
    // directly init a matrix of zeros
    MatrixXf A;
    A = MatrixXf::Zero(3, 3);
    cout << "\n \n"<< A << endl;

    // directly init a matrix of ones
    MatrixXf B;
    B = MatrixXf::Ones(3, 3);
    cout << "\n \n"<< B << endl;

    // directly init a matrix filled with a constant value
    MatrixXf C;
    C = MatrixXf::Constant(3, 3, 1.2);
    cout << "\n \n"<< C << endl;
}
*/

/*
int main()
{
    MatrixXd V = MatrixXd::Zero(4,4);

    V << 101, 102, 103, 104,
        105, 106, 107, 108,
        109, 110, 111, 112,
        113, 114, 115, 116;

    cout << V << endl;

    MatrixXd Vblock = V.block(0, 0, 2, 2);
    cout << "\n \n" << Vblock << endl;
}
*/

int main()
{
    MatrixXd A1(2, 2);
    MatrixXd B1(2, 2);

    A1 << 1, 2,
        3, 4;
    B1 << 3, 4,
        5, 6;

    // Dot Product
    MatrixXd C1 = A1 * B1;

    // Multiplication by a scalar
    MatrixXd C2 = 2 * A1;

    cout << C1 << endl;
    cout << C2 << endl;

    MatrixXf K1(4,4);
    K1 << 101, 102, 103, 104,
        105, 106, 107, 108,
        109, 110, 111, 112,
        113, 114, 115, 116;

    cout << K1 << endl;
}