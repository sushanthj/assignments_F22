#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

int main()
{
    MatrixXf K1(3,3);
    MatrixXf K2(3,3);

    K1 << 1.5204e+03, 0.0000e+00, 3.0232e+02,
        0.0000e+00, 1.5259e+03, 2.4687e+02,
        0.0000e+00, 0.0000e+00, 1.0000e+00;

    K2 << 1.5204e+03, 0.0000e+00, 3.0232e+02,
        0.0000e+00, 1.5259e+03, 2.4687e+02,
        0.0000e+00, 0.0000e+00, 1.0000e+00;

    cout << "intrinsic of camera 1 is" << K1 << endl;
    cout << "intrinsic of camera 2 is" << K2 << endl;

    return 0;
}