#include <iostream>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
// #include <inlucde/supp_two.hpp>

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

    cout << "the expected fundamental matrix should be" << endl;

    cv::Mat im1;
    cv::Mat im2;
    im1 = cv::imread("/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_4/code/data/im1.png", 1);
    im2 = cv::imread("/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_4/code/data/im2.png", 1);

    /*
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", im1);
    cv::waitKey(0);
    */

    Eigen::MatrixXi pts1(10, 2);
    Eigen::MatrixXi pts2(10, 2);

    pts1 << 157, 231,
        309, 284,
        157, 225,
        149, 330,
        196, 316,
        302, 273,
        159, 324,
        158, 137,
        234, 340,
        240, 261;

    pts2 << 157, 211,
            311, 279,
            157, 203,
            149, 334,
            197, 318,
            305, 268,
            160, 327,
            157, 140,
            237, 346,
            240, 258;

    // cout << pts1 << endl;
    // cout << pts2 << endl;

    

    return 0;
}