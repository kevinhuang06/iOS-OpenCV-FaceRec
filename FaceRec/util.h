//
//  util.h
//  iOS-OpenCV-FaceRec
//
//  Created by Kevin Huang on 2017/2/9.
//  Copyright © 2017年 Fifteen Jugglers Software. All rights reserved.
//

#ifndef util_h
#define util_h


#include <opencv2/highgui/cap_ios.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace cv;

Mat ImgRotate(const Mat & ucmatImg, Mat &map_matrix, double dDegree);
cv::Point GetPointPosition(Mat map_matrix, cv::Point pt);
void ImgRotate2(const Mat & ucmatImg, Mat &map_matrix, double dDegree, Mat Img_input);
void DrawFacialPoint(Mat img, Mat shape, Scalar color);
Mat ShapeRot(Mat shape, cv::Rect box, double pitch, double yaw);
Mat ShapeRot3D(Mat shape, cv::Rect box, Mat rot_matrix);


#endif
/* util_h */
