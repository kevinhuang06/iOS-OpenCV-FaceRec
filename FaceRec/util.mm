//
//  util.m
//  iOS-OpenCV-FaceRec
//
//  Created by Kevin Huang on 2017/2/9.
//  Copyright © 2017年 Fifteen Jugglers Software. All rights reserved.
//


#import <Foundation/Foundation.h>
#import "util.h"



Mat ImgRotate(const Mat & ucmatImg, Mat &map_matrix, double dDegree)
{
    Mat ucImgRotate;
    
    double a = sin(dDegree  * CV_PI / 180);
    double b = cos(dDegree  * CV_PI / 180);
    int width = ucmatImg.cols;
    int height = ucmatImg.rows;
    int width_rotate = int(height * fabs(a) + width * fabs(b));
    int height_rotate = int(width * fabs(a) + height * fabs(b));
    
    cv::Point center = cv::Point(ucmatImg.cols / 2, ucmatImg.rows / 2);
    
    map_matrix = getRotationMatrix2D(center, dDegree, 1.0);
    map_matrix.at<double>(0, 2) += (width_rotate - width) / 2;     // –ﬁ∏ƒ◊¯±Í∆´“∆
    map_matrix.at<double>(1, 2) += (height_rotate - height) / 2;   // –ﬁ∏ƒ◊¯±Í∆´“∆
    
    warpAffine(ucmatImg, ucImgRotate, map_matrix, { width_rotate, height_rotate },
               CV_INTER_CUBIC | CV_WARP_FILL_OUTLIERS, BORDER_CONSTANT, cvScalarAll(0));
    
    return ucImgRotate;
}

cv::Point GetPointPosition(Mat map_matrix, cv::Point pt)
{
    cv::Point result;
    result.x = map_matrix.at<double>(0, 0) * pt.x + map_matrix.at<double>(0, 1) * pt.y + map_matrix.at<double>(0, 2);
    result.y = map_matrix.at<double>(1, 0) * pt.x + map_matrix.at<double>(1, 1) * pt.y + map_matrix.at<double>(1, 2);
    
    return result;
}

void ImgRotate2(const Mat & ucmatImg, Mat &map_matrix, double dDegree, Mat Img_input)
{
    //Mat ucImgRotate;
    double a = sin(dDegree  * CV_PI / 180);
    double b = cos(dDegree  * CV_PI / 180);
    int width = ucmatImg.cols;
    int height = ucmatImg.rows;
    //int width_rotate = int(height * fabs(a) + width * fabs(b));
    //int height_rotate = int(width * fabs(a) + height * fabs(b));
    int width_rotate = Img_input.cols;
    int height_rotate = Img_input.rows;
    
    cv::Point center = cv::Point(ucmatImg.cols / 2, ucmatImg.rows / 2);
    
    map_matrix = getRotationMatrix2D(center, dDegree, 1.0);
    map_matrix.at<double>(0, 2) += (width_rotate - width) / 2;     // –ﬁ∏ƒ◊¯±Í∆´“∆
    map_matrix.at<double>(1, 2) += (height_rotate - height) / 2;   // –ﬁ∏ƒ◊¯±Í∆´“∆
    /*
     warpAffine(ucmatImg, ucImgRotate, map_matrix, { width_rotate, height_rotate },
     CV_INTER_CUBIC | CV_WARP_FILL_OUTLIERS, BORDER_CONSTANT, cvScalarAll(0));
     */
    return;
}

void DrawFacialPoint(Mat img, Mat shape, Scalar color)
{
    int numLandmarks = shape.cols / 2;
    for (int index = 0; index < numLandmarks; index++) {
        int x = shape.at<float>(index);
        int y = shape.at<float>(index + numLandmarks);
        cv::circle(img, cv::Point(x, y), 2, color, -1);
    }
}

Mat ShapeRot(Mat shape, cv::Rect box, double pitch, double yaw)
{
    int numLandmarks = shape.cols / 2;
    cv::Point center(box.x + box.width / 2, box.y + box.height / 2);
    double radius = box.width / 2;
    Mat result(shape.rows, shape.cols, shape.type());
    for (int index = 0; index < numLandmarks; index++) {
        float x = shape.at<float>(index); // yaw
        float delta_cx = radius * sin(yaw * CV_PI / 180);
        float cx_new = center.x + delta_cx;
        float x_new = cx_new + (x - center.x) * cos(yaw * CV_PI / 180);
        float y = shape.at<float>(index + numLandmarks); // pitch
        float delta_cy = radius * sin(pitch * CV_PI / 180);
        float cy_new = center.y + delta_cy;
        float y_new = cy_new + (y - center.y) * cos(pitch * CV_PI / 180);
        result.at<float>(index) = x_new;
        result.at<float>(index + numLandmarks) = y_new;
    }
    return result;
}

Mat ShapeRot3D(Mat shape, cv::Rect box, Mat rot_matrix)
{
    int numLandmarks = shape.cols / 2;
    cv::Point center(box.x + box.width / 2, box.y + box.height / 2);
    double radius = box.width / 2;
    cv::Mat P = cv::Mat_<float>(3, numLandmarks + 1);
    Mat result(shape.rows, shape.cols, shape.type());
    for (int index = 0; index < numLandmarks; index++) {
        P.at<float>(0, index + 1) = (shape.at<float>(index) - center.x);
        P.at<float>(1, index + 1) = (shape.at<float>(index + numLandmarks) - center.y);
        P.at<float>(2, index + 1) = radius;
    }
    P.at<float>(0, 0) = P.at<float>(1, 0) = P.at<float>(2, 0) = 0;
    P = rot_matrix.rowRange(0, 2) * P;
    P.row(0) = P.row(0) + center.x;
    P.row(1) = P.row(1) + center.y;
    for (int index = 0; index < numLandmarks; index++) {
        result.at<float>(index) = 2 * shape.at<float>(index) - P.at<float>(0, index + 1);
        result.at<float>(index + numLandmarks) = 2 * shape.at<float>(index + numLandmarks) - P.at<float>(1, index + 1);
    }
    
    return result;
}

