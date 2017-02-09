//
//  CVCameraProvider.mm
//  opencvtest
//
//  Created by Engin Kurutepe on 16/01/15.
//  Copyright (c) 2015 Fifteen Jugglers Software. All rights reserved.
//

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif


#import "FJFaceDetector.h"
#import "UIImage+OpenCV.h"
#import <SeetaFaceDetectSDK/SeetaFaceDetectSDK.h>
#import <SeetaFaceAlignmentSDK/SeetaFaceAlignmentSDK.h>
#import <KCFTrackerSDK/KCFTrackerSDK.h>
#import <sdmSDK/sdmSDK.h>
#import "util.h"

using namespace cv;

#define HOG true
#define FIXEDWINDOW true
#define MULTISCALE true
#define LAB false

#define TRACK_TH_POS 0.4
#define TRACK_TH_SCALE 2.5
#define RESET 10*15

@interface FJFaceDetector () {
    
    CascadeClassifier _faceDetector;
    seeta::FaceDetection *_seetaFaceDetector;
    seeta::FaceAlignment *_seetaFaceAlignment;
    ldmarkmodel _sdmAlignment;
    vector<KCFTracker> _trackers;
    std::vector<cv::Vec3d> _face_poses;
    
    vector<cv::Rect> _faceRects;
    vector<cv::Mat> _faceImgs;


    bool _use_tracker;
    bool _use_tracker_color;
    bool _use_sdm_alignment;
    unsigned int _frameindex;
}

@property (nonatomic, assign) CGFloat scale;


@end

@implementation FJFaceDetector

- (instancetype)initWithCameraView:(UIImageView *)view scale:(CGFloat)scale {
    self = [super init];
    if (self) {
        self.videoCamera = [[CvVideoCamera alloc] initWithParentView:view];
        self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
        //self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
        self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPresetHigh;
        self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
        self.videoCamera.defaultFPS = 25;
        self.videoCamera.grayscaleMode = NO;
        self.videoCamera.delegate = self;
        self.scale = scale;
        
        NSString *faceCascadePath = [[NSBundle mainBundle] pathForResource:@"haarcascade_frontalface_alt2"
                                                                    ofType:@"xml"];
        
        const CFIndex CASCADE_NAME_LEN = 2048;
        char *CASCADE_NAME = (char *) malloc(CASCADE_NAME_LEN);
        CFStringGetFileSystemRepresentation( (CFStringRef)faceCascadePath, CASCADE_NAME, CASCADE_NAME_LEN);
        _faceDetector.load(CASCADE_NAME);
        
        free(CASCADE_NAME);
        //seetaFaceDetection
        NSString * detectModelPath = [[NSBundle mainBundle] pathForResource:@"seeta_fd_frontal_v1.0"
                                                                     ofType:@"bin"];
        const CFIndex PATH_NAME_LEN = 2048;
        char * detectModelPathChar = (char *) malloc(PATH_NAME_LEN);
        CFStringGetFileSystemRepresentation( (CFStringRef)detectModelPath, detectModelPathChar, PATH_NAME_LEN);
        _seetaFaceDetector = new seeta::FaceDetection(detectModelPathChar);
        _seetaFaceDetector->SetMinFaceSize(80);
        _seetaFaceDetector->SetScoreThresh(2.f);
        _seetaFaceDetector->SetImagePyramidScaleFactor(0.8f);
        _seetaFaceDetector->SetWindowStep(4, 4);
        free(detectModelPathChar);
        //seetaFaceAlignment
        NSString * alignmentModelPath = [[NSBundle mainBundle] pathForResource:@"seeta_fa_v1.1"
                                                                        ofType:@"bin"];
        char * alignmentModelPathChar = (char *) malloc(PATH_NAME_LEN);
        CFStringGetFileSystemRepresentation( (CFStringRef)alignmentModelPath, alignmentModelPathChar, PATH_NAME_LEN);
        
        _seetaFaceAlignment = new seeta::FaceAlignment(alignmentModelPathChar);
        free(alignmentModelPathChar);
        //sdm Alignment
        NSString * sdmAlignmentModelPath = [[NSBundle mainBundle] pathForResource:@"roboman-landmark-model"
                                                                           ofType:@"bin"];
        char * sdmAlignmentModelPathChar = (char *) malloc(PATH_NAME_LEN);
        CFStringGetFileSystemRepresentation( (CFStringRef)sdmAlignmentModelPath, sdmAlignmentModelPathChar, PATH_NAME_LEN);
        load_ldmarkmodel(sdmAlignmentModelPathChar, _sdmAlignment);
        free(sdmAlignmentModelPathChar);
        
        //KCF tracker
        _use_tracker = false;
        _use_tracker_color = false;
        _use_sdm_alignment = true;
        _frameindex = 0;
    }
    
    return self;
}

- (void)dealloc
{
    delete _seetaFaceDetector;
    delete _seetaFaceAlignment;
}

- (void)startCapture {
    [self.videoCamera start];
}

- (void)stopCapture; {
    [self.videoCamera stop];
}

- (NSArray *)detectedFaces {
    NSMutableArray *facesArray = [NSMutableArray array];
    for( vector<cv::Rect>::const_iterator r = _faceRects.begin(); r != _faceRects.end(); r++ )
    {
        CGRect faceRect = CGRectMake(_scale*r->x/480., _scale*r->y/640., _scale*r->width/480., _scale*r->height/640.);
        [facesArray addObject:[NSValue valueWithCGRect:faceRect]];
    }
    return facesArray;
}

- (UIImage *)faceWithIndex:(NSInteger)idx {
    
    cv::Mat img = self->_faceImgs[idx];
    
    UIImage *ret = [UIImage imageFromCVMat:img];
    
    return ret;
}



- (void)processImage:(cv::Mat&)image {
    // Do some OpenCV stuff with the image
    //[self detectAndDrawFacesSeetaOn:image scale:_scale];
    //[self detectAndTrackFacesOn:image scale:_scale];
    [self detectTrackTiltFaceOn:image scale:_scale];
    
    
}

- (void)detectAndDrawFacesOn:(Mat&) img scale:(double) scale
{
    int i = 0;
    double t = 0;
    
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    cvtColor( img, gray, COLOR_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    
    
    
    t = (double)cvGetTickCount();
    double scalingFactor = 1.1;
    int minRects = 2;
    cv::Size minSize(30,30);
    
    self->_faceDetector.detectMultiScale( smallImg, self->_faceRects,
                                         scalingFactor, minRects, 0,
                                         minSize );
    
    t = (double)cvGetTickCount() - t;
    //    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    vector<cv::Mat> faceImages;
    
    for( vector<cv::Rect>::const_iterator r = _faceRects.begin(); r != _faceRects.end(); r++, i++ )
    {
        cv::Mat smallImgROI;
        cv::Point center;
        Scalar color = colors[i%8];
        vector<cv::Rect> nestedObjects;
        rectangle(img,
                  cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                  cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                  color, 1, 8, 0);
        
        //eye detection is pretty low accuracy
        //        if( self->eyesDetector.empty() )
        //            continue;
        //
        smallImgROI = smallImg(*r);
        
        faceImages.push_back(smallImgROI.clone());
        //
        //
        //
        //        self->eyesDetector.detectMultiScale( smallImgROI, nestedObjects,
        //                                       1.1, 2, 0,
        //                                            cv::Size(5, 5) );
        //        for( vector<cv::Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++ )
        //        {
        //            center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
        //            center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
        //            int radius = cvRound((nr->width + nr->height)*0.25*scale);
        //            circle( img, center, radius, color, 3, 8, 0 );
        //        }
        
        
    }
    
    @synchronized(self) {
        self->_faceImgs = faceImages;
    }
    
}
- (void)detectAndDrawFacesSeetaOn:(Mat&) img scale:(double) scale
{
    int i = 0;
    double tDetect,tAlign;
    
    const static Scalar colors[] =  {
        CV_RGB(255,0,0),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    
    
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    
    cvtColor( img, gray, COLOR_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    
    
    seeta::ImageData img_data;
    img_data.data = smallImg.data;
    img_data.width = smallImg.cols;
    img_data.height = smallImg.rows;
    img_data.num_channels = 1;
    
    tDetect= cv::getTickCount();
    std::vector<seeta::FaceInfo> faces = _seetaFaceDetector->Detect(img_data);
    tDetect = cv::getTickCount() - tDetect;
    printf( "detection time = %g ms\n", tDetect/((double)cvGetTickFrequency()*1000.) );
    
    cv::Rect face_rect;
    vector<cv::Rect> localFaceRects;
    int32_t num_face = static_cast<int32_t>(faces.size());
    
    for (int32_t i = 0; i < num_face; i++) {
        
        face_rect.x = faces[i].bbox.x;
        face_rect.y = faces[i].bbox.y;
        face_rect.width = faces[i].bbox.width;
        face_rect.height = faces[i].bbox.height;
        
        if(face_rect.x>0 and face_rect.y>0 and (face_rect.width+face_rect.x)<smallImg.cols  and (face_rect.height+face_rect.y)< smallImg.rows){
            localFaceRects.push_back(face_rect);
            printf("scale:%f,x:%d,y:%d,w:%d,h:%d,imgw:%d,imgh:%d\n",scale,face_rect.x,face_rect.y, face_rect.width,face_rect.height,smallImg.cols,smallImg.rows);
        }
        // cv::rectangle(img, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);
    }
    
    vector<cv::Mat> faceImages;
    
    for( vector<cv::Rect>::const_iterator r = localFaceRects.begin(); r != localFaceRects.end(); r++, i++ )
    {
        cv::Mat smallImgROI;
        cv::Point center;
        Scalar color = colors[i%8];
        vector<cv::Rect> nestedObjects;
        rectangle(img,
                  cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                  cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                  color, 1, 8, 0);
        
        //eye detection is pretty low accuracy
        //        if( self->eyesDetector.empty() )
        //            continue;
        //
        smallImgROI = smallImg(*r);
        
        faceImages.push_back(smallImgROI.clone());
        
    }
    if (localFaceRects.size() > 0){
        seeta::FacialLandmark points[5];
        tAlign= cv::getTickCount();
        _seetaFaceAlignment->PointDetectLandmarks(img_data, faces[0], points);
        tAlign = cv::getTickCount() - tAlign;
        printf( "Alignment time = %g ms\n", tAlign/((double)cvGetTickFrequency()*1000.) );
        
        for (int i = 0; i<5; i++){
            cv::circle(img, cvPoint(points[i].x*scale, points[i].y*scale), 2, CV_RGB(0, 255, 0), CV_FILLED);
        }
    }
    @synchronized(self) {
        self->_faceImgs = faceImages;
    }
    
}

- (void)cvRect:(cv::Rect &)cv_rect toSeetaRect:(seeta::Rect &)seeta_rect
{
    seeta_rect.x = cv_rect.x;
    seeta_rect.y = cv_rect.y;
    seeta_rect.width = cv_rect.width;
    seeta_rect.height = cv_rect.height;
}

- (void)seetaRect:(seeta::Rect &)seeta_rect toCvRect:(cv::Rect &)cv_rect
{
    cv_rect.x = seeta_rect.x;
    cv_rect.y = seeta_rect.y;
    cv_rect.width = seeta_rect.width;
    cv_rect.height =seeta_rect.height;
}

- (bool)checkBoudary:(cv::Rect &) cv_rect inImg:(Mat &)img
{
    if(cv_rect.x>0 and cv_rect.y>0 and (cv_rect.width+cv_rect.x)<img.cols \
       and (cv_rect.height+cv_rect.y)< img.rows) {
        return true;
    }else{
        return false;
    }
}

- (bool)checkTracker:(KCFTracker &)_tracker
{
    if(_tracker.value < TRACK_TH_POS || _tracker.value_scale < TRACK_TH_SCALE
       || (_frameindex > RESET && (RESET >= 0)))
        return true;
    else
        return false;
}

- (void)putText:(Mat &)img useTracker:(bool)use_tracker
{
    if (use_tracker){
        putText(img, "Tracking..", cv::Point(100,100),CV_FONT_HERSHEY_SIMPLEX,1,cv::Scalar(128,128,128),4);
    }else{
        putText(img, "Detecting..", cv::Point(100,100),CV_FONT_HERSHEY_SIMPLEX,1,cv::Scalar(128,128,128),4);
    }
}

- (void) DrawFacialPoint:(Mat &)img shape:(Mat &)shape color:(Scalar)color
{
    int numLandmarks = shape.cols / 2;
    for (int index = 0; index < numLandmarks; index++) {
        int x = shape.at<float>(index);
        int y = shape.at<float>(index + numLandmarks);
        cv::circle(img, cv::Point(x, y), 4, color, -1);
    }
}

- (Mat)sdmAlignment:(Mat&)gray face_rect:(cv::Rect &)face_rect
{
    Mat current_shape;
    current_shape = align_mean(_sdmAlignment.getMeanShape(), face_rect);
    _sdmAlignment.FaceAlignment(current_shape, face_rect, gray);
    return current_shape;
}

- (void)scaleShape:(double)scale shape:(Mat &)shape
{
    for(int32_t i=0; i<shape.cols; i++){
        shape.at<float>(i) = shape.at<float>(i) * scale;
    }
}

- (void)detectAndTrackFacesOn:(Mat&) img scale:(double) scale
{
    
    double tDetect,tAlign;
    
    Mat img_4detect;
    cv::transpose(img, img_4detect);
    Mat gray, smallImg( cvRound (img_4detect.rows/scale), cvRound(img_4detect.cols/scale), CV_8UC1 );
    cvtColor( img_4detect, gray, COLOR_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    
    //data for tracker
    Mat smallImgColor( cvRound (img_4detect.rows/scale), cvRound(img_4detect.cols/scale), CV_8UC3 );
    resize( img_4detect, smallImgColor, smallImgColor.size(), 0, 0, INTER_LINEAR );
    cvtColor(smallImgColor, smallImgColor, COLOR_BGRA2BGR);
    
    Mat BGRImage(img_4detect.rows, img_4detect.cols, CV_8UC3 );
    cvtColor(img_4detect, BGRImage, COLOR_BGRA2BGR);
    
    //CvScalar scal = cvGet2D(img.data, 0, 0);
    //CvScalar scal3channel = cvGet2D(smallImgColor.data, 0, 0);
    
    seeta::ImageData img_data;
    img_data.data = smallImg.data;
    img_data.width = smallImg.cols;
    img_data.height = smallImg.rows;
    img_data.num_channels = 1;
    
    std::vector<seeta::FaceInfo> faces;
    vector<cv::Rect> localFaceRects;
    
    if (_use_tracker){
        
        _frameindex += 1;
        int32_t num_tracker = static_cast<int32_t>(_trackers.size());
        
        for(int32_t i=0; i<num_tracker; i++){
            
            KCFTracker& _tracker = _trackers[i];
            
            tDetect= cv::getTickCount();
            cv::Rect face_rect = _tracker.update(smallImgColor);
            tDetect = cv::getTickCount() - tDetect;
            printf( "tracking time = %g ms\n", tDetect/((double)cvGetTickFrequency()*1000.) );
            
            _use_tracker_color = true;
            localFaceRects.push_back(face_rect);
            
            if([self checkTracker:_tracker]){
                _use_tracker = false;
                _trackers.clear();
                _use_tracker_color = false;
                localFaceRects.clear();
                printf("fail track %f, %f, %f, %d, wdith:%d, height:%d\n", _tracker.value, _tracker.value_scale,_tracker.currentScaleFactor, _frameindex,smallImgColor.cols,smallImgColor.rows);
                break;
            }
        }
    }
    
    if (!_use_tracker){ //use detector
        tDetect= cv::getTickCount();
        faces = _seetaFaceDetector->Detect(img_data);
        tDetect = cv::getTickCount() - tDetect;
        //printf( "detection time = %g ms\n", tDetect/((double)cvGetTickFrequency()*1000.) );
        
        int32_t num_face = static_cast<int32_t>(faces.size());
        for (int32_t i = 0; i < num_face; i++) {
            cv::Rect face_rect;
            [self seetaRect:faces[i].bbox toCvRect:face_rect];
            if ([self checkBoudary:face_rect inImg:smallImg]){
                
                KCFTracker _tracker(true, true, true, false);
                _tracker.init( face_rect, smallImgColor );
                _trackers.push_back(_tracker);
                
                localFaceRects.push_back(face_rect);
            }
        }
        
        if (_trackers.size() > 0){
            printf("init tracker......detect %lu faces\n",faces.size());
            _use_tracker = true;
            _frameindex = 0;
        }
    }
    //Alignment
    
    int32_t num_face = static_cast<int32_t>(localFaceRects.size());
    
    for (int32_t i = 0; i < num_face; i++) {
        
        if(_use_sdm_alignment){
            tAlign= cv::getTickCount();
            Mat current_shape = [self sdmAlignment:smallImg face_rect:localFaceRects[i]];
            tAlign = cv::getTickCount() - tAlign;
            printf( "SDM Alignment time = %g ms\n", tAlign/((double)cvGetTickFrequency()*1000.) );
            [self scaleShape:scale shape:current_shape];
            [self DrawFacialPoint:BGRImage shape:current_shape color:CV_RGB(255, 0, 0)];
            
        }else{
            seeta::FacialLandmark points[5];
            seeta::FaceInfo f;
            [self cvRect:localFaceRects[i] toSeetaRect:f.bbox];
            tAlign= cv::getTickCount();
            _seetaFaceAlignment->PointDetectLandmarks(img_data, f, points);
            tAlign = cv::getTickCount() - tAlign;
            printf( "Alignment time = %g ms\n", tAlign/((double)cvGetTickFrequency()*1000.) );
            
            for (int i = 0; i<5; i++){
                cv::circle(BGRImage, cvPoint(points[i].x*scale, points[i].y*scale), 4, CV_RGB(0, 255, 0), CV_FILLED);
            }
        }
    }
    
    
    //draw box and align-point
    for( vector<cv::Rect>::const_iterator r = localFaceRects.begin(); r != localFaceRects.end(); r++ )
    {
        cv::Mat smallImgROI;
        cv::Point center;
        Scalar color;
        vector<cv::Rect> nestedObjects;
        
        if (_use_tracker_color){
            color = CV_RGB(0, 255, 0);
        }else{
            color = CV_RGB(0, 0, 255);
        }
        rectangle(BGRImage,
                  cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                  cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                  color, 4, 8, 0);
    }
    
    cvtColor(BGRImage, img_4detect,COLOR_BGR2BGRA);
    cv::transpose(img_4detect,img);
    [self putText:img useTracker:_use_tracker_color];
    @synchronized(self) {
        vector<cv::Mat> faceImages;
        self->_faceImgs = faceImages;
    }
    
}
- (void)detectTrackTiltFaceOn:(Mat&) img scale:(double) scale
{
    
    double tDetect,tAlign;
    
    cv::Rect face_box;
    cv::Vec3d pose;
    
    Mat img_4detect;
    cv::transpose(img, img_4detect);
    Mat gray, smallImg( cvRound (img_4detect.rows/scale), cvRound(img_4detect.cols/scale), CV_8UC1 );
    cvtColor( img_4detect, gray, COLOR_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    
    //data for tracker
    Mat smallImgColor( cvRound (img_4detect.rows/scale), cvRound(img_4detect.cols/scale), CV_8UC3 );
    resize( img_4detect, smallImgColor, smallImgColor.size(), 0, 0, INTER_LINEAR );
    cvtColor(smallImgColor, smallImgColor, COLOR_BGRA2BGR);
    
    Mat BGRImage(img_4detect.rows, img_4detect.cols, CV_8UC3 );
    cvtColor(img_4detect, BGRImage, COLOR_BGRA2BGR);
    
    //CvScalar scal = cvGet2D(img.data, 0, 0);
    //CvScalar scal3channel = cvGet2D(smallImgColor.data, 0, 0);
    
    seeta::ImageData img_data;
    img_data.data = smallImg.data;
    img_data.width = smallImg.cols;
    img_data.height = smallImg.rows;
    img_data.num_channels = 1;
    
    std::vector<seeta::FaceInfo> faces;
    vector<cv::Rect> localFaceRects;
    
    if (_use_tracker){
        
        _frameindex += 1;
        int32_t num_tracker = static_cast<int32_t>(_trackers.size());
        
        for(int32_t i=0; i<num_tracker; i++){
            
            KCFTracker& _tracker = _trackers[i];
            
            tDetect= cv::getTickCount();
            cv::Rect face_rect = _tracker.update(smallImgColor);
            tDetect = cv::getTickCount() - tDetect;
            printf( "tracking time = %g ms\n", tDetect/((double)cvGetTickFrequency()*1000.) );
            
            _use_tracker_color = true;
            localFaceRects.push_back(face_rect);
            
            if([self checkTracker:_tracker]){
                _use_tracker = false;
                _trackers.clear();
                _face_poses.clear();
                _use_tracker_color = false;
                localFaceRects.clear();
                printf("fail track %f, %f, %f, %d, wdith:%d, height:%d\n", _tracker.value, _tracker.value_scale,_tracker.currentScaleFactor, _frameindex,smallImgColor.cols,smallImgColor.rows);
                break;
            }
        }
    }
    
    if (!_use_tracker){ //use detector
        tDetect= cv::getTickCount();
        faces = _seetaFaceDetector->Detect(img_data);
        tDetect = cv::getTickCount() - tDetect;
        //printf( "detection time = %g ms\n", tDetect/((double)cvGetTickFrequency()*1000.) );
        
        int32_t num_face = static_cast<int32_t>(faces.size());
        for (int32_t i = 0; i < num_face; i++) {
            cv::Rect face_rect;
            [self seetaRect:faces[i].bbox toCvRect:face_rect];
            if ([self checkBoudary:face_rect inImg:smallImg]){
                
                KCFTracker _tracker(true, true, true, false);
                _tracker.init( face_rect, smallImgColor );
                _trackers.push_back(_tracker);
                
                localFaceRects.push_back(face_rect);
                cv::Vec3d po;
                _face_poses.push_back(po);
            }
        }
        
        if (_trackers.size() > 0){
            printf("init tracker......detect %lu faces\n",faces.size());
            _use_tracker = true;
            _frameindex = 0;
        }
    }
    //Alignment
    
    int32_t num_face = static_cast<int32_t>(localFaceRects.size());
    
    for (int32_t i = 0; i < num_face; i++) {
        
        if(_use_sdm_alignment){
            Mat current_shape;
            if (_frameindex>0){
                Mat  grayImage, rot, map_matrix;
                rot = ImgRotate(smallImgColor, map_matrix, _face_poses[i][2]);
                // convert to gray image
                face_box = localFaceRects[i];
                cv::cvtColor(rot, grayImage, CV_BGR2GRAY);
                cv::Point pt_center_input(face_box.x + face_box.width / 2, face_box.y + face_box.height / 2);
                cv::Point pt_center_output = GetPointPosition(map_matrix, pt_center_input);
                cv::Rect box_output(pt_center_output.x - face_box.width / 2, pt_center_output.y - face_box.height / 2, face_box.width, face_box.height);
                
                // alignment
                tAlign= cv::getTickCount();
                current_shape = [self sdmAlignment:grayImage face_rect:box_output];
                tAlign = cv::getTickCount() - tAlign;
                printf( "SDM Alignment time = %g ms\n", tAlign/((double)cvGetTickFrequency()*1000.) );
                /*   */
                ImgRotate2(rot, map_matrix, -_face_poses[i][2], smallImgColor);
                // inverse rotation current_shape
                int numLandmarks = current_shape.cols / 2;
                for (int index = 0; index < numLandmarks; index++) {
                    int x = current_shape.at<float>(index);
                    int y = current_shape.at<float>(index + numLandmarks);
                    cv::Point pt = GetPointPosition(map_matrix, cv::Point(x, y));
                    current_shape.at<float>(index) = pt.x;
                    current_shape.at<float>(index + numLandmarks) = pt.y;
                }
            }else{
                tAlign= cv::getTickCount();
                current_shape = [self sdmAlignment:smallImg face_rect:localFaceRects[i]];
                tAlign = cv::getTickCount() - tAlign;
            }
            // estimate pose
            _sdmAlignment.EstimateHeadPose(current_shape, pose);
            _face_poses[i][0] = pose[0];
            _face_poses[i][1] = pose[1];
            _face_poses[i][2] = pose[2];
            
            [self scaleShape:scale shape:current_shape];
            [self DrawFacialPoint:BGRImage shape:current_shape color:CV_RGB(255, 0, 0)];
            
        }else{
            seeta::FacialLandmark points[5];
            seeta::FaceInfo f;
            [self cvRect:localFaceRects[i] toSeetaRect:f.bbox];
            tAlign= cv::getTickCount();
            _seetaFaceAlignment->PointDetectLandmarks(img_data, f, points);
            tAlign = cv::getTickCount() - tAlign;
            printf( "Alignment time = %g ms\n", tAlign/((double)cvGetTickFrequency()*1000.) );
            
            for (int i = 0; i<5; i++){
                cv::circle(BGRImage, cvPoint(points[i].x*scale, points[i].y*scale), 4, CV_RGB(0, 255, 0), CV_FILLED);
            }
        }
    }
    
    
    //draw box and align-point
    for( vector<cv::Rect>::const_iterator r = localFaceRects.begin(); r != localFaceRects.end(); r++ )
    {
        cv::Mat smallImgROI;
        cv::Point center;
        Scalar color;
        vector<cv::Rect> nestedObjects;
        
        if (_use_tracker_color){
            color = CV_RGB(0, 255, 0);
        }else{
            color = CV_RGB(0, 0, 255);
        }
        rectangle(BGRImage,
                  cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                  cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                  color, 4, 8, 0);
    }
    
    cvtColor(BGRImage, img_4detect,COLOR_BGR2BGRA);
    cv::transpose(img_4detect,img);
    [self putText:img useTracker:_use_tracker_color];
    @synchronized(self) {
        vector<cv::Mat> faceImages;
        self->_faceImgs = faceImages;
    }
    
}

@end





