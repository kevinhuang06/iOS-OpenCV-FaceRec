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
using namespace cv;

@interface FJFaceDetector () {
    
    CascadeClassifier _faceDetector;
    seeta::FaceDetection *_seetaFaceDetector;
    seeta::FaceAlignment *_seetaFaceAlignment;
    KCFTracker *_tracker;
    
    vector<cv::Rect> _faceRects;
    vector<cv::Mat> _faceImgs;
    
}

@property (nonatomic, assign) CGFloat scale;


@end

@implementation FJFaceDetector

- (instancetype)initWithCameraView:(UIImageView *)view scale:(CGFloat)scale {
    self = [super init];
    if (self) {
        self.videoCamera = [[CvVideoCamera alloc] initWithParentView:view];
        self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
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
        //KCF tracker
        bool HOG = true;
        bool FIXEDWINDOW = true;
        bool MULTISCALE = true;
        bool LAB = false;
        _tracker = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    }
    
    return self;
}

- (void)dealloc
{
    delete _seetaFaceDetector;
    delete _seetaFaceAlignment;
    delete _tracker;
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
    [self detectAndDrawFacesSeetaOn:image scale:_scale];
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

@end





