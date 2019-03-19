#include <stdio.h>
#include <math.h>
#include <iostream>

#include <cv.h>
#include <opencv2/highgui.hpp>
#include "opencv2/video/background_segm.hpp"
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc_c.h>
using namespace std;
using namespace cv;

#define H_THRESHOLD   1.000  // 图像无信号阈值

#define H_BLUR        100    // 图像模糊域值
#define H_impulseRate 0.08   // 图像噪声域值
#define H_COLORRate   0.8    // 偏色域值
#define H_DDB         700    // 对比度域值

#define d_thresh	  10     // 图像噪点阈值
#define L1_threshold  10	 // 图像噪点阈值
#define L2_threshold  10	 // 图像噪点阈值

#define brighThresh   220    // 图像过亮阈值
#define darkThresh    20	 // 图像过暗阈值

#define d_threshold   0      // 条纹检测
#define L1_threshold  0      // 条纹检测
#define L2_threshold  0      // 条纹检测

//#define STRIMG "rtsp://admin:123456@10.26.14.201/H264?ch=1&subtype=0"
#define STRIMG "偏色.Mp4"
#define WIDTH 640
#define HEIGHT 480


float fHcol(Mat img)
{
	if(img.channels() != 1)
		cvtColor(img, img, CV_BGR2GRAY);

	Mat imgHist;
	float range[] = {0, 255};
	const float* histRange = { range };
	int histSize = 255;

	cv::calcHist(&img, 1, 0, Mat(), imgHist, 1, &histSize, &histRange);
	float flpN[26];
	memset( flpN, 0.0, sizeof(float)*26 );

	int sum = 0;
	int nAll = img.rows*img.cols;
	for(int i=0; i<25; i++){
		for(int j=0; j<10; j++){
			sum += imgHist.at<float>(i*10+j);
		}
		flpN[i] =  (float)sum/nAll;
		sum = 0;
	}
	for(int i=250; i<255; i++){
		sum += imgHist.at<float>(i);
	}
	flpN[25] = (float)sum/nAll;
	
	float Hcol = 0.0;
	for( int i=0; i<26; i++){
		if( flpN[i] != 0){
			float sf = log(flpN[i]);
			Hcol += flpN[i]*sf;
		}
	}
	Hcol = 0-Hcol;

	// draw hist
	int hist_w = 400;
	int hist_h = 400;
	int bin_w = cvRound( (double)hist_w/histSize );
	Mat histImage( hist_w, hist_h, CV_8UC3, Scalar(0,0,0) );
	normalize(imgHist, imgHist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	for( int i=1; i<histSize; i++ ){
		line( histImage, Point( bin_w*(i-1), hist_h-cvRound(imgHist.at<float>(i-1))),
			Point(bin_w*(i), hist_h-cvRound(imgHist.at<float>(i))), Scalar(0,0,255), 2, 8, 0 );
	}
	namedWindow("calchist", 1);
	imshow("calchist", histImage);

	cout << Hcol <<endl;
	return Hcol;
}

int fSignal(Mat img)
{
	Mat frame = img.clone();
	Rect rcLt(0,0,img.cols/2-1, img.rows/2-1);
	Rect rcRt(img.cols/2, 0, img.cols/2-1, img.cols/2-1);
	Rect rcLb(0, img.rows/2, img.cols/2-1, img.rows/2-1);
	Rect rcRb(img.cols/2, img.rows/2, img.cols/2-1, img.rows/2-1);

	Mat imgLt = img(rcLt);
	Mat imgRt = img(rcRt);
	Mat imgLb = img(rcLb);
	Mat imgRb = img(rcRb);

	float fHall = 0.0;
	fHall = fHcol(img);
	float fav1 = fHcol(imgLt);
	float fav2 = fHcol(imgRt);
	float fav3 = fHcol(imgLb);
	float fav4 = fHcol(imgRb);
	
	float fav = (float)( fav1+fav2+fav3+fav4 )/4;
	float fimgH = fHall+0.2*fav;

	if( fimgH <= H_THRESHOLD )
		return -1;
	
	return 0;
}

int fNoise(Mat img, int size)
{
	Mat imgGray;
	if( img.channels() == 3)
		cvtColor(img, imgGray, CV_RGB2GRAY);

	int sum = 0;
	int icount = 0;
	int L1 = 0, L2 = 0;
	float impulseRate = 0.0f;
	int imgSize = size;
	int j=0;
	for( int i=1; i<imgGray.rows-1; i++ ){
		uchar* data = imgGray.ptr<uchar>(i);
		for( j=1; j<imgGray.cols-3; j++ ){
			uchar dataA = data[j];
			uchar dataB = data[j+2];
			if( abs(dataA-dataB) > d_thresh ){
				for( int k=-1; k<2; k++ ){
					uchar* dataT = imgGray.ptr<uchar>(i+k);
					for( int t=-1; t<2; t++ ){
						int sTemp = abs( dataT[j+t] - data[j] );
						int sTempB = abs( dataT[j+2+t] - data[j+2]);
						L1 += sTemp;
						L2 += sTempB;
					}
				}
				
				if( L1 > L2 && L1 > L1_threshold ){
					icount ++;
					if( j < imgGray.cols-3)
						j = j+3;
				}
				else if( L1 <= L2 && L2 > L2_threshold ){
					icount ++;
				}
				else{
				}
				if( j < imgGray.cols-3)
					j = j+3;
			}
		}	
	}
	impulseRate = (float)icount/imgSize;
	if( impulseRate > H_impulseRate)
		return -1;
	return 0;

}

int fSnow(Mat img, int size) 
{
	Mat imgGray;
	if( img.channels() != 1 )
		cvtColor( img, imgGray, CV_BGR2GRAY );

	for( int i=1; i<imgGray.rows; i++ ){
		uchar* data = imgGray.at<uchar*>(i);
		for( int j=1; j<imgGray.cols; j++ ){
			uchar dataA = data[j];
			uchar dataB = data[j+4];

			int d = abs( dataA - dataB );
			int avgA = 0, avgB = 0;
			if( d > d_thresh ){
				for( int k=-1; k<2; k++ ){
					uchar* dataAvg = imgGray.ptr<uchar>(i+k);
					for( int h=-1; h<-2; h++){
						avgA += dataAvg[j+h];
						avgB += dataAvg[j+4+h];
					}
				}
				avgA = avgA/9;
				avgB = avgB/9;

			}
			
		}
	}

	return 0;
}

int fBrightRate( Mat img, int size )
{
	//int sumY = 0;
	//int nSize = size;
	//for( int i=0; i<img.rows; i++ ){
	//	uchar* data = img.ptr<uchar>(i);
	//	for( int j=0; j<img.cols; j=j+3 ){
	//		int sum = 0.299*data[j] + 0.587*data[j+1] + 0.144*data[j+2];
	//		sumY += sum;
	//	}
	//}

	//float brightRate = (float)sumY/nSize;
	//if( brightRate > brighThresh )
	//	return 1;
	//else if( brightRate < darkThresh )
	//	return 2;
	
	Mat imgGray;
	cvtColor(img, imgGray, CV_BGR2GRAY);

	int sumY = 0;
	int nSize = size;
	for( int i=0; i<imgGray.rows; i++ ){
		uchar* data = imgGray.ptr<uchar>(i);
		for( int j=0; j<imgGray.cols; j++ ){
			sumY += data[j];
		}
	}

	// 计算平均灰度值
	sumY = sumY/size;

	if(sumY > brighThresh && brighThresh <= 255)
		return 1;  // 亮
	else if( sumY < darkThresh)
		return 2;  // 暗
	else
		return 0;
}


int fStripRate( Mat img, int size )
{
	vector<Mat> vImg;
	Mat imgGray, imgR, imgG, imgB;
	if( img.channels() != 1 ){
		cvtColor( img, imgGray,CV_BGR2GRAY );
	}
	
	cv::split(img, vImg);
	imgR = vImg[0].clone();
	imgG = vImg[1].clone();
	imgB = vImg[2].clone();

	int count = 0;
	for( int i=7; i<img.rows; i++ ){
		uchar* data = img.at<uchar*>(i);
		uchar* dataB = img.at<uchar*>(i-3);
		for( int j=7; j<img.cols; j++ ){
			int L1A = 0, L2AB = 0;
			// 1.水平方向, dataB为垂直方向i-3，大于d_threshold阈值，有可能出现条纹
			if( abs(dataB[j]-data[j]) > d_threshold ){
				for( int h=-6; h<7; h++){
					L1A  += abs(data[j+h]-data[j]);
					L2AB += abs(dataB[j+h]-data[j+h]);
				}
				if( L1A>L1_threshold && L2AB>L2_threshold ){
					count ++;
					j=j+3;
				}
			}
			L1A = 0;
			L2AB = 0;

			// 2.22.5°方向
			if( abs(dataB[j-2]-data[j]) > d_threshold ){
				for( int h=-3; h<4; h++ ){
					uchar* dataNew = img.at<uchar*>(i+h*2);
					uchar* dataBNew = img.at<uchar*>(i-3+h*2);
					L1A  += abs( dataNew[j+h]-data[j] );
					L2AB += abs( dataNew[j+h]- dataBNew[j+h]);
				}
				if( L1A>L1_threshold && L2AB>L2_threshold ){
					count ++;
					j=j+3;
				}
			}
			L1A = 0;
			L2AB = 0;

			// 3.45°方向
			if( abs(dataB[j-3]-data[j]) > d_threshold ){
				for( int h=-6; h<7; h++ ){
					uchar* dataNew = img.at<uchar*>(i+h);
					uchar* dataBNew = img.at<uchar*>(i+h-3);
					L1A  += abs( data[j] - dataNew[j+h] );
					L2AB += abs( dataNew[j+h] - dataBNew[j+h-3]);
				}
				if( L1A>L1_threshold && L2AB>L2_threshold ){
					count ++;
					j=j+3;
				}
			}
			L1A = 0;
			L2AB = 0;

			// 4.90°方向

		}
		i=i+10;
	}
	
	float stripRate = (float)count/size;

	return 0;
}

Mat clusterColorImgKmeans(Mat img, int clusterCounts)
{
	Scalar colorTab[] = 
	{
		Scalar(0, 0, 255),
		Scalar(0, 255, 0),
		Scalar(255, 100, 100),
		Scalar(255, 0, 255),
		Scalar(0, 255, 255),
		Scalar(255, 0, 0),
		Scalar(255, 255, 0),
		Scalar(255, 0, 100),
		Scalar(100, 100, 100),
		Scalar(50, 125, 125),
	};
	assert(img.channels() != 1);

	int rows = img.rows;
	int cols = img.cols;
	int channels = img.channels();

	
	Mat labels;
	Mat clusteredMat( rows, cols, CV_8UC3 );
	clusteredMat.setTo(Scalar::all(0));

	Mat pixels(rows*cols, 1, CV_32FC3);
	pixels.setTo(Scalar::all(0));

	for( int i=0; i<rows; i++ ){
		const uchar* idata = img.ptr<uchar>(i);
		float *pdata = pixels.ptr<float>(0);

		for( int j=0; j<cols*channels; j++){
			pdata[i*cols*channels+j] = saturate_cast<float>(idata[j]);
		}
	}

	cv::kmeans( pixels, clusterCounts, labels, 
		TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 0),
               5, KMEANS_PP_CENTERS );


	for( int i=0; i<rows; i++ ){
		for( int j=0; j<cols*channels; j+=channels ){
			cv::circle( clusteredMat, Point(j/channels, i), 1, colorTab[labels.at<int>
				(i*cols+(j/channels))]);
		}
	}

	return clusteredMat;
}

int fColorCast(Mat img)
{
	Mat imgC;
	if(img.channels() != 1)
		cvtColor(img, imgC, CV_BGR2HSV);
	vector<Mat > channels;
	cv::split(imgC, channels);
	Mat imgH = channels.at(0);

	Mat imgHist;
	float range[] = {0, 180};
	const float* histRange = { range };
	int histSize = 16;

	cv::calcHist(&imgH, 1, 0, Mat(), imgHist, 1, &histSize, &histRange);
	double minValue = 0;
	double maxValue = 0;

	cv::minMaxLoc(imgHist, &minValue, &maxValue);

	int size = imgH.rows*imgH.cols;
	float fcolorRate = (float)maxValue/size;
	cout << fcolorRate << endl;
	if( fcolorRate > H_COLORRate)
		return -1;
	return 0;
}

//1.把图像分割成N*M的区域。
//2/求每个区域的对比度：(max-min)/max(或者其他？).
//3.求总的平均对比度即为模糊率。

int fBlur_contrast( Mat img, int size )
{
	/*int sum_R = 0, sum_G = 0, sum_B = 0;
	for( int i=1; i<img.rows-1; i++ ){
		uchar* data = img.ptr<uchar>(i);
		for( int j=1; j<img.cols-1; j=j+3 ){
			int sum_Rtemp = 0, sum_Gtemp = 0, sum_Btemp = 0;
			for( int k=-1; k<2; k++ ){
				uchar* dataTemp = img.ptr<uchar>(i+k);
				for( int h=-1; h<2; h++ ){
					sum_Rtemp += abs( data[j] - dataTemp[j+h] );
					sum_Gtemp += abs( data[j+1] - dataTemp[j+1+h] );
					sum_Btemp += abs( data[j+2] - dataTemp[j+2+h] );
				}
			}
			sum_R += sum_Rtemp;
			sum_G += sum_Gtemp;
			sum_B += sum_Btemp;
		}
	}

	float Y = 0.2990*sum_R + 0.5870*sum_G + 0.1440*sum_B;
	float blurRate = (float)Y/size;

	if( blurRate < H_BLUR)
		return -1;
*/

	Mat imgGray;
	cvtColor(img , imgGray, CV_BGR2GRAY);
	
	double minValue1 = 0;
	double maxValue1 = 0;
	double minValue2 = 0;
	double maxValue2 = 0;
	double minValue3 = 0;
	double maxValue3 = 0;
	double minValue4 = 0;
	double maxValue4 = 0;

	int width = imgGray.cols;
	int height = imgGray.rows;
	Mat img1 = imgGray(Rect(0,0,width/2-1, height/2-1));
	Mat img2 = imgGray(Rect(width/2,0,width/2-1, height/2-1));
	Mat img3 = imgGray(Rect(0,height/2,width/2-1, height/2-1));
	Mat img4 = imgGray(Rect(width/2,height/2,width/2-1, height/2-1));

	//cv::minMaxIdx(img1, &minValue1, &maxValue1);
	//cv::minMaxIdx(img2, &minValue2, &maxValue2);
	//cv::minMaxIdx(img3, &minValue3, &maxValue3);
	//cv::minMaxIdx(img4, &minValue4, &maxValue4);

	//float rate1 = (float)(maxValue1-minValue1)/maxValue1;
	//float rate2 = (float)(maxValue2-minValue2)/maxValue2;
	//float rate3 = (float)(maxValue3-minValue3)/maxValue3;
	//float rate4 = (float)(maxValue4-minValue4)/maxValue4;

	//float rate = (float)(rate1+rate2+rate3+rate4)/4;
	
	int add = 0;
	for( int i=1; i<img1.rows-1; i++){
		for( int j=1; j<img1.cols-1; j++ ){
			uchar data = img1.ptr<uchar>(i)[j];

			uchar dataleft = img1.ptr<uchar>(i)[j-1];
			uchar dataright = img1.ptr<uchar>(i)[j+1];
			uchar dataup = img1.ptr<uchar>(i-1)[j];
			uchar datadown = img1.ptr<uchar>(i+1)[j];

			add += (data-dataleft)*(data-dataleft);
			add += (dataright-data)*(dataright-data);
			add += (data-dataup)*(data-dataup);
			add += (datadown-data)*(datadown-data);
		}
	}

	float ddb1 = add*1.0f/(4*(img1.cols-2)*(img1.rows-2)+2*(img1.cols-2)*3+2*(img1.rows-2)*3+4*2);

	add = 0;
	for( int i=1; i<img2.rows-1; i++){
		for( int j=1; j<img2.cols-1; j++ ){
			uchar data = img2.ptr<uchar>(i)[j];

			uchar dataleft = img2.ptr<uchar>(i)[j-1];
			uchar dataright = img2.ptr<uchar>(i)[j+1];
			uchar dataup = img2.ptr<uchar>(i-1)[j];
			uchar datadown = img2.ptr<uchar>(i+1)[j];

			add += (data-dataleft)*(data-dataleft);
			add += (dataright-data)*(dataright-data);
			add += (data-dataup)*(data-dataup);
			add += (datadown-data)*(datadown-data);
		}
	}

	float ddb2 = add*1.0f/(4*(img2.cols-2)*(img2.rows-2)+2*(img2.cols-2)*3+2*(img2.rows-2)*3+4*2);
	add = 0;
	for( int i=1; i<img3.rows-1; i++){
		for( int j=1; j<img3.cols-1; j++ ){
			uchar data = img3.ptr<uchar>(i)[j];

			uchar dataleft = img3.ptr<uchar>(i)[j-1];
			uchar dataright = img3.ptr<uchar>(i)[j+1];
			uchar dataup = img3.ptr<uchar>(i-1)[j];
			uchar datadown = img3.ptr<uchar>(i+1)[j];

			add += (data-dataleft)*(data-dataleft);
			add += (dataright-data)*(dataright-data);
			add += (data-dataup)*(data-dataup);
			add += (datadown-data)*(datadown-data);
		}
	}

	float ddb3 = add*1.0f/(4*(img3.cols-2)*(img3.rows-2)+2*(img3.cols-2)*3+2*(img3.rows-2)*3+4*2);

	add = 0;
	for( int i=1; i<img4.rows-1; i++){
		for( int j=1; j<img4.cols-1; j++ ){
			uchar data = img4.ptr<uchar>(i)[j];

			uchar dataleft = img4.ptr<uchar>(i)[j-1];
			uchar dataright = img4.ptr<uchar>(i)[j+1];
			uchar dataup = img4.ptr<uchar>(i-1)[j];
			uchar datadown = img4.ptr<uchar>(i+1)[j];

			add += (data-dataleft)*(data-dataleft);
			add += (dataright-data)*(dataright-data);
			add += (data-dataup)*(data-dataup);
			add += (datadown-data)*(datadown-data);
		}
	}

	float ddb4 = add*1.0f/(4*(img4.cols-2)*(img4.rows-2)+2*(img4.cols-2)*3+2*(img4.rows-2)*3+4*2);
	float ddb  = (float)(ddb1+ddb2+ddb3+ddb4)/4;

	if( ddb < H_DDB )
		return -1;
	return 0;
}

int fBlur_Laplacian(Mat img){
	Mat imgGray;
	cvtColor(img , imgGray, CV_BGR2GRAY);
	Mat dst;
	Laplacian(imgGray, dst, CV_64F);
	
	imshow("laplacian",dst);

	Mat tmp_m, tmp_sd;  
	double m = 0, sd = 0;  

	meanStdDev(dst, tmp_m, tmp_sd);  
	m = tmp_m.at<double>(0,0);  
	sd = tmp_sd.at<double>(0,0);  
	float var = sd * sd;
	cout << var <<endl;
	if (var < H_BLUR)
	{
		return -1;
	}
	return 0;
}
int main2()
{
	/*Kmeans
	const int MAX_CLUSTERS = 5;
    Scalar colorTab[] =
    {
        Scalar(0, 0, 255),
        Scalar(0,255,0),
        Scalar(255,100,100),
        Scalar(255,0,255),
        Scalar(0,255,255)
    };

    Mat img(500, 500, CV_8UC3);
    RNG rng(12345);

    for(;;)
    {
        int k, clusterCount = rng.uniform(2, MAX_CLUSTERS+1);
        int i, sampleCount = rng.uniform(1, 1001);
        Mat points(sampleCount, 1, CV_32FC2), labels;

        clusterCount = MIN(clusterCount, sampleCount);
        Mat centers;

        for( k = 0; k < clusterCount; k++ )
        {
            Point center;
            center.x = rng.uniform(0, img.cols);
            center.y = rng.uniform(0, img.rows);
            Mat pointChunk = points.rowRange(k*sampleCount/clusterCount,
                                             k == clusterCount - 1 ? sampleCount :
                                             (k+1)*sampleCount/clusterCount);
            rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
        }
		        randShuffle(points, 1, &rng);

        kmeans(points, clusterCount, labels,
            TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
               3, KMEANS_PP_CENTERS, centers);

        img = Scalar::all(0);

        for( i = 0; i < sampleCount; i++ )
        {
            int clusterIdx = labels.at<int>(i);
            Point ipt = points.at<Point2f>(i);
            circle( img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA );
        }

        imshow("clusters", img);

        char key = (char)waitKey();
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;
    }
	*/
	


	cv::VideoCapture cap;
	string filename = STRIMG;
	cap.open(filename);  
	Mat img;
	//cap >> img;
	//int size = img.rows*img.cols;
	Mat frame, frameN;

	while(1) {	
		cap >> frame;

		//cv::resize(frame, frameN, Size(WIDTH, HEIGHT));
		int size = WIDTH*HEIGHT;

		// 噪点检测
		int iNoise = fNoise(frame, size);
		if( iNoise < 0 )
			printf("图像有噪声\n");

		// 雪花检测
	//	fSnow(frame, size);

		// 有无信号检测
		int isignal = fSignal(frame);
		if( isignal < 0 )
			printf("图像无信号\n");

		// 亮度异常检测
		int irezult = fBrightRate(frame, size);
		switch(irezult)
		{
		case 1:
			printf("图像过亮\n");
			break;
		case 2:
			printf("图像过暗\n");
			break;
		default:
			break;
		}

		// 模糊检测
		int iblur = fBlur_contrast(frame,size);
		if( iblur < 0 )
			printf("图像模糊\n");

		// 偏色检测
		int iColor = fColorCast(frame);
		if( iColor < 0)
			printf("图像偏色\n");

		
		cv::namedWindow("frame",1);
		imshow("frame", frame);
		
		//cv::namedWindow("frameN", 1);
		//imshow("frameN", frameN);
		cv::waitKey(3);
	}
	
	return 0;

}
int main(){
	Mat image;
	image = imread("lena.jpg");
	/*int iColor = fColorCast(image);
	if( iColor < 0)
		printf("图像偏色\n");*/
	float f = fHcol(image);
	waitKey(0);
	return 0;
}