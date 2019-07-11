#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "addseam.hpp"
#include "global_warp.hpp"
using namespace cv;
void my_imfillholes(Mat &src)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(src,contours,hierarchy,cv::RETR_LIST,CHAIN_APPROX_NONE);
	if (!contours.empty() && !hierarchy.empty())
	{
		for (int idx =0;idx<contours.size();idx++)
		{
			if (contours[idx].size()<100)
            {
				drawContours(src,contours,idx, Scalar::all(0),CV_FILLED,8);
			}
		}
	}
}
int mask_fg(Mat& rgbImg,int thrs,Mat &mask)
{
	Mat grayImg;
	cvtColor(rgbImg,grayImg,CV_BGR2GRAY);
	int rows = rgbImg.rows;
	int cols = rgbImg.cols;
	for (int i=0;i<rows;i++){
		for (int j=0;j<cols;j++){
			if (grayImg.at<uchar>(i,j)>thrs-3)
				mask.at<uchar>(i,j)=1;
			else
				mask.at<uchar>(i,j)=0;
		}
	}
	my_imfillholes(mask);
	for(int i=0;i<mask.rows;i++)
    {
		mask.at<uchar>(i,0)=1;
		mask.at<uchar>(i,mask.cols-1)=1;
	}
	for(int i=0;i<mask.cols;i++)
	{
		mask.at<uchar>(0,i)=1;
		mask.at<uchar>(mask.rows-1,i)=1;
	}
	filter2D(mask, mask, mask.depth(), Mat::ones(7, 7, CV_8UC1));
	filter2D(mask, mask, mask.depth(), Mat::ones(2, 2, CV_8UC1));
	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			if (mask.at<uchar>(i,j)>1)
				mask.at<uchar>(i,j)=1;
			else
				mask.at<uchar>(i,j)=0;
		}
	}
	return 0;
}
int main(int argc, char *argv[])
{
    int ths=254;
    Mat img = imread("2_input.jpg",1);
    if(img.empty())
       return -1;
    Mat grayImg;
    Mat new_img;
    Mat outimg;
    Mat oriimg;
    Mat orimask;
    int col=img.cols;
    int row=img.rows;
    int s=col*row;
    double scale=sqrt((double)250000/s);
    resize(img,oriimg,Size(col*scale,row*scale) ,0, 0, INTER_NEAREST);
    cvtColor(img,grayImg, CV_BGR2GRAY);
    Mat mask(Size(img.cols,img.rows),CV_8UC1);
    resize(mask,orimask,Size(col*scale,row*scale),0, 0, INTER_NEAREST);
    Mat disimg=Mat::zeros(Size(oriimg.cols,oriimg.rows),CV_32FC2);
    mask_fg(oriimg, ths, orimask);
    localwrap(oriimg,orimask,disimg,new_img);
    global_warp(oriimg,disimg,orimask,outimg);
    resize(outimg,outimg,Size(col,row),0, 0, INTER_NEAREST);
    imshow("final",outimg);
    waitKey(0);
    return 0;
}

