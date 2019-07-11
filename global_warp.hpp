#include<bits/stdc++.h>
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<Eigen\Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <opencv2/core/eigen.hpp>
const double pi=acos(-1.0);
using namespace std ;
using namespace cv;
using namespace Eigen;
void drawGrid(Mat& gridmask,Mat& img,Mat& outimage)
{
    outimage=img.clone();
	for(int i=0;i<gridmask.rows;i++)
    {
		for(int j=0;j<gridmask.cols;j++)
		{
			if (gridmask.at<float>(i,j)==1)
			{
				outimage.at<Vec3b>(i,j)[0]=0;
				outimage.at<Vec3b>(i,j)[1]=255;
				outimage.at<Vec3b>(i,j)[2]=0;
			}
		}
	}
}
void getvertices(int yy,int xx,Mat& ygrid,Mat& xgrid,Mat& vx,Mat& vy)
{
    vx.at<float>(0,0)=xgrid.at<float>(yy,xx);
    vx.at<float>(0,1)=xgrid.at<float>(yy,xx+1);
    vx.at<float>(1,0)=xgrid.at<float>(yy+1,xx);
    vx.at<float>(1,1)=xgrid.at<float>(yy+1,xx+1);
    vy.at<float>(0,0)=ygrid.at<float>(yy,xx);
    vy.at<float>(0,1)=ygrid.at<float>(yy,xx+1);
    vy.at<float>(1,0)=ygrid.at<float>(yy+1,xx);
    vy.at<float>(1,1)=ygrid.at<float>(yy+1,xx+1);
    //cout<<vy<<endl;
}
void calintersec(Mat& s1,Mat& s2,int& flag,Mat& p)
{
    int a1=(int)s1.at<float>(1,1)-(int)s1.at<float>(0,1);
    int a2=(int)s2.at<float>(1,1)-(int)s2.at<float>(0,1);
    int b1=(int)s1.at<float>(0,0)-(int)s1.at<float>(1,0);
    int b2=(int)s2.at<float>(0,0)-(int)s2.at<float>(1,0);
    int c1=(int)s1.at<float>(1,0)*(int)s1.at<float>(0,1)-(int)s1.at<float>(1,1)*(int)s1.at<float>(0,0);
    int c2=(int)s2.at<float>(1,0)*(int)s2.at<float>(0,1)-(int)s2.at<float>(1,1)*(int)s2.at<float>(0,0);
	Mat ab_matrix=(Mat_<float>(2,2)<<a1,b1,a2,b2);
	Mat C_matrix=(Mat_<float>(2,1)<<-c1,-c2);
	p=p=ab_matrix.inv()*C_matrix;
	p=p.t();
	flag=0;
	if((p.at<float>(0,0)-s1.at<float>(0,0))*(p.at<float>(0,0)-s1.at<float>(1,0))<=0)
    {
        if((p.at<float>(0,0)-s2.at<float>(0,0))*(p.at<float>(0,0)-s2.at<float>(1,0))<=0)
        {
            if((p.at<float>(0,1)-s1.at<float>(0,1))*(p.at<float>(0,1)-s1.at<float>(1,1))<=0)
            {
                if((p.at<float>(0,1)-s2.at<float>(0,1))*(p.at<float>(0,1)-s2.at<float>(1,1))<=0)
                    flag=1;
            }
        }
    }
}
int checkIsIn(Mat& vy,Mat& vx,int pstartx,int pstarty,int pendx,int pendy){
	int min_x=min(pstartx,pendx);
	int min_y=min(pstarty,pendy);
	int max_x=max(pstartx,pendx);
	int max_y=max(pstarty,pendy);
	if((min_x<vx.at<float>(0,0)&&min_x<vx.at<float>(1,0))||max_x>vx.at<float>(0,1)&&max_x>vx.at<float>(1,1))
	{
		return 0;
	}
	else if((min_y<vy.at<float>(0,0)&&min_y<vy.at<float>(0,1))||(max_y>vy.at<float>(1,0)&&max_y>vy.at<float>(1,1)))
	{
		return 0;
	}
	else
	{
		return 1;
	}
}
void trans_mat(Mat& vx,Mat& vy,Mat& p,Mat& TP)
{//p=2*1
    /*
    Mat quan(1,8,CV_32FC1);
    quan.at<float>(0,0)=vx.at<float>(0,0);quan.at<float>(0,1)=vy.at<float>(0,0);
    quan.at<float>(0,2)=vx.at<float>(0,1);quan.at<float>(0,3)=vy.at<float>(0,1);
    quan.at<float>(0,4)=vx.at<float>(1,0);quan.at<float>(0,5)=vy.at<float>(1,0);
    quan.at<float>(0,6)=vx.at<float>(1,1);quan.at<float>(0,7)=vy.at<float>(1,1);
    Mat I=Mat::eye(8,8,CV_32FC1);
    Mat II=Mat::zeros(1,1,CV_32FC1);
    Mat zz=Mat::zeros(2,8,CV_32FC1);
    hconcat(I,quan.t(),I);
    hconcat(quan,II,II);
    vconcat(I,II,I);
    hconcat(zz,p,zz);
    Mat res=zz*I.inv();
    T=res.colRange(0,8);*/
    Mat quan(2,4,CV_32FC1);
    quan.at<float>(0,0)=vx.at<float>(0,0);quan.at<float>(1,0)=vy.at<float>(0,0);
    quan.at<float>(0,1)=vx.at<float>(0,1);quan.at<float>(1,1)=vy.at<float>(0,1);
    quan.at<float>(0,2)=vx.at<float>(1,0);quan.at<float>(1,2)=vy.at<float>(1,0);
    quan.at<float>(0,3)=vx.at<float>(1,1);quan.at<float>(1,3)=vy.at<float>(1,1);
    Mat zz=Mat::eye(4,4,CV_32FC1);
    Mat z=Mat::zeros(2,2,CV_32FC1);
    Mat z1=Mat::zeros(4,1,CV_32FC1);
    Mat tmp1,tmp2,tmp;
    hconcat(zz,quan.t(),tmp1);
    hconcat(quan,z,tmp2);
    vconcat(tmp1,tmp2,tmp1);
    vconcat(z1,p,tmp);
    Mat x=tmp1.inv()*tmp;
    Mat TT=x.rowRange(0,4);
    if(norm(quan*TT-p)>0.0001) cout<<"error"<<endl;
    Mat T(2,8,CV_32FC1);
    T.at<float>(0,0)=TT.at<float>(0,0);T.at<float>(1,0)=0;
    T.at<float>(0,1)=0;T.at<float>(1,1)=TT.at<float>(0,0);
    T.at<float>(0,2)=TT.at<float>(1,0);T.at<float>(1,2)=0;
    T.at<float>(0,3)=0;T.at<float>(1,3)=TT.at<float>(1,0);
    T.at<float>(0,4)=TT.at<float>(2,0);T.at<float>(1,4)=0;
    T.at<float>(0,5)=0;T.at<float>(1,5)=TT.at<float>(2,0);
    T.at<float>(0,6)=TT.at<float>(3,0);T.at<float>(1,6)=0;
    T.at<float>(0,7)=0;T.at<float>(1,7)=TT.at<float>(3,0);
    TP=T.clone();
}
void getLinTrans(float pstart_y, float pstart_x, Mat& yVq, Mat& xVq, Mat& T,int& flag)
{
	Mat V(8, 1, CV_32FC1);
	V.at<float>(0, 0) = xVq.at<float>(0, 0); V.at<float>(1, 0) = yVq.at<float>(0, 0);
	V.at<float>(2, 0) = xVq.at<float>(0, 1); V.at<float>(3, 0) = yVq.at<float>(0, 1);
	V.at<float>(4, 0) = xVq.at<float>(1, 0); V.at<float>(5, 0) = yVq.at<float>(1, 0);
	V.at<float>(6, 0) = xVq.at<float>(1, 1); V.at<float>(7, 0) = yVq.at<float>(1, 1);
	Mat v1(2, 1, CV_32FC1), v2(2, 1, CV_32FC1), v3(2, 1, CV_32FC1), v4(2, 1, CV_32FC1);
	v1.at<float>(0, 0) = xVq.at<float>(0, 0); v1.at<float>(1, 0) = yVq.at<float>(0, 0);
	v2.at<float>(0, 0) = xVq.at<float>(0, 1); v2.at<float>(1, 0) = yVq.at<float>(0, 1);
	v3.at<float>(0, 0) = xVq.at<float>(1, 0); v3.at<float>(1, 0) = yVq.at<float>(1, 0);
	v4.at<float>(0, 0) = xVq.at<float>(1, 1); v4.at<float>(1, 0) = yVq.at<float>(1, 1);
	Mat v21 = v2 - v1, v31 = v3 - v1, v41 = v4 - v1;
	Mat p(2, 1, CV_32FC1);
	p.at<float>(0, 0) = pstart_x;  p.at<float>(1, 0) = pstart_y;
	Mat p1 = p - v1;
	double a1 = v31.at<float>(0, 0), a2 = v21.at<float>(0, 0),          //x
		a3 = v41.at<float>(0, 0) - v31.at<float>(0, 0) - v21.at<float>(0, 0);
	double b1 = v31.at<float>(1, 0), b2 = v21.at<float>(1, 0),      //y
		b3 = v41.at<float>(1, 0) - v31.at<float>(1, 0) - v21.at<float>(1, 0);
	double px = p1.at<float>(0, 0), py = p1.at<float>(1, 0);
	Mat tvec, mat_t;
	double t1n, t2n;
	double a, b, c;
	if (a3 == 0 && b3 == 0)
    {
		hconcat(v31, v21, mat_t);
		tvec = mat_t.inv()*p1;
		t1n = tvec.at<float>(0, 0);
		t2n = tvec.at<float>(1, 0);
	}
	else
	{
		a = (b2*a3 - a2*b3);
		b = (-a2*b1 + b2*a1 + px*b3 - a3*py);
		c = px*b1 - py*a1;
		if (a == 0)
		{
			t2n = -c / b;
		}
		else
		{
			if ((b*b - 4 * a*c) > 0)
			{
				t2n = (-b - sqrt(b*b - 4 * a*c)) / (2 * a);
			}
			else
			{
				t2n = (-b - 0) / (2 * a);
			}
		}
		if (abs(a1 + t2n*a3) <= 0.0000001)
		{
			t1n = (py - t2n*b2) / (b1 + t2n*b3);
		}
		else
		{
			t1n = (px - t2n*a2) / (a1 + t2n*a3);
		}
	}
	Mat m1 = v1 + t1n*(v3 - v1);
	Mat m4 = v2 + t1n*(v4 - v2);
	Mat ptest = m1 + t2n*(m4 - m1);
	double v1w = 1 - t1n - t2n + t1n*t2n;
	double v2w = t2n - t1n*t2n;
	double v3w = t1n - t1n*t2n;
	double v4w = t1n*t2n;
	Mat out(2, 8, CV_32FC1);
	out.at<float>(0, 0) = v1w;  out.at<float>(1, 0) = 0;
	out.at<float>(0, 1) = 0;    out.at<float>(1, 1) = v1w;
	out.at<float>(0, 2) = v2w;  out.at<float>(1, 2) = 0;
	out.at<float>(0, 3) = 0;    out.at<float>(1, 3) = v2w;
	out.at<float>(0, 4) = v3w;  out.at<float>(1, 4) = 0;
	out.at<float>(0, 5) = 0;    out.at<float>(1, 5) = v3w;
	out.at<float>(0, 6) = v4w;  out.at<float>(1, 6) = 0;
	out.at<float>(0, 7) = 0;    out.at<float>(1, 7) = v4w;
	T = out.clone();
	if(norm(T*V - p) > 0.01)
    {
        flag=1;
        //cout<<norm(T*V - p)<<endl;
    }
}
void blkdiag(Mat& input1, Mat &input2, Mat& output)
{
	if (input1.type() == CV_8UC1){
		Mat out = Mat::zeros(input1.rows+input2.rows, input1.cols+input2.cols, CV_8UC1);
		for (int i = 0; i < input1.rows; i++){
			for (int j = 0; j < input1.cols; j++){
				out.at<uchar>(i, j) = input1.at<uchar>(i, j);
			}
		}
		for (int i = 0; i < input2.rows; i++){
			for (int j = 0; j < input2.cols; j++){
				out.at<uchar>(i + input1.rows, j + input1.cols) = input2.at<uchar>(i, j);
			}
		}
		output = out;
	}
	else if (input1.type() == CV_32FC1){
		Mat out = Mat::zeros(input1.rows+input2.rows,input1.cols+ input2.cols, CV_32FC1);
		for (int i = 0; i < input1.rows; i++){
			for (int j = 0; j < input1.cols; j++){
				out.at<float>(i, j) = input1.at<float>(i, j);
			}
		}
		for (int i = 0; i < input2.rows; i++){
			for (int j = 0; j < input2.cols; j++){
				out.at<float>(i + input1.rows, j + input1.cols) = input2.at<float>(i, j);
			}
		}
		output = out;
	}
	else if (input1.type() == CV_32SC1){
		Mat out = Mat::zeros(input1.rows+input2.rows, input1.cols+input2.cols, CV_32SC1);
		for (int i = 0; i < input1.rows; i++){
			for (int j = 0; j < input1.cols; j++){
				out.at<int>(i, j) = input1.at<int>(i, j);
			}
		}
		for (int i = 0; i < input2.rows; i++){
			for (int j = 0; j < input2.cols; j++){
				out.at<int>(i + input1.rows, j + input1.cols) = input2.at<int>(i, j);
			}
		}
		output = out;
	}
}

void drawGridmask(Mat& ygrid,Mat& xgrid,int rows, int cols, Mat& gridmask)
{
	int xgridN = ygrid.cols;
	int ygridN = ygrid.rows;
    Mat outmask=Mat::zeros(rows, cols, CV_32FC1);
	double m;
    for (int y = 0; y < ygridN; y++){
		for (int x = 0; x < xgridN; x++){
			if (y != 0){
				if (ygrid.at<float>(y, x) != ygrid.at<float>(y - 1, x)){
					for (int i = ygrid.at<float>(y - 1, x); i <= ygrid.at<float>(y, x); i++){
						m = double(xgrid.at<float>(y, x) - xgrid.at<float>(y - 1, x)) /
							(ygrid.at<float>(y, x) - ygrid.at<float>(y - 1, x));
						outmask.at<float>(i, int(xgrid.at<float>(y - 1, x) +
							int(m*(i - ygrid.at<float>(y - 1, x))))) = 1;
					}
				}
			}
			if (x != 0){
				if (xgrid.at<float>(y, x) != xgrid.at<float>(y, x - 1)){
					for (int j = xgrid.at<float>(y, x - 1); j <= xgrid.at<float>(y, x); j++){
						m = double(ygrid.at<float>(y, x) - ygrid.at<float>(y, x - 1)) /
							(xgrid.at<float>(y, x) - xgrid.at<float>(y, x - 1));
						outmask.at<float>(int(ygrid.at<float>(y, x - 1) +
							int(m*(j - xgrid.at<float>(y, x - 1)))), j) = 1;
					}
				}
			}
		}
	}
	gridmask = outmask.clone();
}
void global_warp(Mat& img,Mat& disimg,Mat& mask,Mat& output)
{
    int cols=img.cols;
    int rows=img.rows;
    int x_num=30;
    int y_num=20;
    /**********rectangle grid************/
    Mat xgrid(y_num,x_num,CV_32FC1);
    Mat ygrid(y_num,x_num,CV_32FC1);
    int x=0,y=0;
    for(double i=0;i<rows;i+=1.0*(rows-1)/(y_num-1))
    {
        for(double j=0;j<cols;j+=1.0*(cols-1)/(x_num-1))
        {
            xgrid.at<float>(x,y)=(int)j;
            ygrid.at<float>(x,y)=(int)i;
            y++;
        }
        x++;
        y=0;
    }
    /*
    Mat gridmask,imageGrided;
    drawGridmask(ygrid,xgrid,rows,cols,gridmask);
	drawGrid(gridmask,img,imageGrided);
	imshow("imageGrided",imageGrided);
	waitKey(1000);*/
	/**************************/

	/***********warp grid********/
	Mat warp_xgrid=xgrid.clone();
	Mat warp_ygrid=ygrid.clone();
	//cout<<warp_xgrid.rows<<" "<<warp_xgrid.cols<<endl;
	for(int i=0;i<warp_xgrid.rows;i++)
    {
        for(int j=0;j<warp_xgrid.cols;j++)
        {
            warp_xgrid.at<float>(i,j)=xgrid.at<float>(i,j)-disimg.at<Vec2f>(ygrid.at<float>(i,j),xgrid.at<float>(i,j))[1];
            warp_ygrid.at<float>(i,j)=ygrid.at<float>(i,j)-disimg.at<Vec2f>(ygrid.at<float>(i,j),xgrid.at<float>(i,j))[0];
        }
    }

/*
    Mat gridmask1,imageGrided1;
    drawGridmask(warp_ygrid,warp_xgrid,rows,cols,gridmask1);
	drawGrid(gridmask1,img,imageGrided1);
	imshow("imageGrided1",imageGrided1);
	waitKey(1000);//fan*/
	/*******************/

	/**********shape reserve mat*************/

	int gridrows=y_num-1;
	int gridcols=x_num-1;
	Mat **Ses=new Mat*[gridrows];
	for(int i=0;i<gridrows;i++) Ses[i]=new Mat[gridcols];
	Mat Aq(8,4,CV_32FC1);
	Mat tmp(4,2,CV_32FC1);
	for(int i=0;i<gridrows;i++)
    {
        for(int j=0;j<gridcols;j++)
        {
            tmp.at<float>(0,0)=warp_xgrid.at<float>(i,j);
            tmp.at<float>(0,1)=warp_ygrid.at<float>(i,j);
            tmp.at<float>(1,0)=warp_xgrid.at<float>(i,j+1);
            tmp.at<float>(1,1)=warp_ygrid.at<float>(i,j+1);
            tmp.at<float>(2,0)=warp_xgrid.at<float>(i+1,j);
            tmp.at<float>(2,1)=warp_ygrid.at<float>(i+1,j);
            tmp.at<float>(3,0)=warp_xgrid.at<float>(i+1,j+1);
            tmp.at<float>(3,1)=warp_ygrid.at<float>(i+1,j+1);
            for(int k=0;k<4;k++)
            {
                Aq.at<float>(k*2,0)=tmp.at<float>(k,0);
                Aq.at<float>(k*2,1)=-tmp.at<float>(k,1);
                Aq.at<float>(k*2,2)=1;
                Aq.at<float>(k*2,3)=0;
                Aq.at<float>(2*k+1,0)=tmp.at<float>(k,1);
                Aq.at<float>(2*k+1,1)=tmp.at<float>(k,0);
                Aq.at<float>(2*k+1,2)=0;
                Aq.at<float>(2*k+1,3)=1;
            }
            Mat I=Mat::eye(8,8,CV_32FC1);
            Ses[i][j]=Aq*(Aq.t()*Aq).inv()*Aq.t()-I;
            //cout<<sum(Ses[i][j])[0]<<endl;//true
            //cout<<sum(Aq)[0]<<endl;
        }
    }
    /************************************/

    /***************line cut*********************/
    Mat img_gray;
	cvtColor(img,img_gray,CV_BGR2GRAY);
	Mat line_gray=img_gray.clone();
	Mat imgx=img_gray.clone();
	vector<Vec4f>lines;
    Ptr<LineSegmentDetector>ls=createLineSegmentDetector(LSD_REFINE_STD);
    ls->detect(img_gray,lines);
    Mat drawnLines(img_gray);
	ls->drawSegments(drawnLines, lines);
	//imshow("Standard refinement", drawnLines);
	//waitKey(1000);
	int line_num=lines.size();
	int num[y_num][x_num];
	memset(num,0,sizeof(num));
    Mat **lineSeg = new Mat*[y_num-1];
    for (int i = 0; i < y_num-1; i++) lineSeg[i] = new Mat[x_num-1];
    for(int i=0;i<line_num;i++)
    {
        /*line(imgx, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 1, CV_AA);
		imshow("line", imgx);
		waitKey(30);*/
		//cout<<i<<endl;
        Mat aline(2,2,CV_32SC1);
        aline.at<int>(0,1)=lines[i][0];aline.at<int>(0,0)=lines[i][1];
        aline.at<int>(1,1)=lines[i][2];aline.at<int>(1,0)=lines[i][3];
        if((mask.at<uchar>(aline.at<int>(0,0),aline.at<int>(0,1))==1)||(mask.at<uchar>(aline.at<int>(1,0),aline.at<int>(1,1))==1))
            continue;

        int outy1=aline.at<int>(0,0)+disimg.at<Vec2f>(aline.at<int>(0,0),aline.at<int>(0,1))[1];
        int outx1=aline.at<int>(0,1)+disimg.at<Vec2f>(aline.at<int>(0,0),aline.at<int>(0,1))[0];
        //int outx2=aline.at<int>(1,0)+disimg.at<Vec2f>(aline.at<int>(1,0),aline.at<int>(1,1))[0];
        //int outy2=aline.at<int>(1,1)+disimg.at<Vec2f>(aline.at<int>(1,0),aline.at<int>(1,1))[1];
        float gw=(img.cols-1)/(gridcols-1);
        float gh=(img.rows-1)/(gridrows-1);
        int stgrid_y=1.0*outy1/gh;
        int stgrid_x=1.0*outx1/gw;
        //int engrid_y=1.0*outy2/gw;
        //int engrid_x=1.0*outx2/gh;

        int now_x=stgrid_x;
        int now_y=stgrid_y;
        Mat vx(2,2,CV_32FC1);
        Mat vy(2,2,CV_32FC1);
        int dir[4][2]={{1,0},{0,1},{-1,0},{0,-1}};
        Mat pst(1,2,CV_32FC1);
        Mat pen(1,2,CV_32FC1);
        Mat pnow(1,2,CV_32FC1);
        pst.at<float>(0,0)=aline.at<int>(0,0);pst.at<float>(0,1)=aline.at<int>(0,1);
        pen.at<float>(0,0)=aline.at<int>(1,0);pen.at<float>(0,1)=aline.at<int>(1,1);
        pnow=pen.clone();
        int flag;
        Mat p(2,2,CV_32FC1);
        int last=-1,tt=0;
        while(1)
        {
            tt++;
            if(tt>5) break;
            int ok=0;
            //cout<<now_x<<" "<<now_y<<endl;
            if(now_y>=y_num-1||now_x>=x_num-1||now_x<0||now_y<0) break;
            getvertices(now_y,now_x,warp_ygrid,warp_xgrid,vx,vy);
            int isin=checkIsIn(vy,vx,pst.at<float>(0,1),pst.at<float>(0,0),pen.at<float>(0,1),pen.at<float>(0,0));
            if(isin==0)
            {
                Mat quad(2,2,CV_32FC4);
                quad.at<Vec4f>(0,0)[0]=vy.at<float>(0,1);quad.at<Vec4f>(0,1)[0]=vx.at<float>(0,1);
				quad.at<Vec4f>(1,0)[0]=vy.at<float>(1,1);quad.at<Vec4f>(1,1)[0]=vx.at<float>(1,1);
				quad.at<Vec4f>(0,0)[1]=vy.at<float>(1,1);quad.at<Vec4f>(0,1)[1]=vx.at<float>(1,1);
				quad.at<Vec4f>(1,0)[1]=vy.at<float>(1,0);quad.at<Vec4f>(1,1)[1]=vx.at<float>(1,0);
				quad.at<Vec4f>(0,0)[2]=vy.at<float>(1,0);quad.at<Vec4f>(0,1)[2]=vx.at<float>(1,0);
				quad.at<Vec4f>(1,0)[2]=vy.at<float>(0,0);quad.at<Vec4f>(1,1)[2]=vx.at<float>(0,0);
				quad.at<Vec4f>(0,0)[3]=vy.at<float>(0,0);quad.at<Vec4f>(0,1)[3]=vx.at<float>(0,0);
				quad.at<Vec4f>(1,0)[3]=vy.at<float>(0,1);quad.at<Vec4f>(1,1)[3]=vx.at<float>(0,1);//quad,x1y1x2y2
				//cout<<vy<<endl;
				Mat line_now;
				vconcat(pst,pen,line_now);

				//cout<<line_now<<endl;
				/*line(imgx,Point(line_now.at<float>(0,1),line_now.at<float>(0,0)),
                     Point(line_now.at<float>(1,1),line_now.at<float>(1,0)), Scalar(255, 0, 0), 1, CV_AA);
                imshow("line", imgx);
                waitKey(30);*/
                for(int k=0;k<4;k++)
                {
                    if(abs(last-k)==2) continue;
                    Mat quad1(2,2,CV_32FC1);
                    for(int i=0;i<2;i++)
                        for(int j=0;j<2;j++)
                            quad1.at<float>(i,j)=quad.at<Vec4f>(i,j)[k];
                    //cout<<quad1<<endl;
                    /*line(imgx,Point(quad1.at<float>(0,1),quad1.at<float>(0,0)),
                        Point(quad1.at<float>(1,1),quad1.at<float>(1,0)), Scalar(0, 0, 255), 1, CV_AA);
                    imshow("line", imgx);
                    waitKey(30);*/
                    calintersec(quad1,line_now,flag,p);
                    //cout<<flag<<p<<endl;
                    //cout<<quad1<<line_now<<endl;
                    if(flag==1)
                    {
                        //cout<<i<<endl;
                        last=k;
                        ok=1;
                        now_x+=dir[k][0];
                        now_y+=dir[k][1];
                        pnow=p.clone();
                        break;
                    }
                }
            }
            Mat mat_t;
            Mat tt_zero(1,1,CV_32FC1);tt_zero.at<float>(0,0)=0;
            hconcat(pst,pen,mat_t);
            hconcat(mat_t,tt_zero,mat_t);
            if(now_x>x_num||now_y>y_num||now_x<0||now_y<0) break;
            if (num[now_y][now_x]==0)
            {
                lineSeg[now_y][now_x]=mat_t.clone();
                num[now_y][now_x]++;
            }
            else vconcat(lineSeg[now_y][now_x],mat_t,lineSeg[now_y][now_x]);
            pst=pnow.clone();
            pnow=pen.clone();
            //cout<<now_x<<" "<<now_y<<" "<<isin<<" "<<ok<<endl;
            if(isin==1) break;
            if(ok==0) break;
        }
        //cout<<i<<endl;
    }
/*************************************************************/
/****************shape mat***********************************/
    int quadrows=y_num-1;
    int quadcols=x_num-1;
	int quadID;
	int topleftverterID;
	Mat Q = Mat::zeros(8*quadrows*quadcols, 2*y_num*x_num, CV_32FC1);
	for (int i = 0; i < quadrows; i++)
    {
		for (int j = 0; j < quadcols; j++)
        {
			quadID = (i*quadcols + j) * 8;
			topleftverterID = (i*x_num + j) * 2;
			Q.at<float>(quadID, topleftverterID) = 1; Q.at<float>(quadID, topleftverterID + 1) = 0;
			Q.at<float>(quadID + 1, topleftverterID) = 0; Q.at<float>(quadID + 1, topleftverterID+1) = 1;
			Q.at<float>(quadID + 2, topleftverterID + 2) = 1; Q.at<float>(quadID + 2, topleftverterID + 3) = 0;
			Q.at<float>(quadID + 3, topleftverterID + 2) = 0; Q.at<float>(quadID + 3, topleftverterID + 3) = 1;
			Q.at<float>(quadID + 4, topleftverterID + x_num * 2) = 1; Q.at<float>(quadID + 4, topleftverterID + x_num * 2 + 1) = 0;
			Q.at<float>(quadID + 5, topleftverterID + x_num * 2) = 0; Q.at<float>(quadID + 5, topleftverterID + x_num * 2 + 1) = 1;
			Q.at<float>(quadID + 6, topleftverterID + x_num * 2 + 2) = 1; Q.at<float>(quadID + 6, topleftverterID + x_num * 2 + 3) = 0;
			Q.at<float>(quadID + 7, topleftverterID + x_num * 2 + 2) = 0; Q.at<float>(quadID + 7, topleftverterID + x_num * 2 + 3) = 1;
		}
	}
	//cout<<Q.t()*Q;
    Mat S;
    int S_flag=0;
    int Si_flag=0;
    for(int i=0;i<quadrows;i++)
    {
        Mat Si;
        Si_flag=0;
        for (int j=0;j<quadcols;j++)
        {
            if(Si_flag==0)
            {
                Si=Ses[i][j];
                Si_flag++;
            }
            else
            {
                blkdiag(Si,Ses[i][j],Si);
            }
        }
        if(S_flag==0)
        {
            S=Si;
            S_flag++;
        }
        else
        {
            blkdiag(S,Si,S);
        }
    }
    //cout<<sum(S)[0]<<endl;
    /********************************************/
    /************get theta**********************/
    //Mat pst(1,2,CV_32FC1);
    //Mat pen(1,2,CV_32FC1);
    double delta=pi/49;
    vector<double>quad_theta[quadrows][quadcols];
    vector<int>quad_bin[quadrows][quadcols];
    for(int i=0;i<quadrows;i++)
    {
        for(int j=0;j<quadcols;j++)
        {
            Mat quadseg=lineSeg[i][j];
            int lineN=quadseg.rows;
            quad_bin[i][j].clear();
            quad_theta[i][j].clear();
            for(int k=0;k<lineN;k++)
            {
                int pst_x=quadseg.at<float>(k,1);int pst_y=quadseg.at<float>(k,0);
                int pen_x=quadseg.at<float>(k,3);int pen_y=quadseg.at<float>(k,2);
                //line(imgx, Point(pst_y, pst_x), Point(pen_y, pen_x), Scalar(0, 0, 255), 1, CV_AA);
				//imshow("line", imgx);
				//waitKey(30);
				double angle;
				if(pst_x==pen_x) angle=pi/2;
				else angle=atan(double(pst_y-pen_y)/(pst_x-pen_x));
				int theta=(int)((angle+pi/2)/delta);
				quad_theta[i][j].push_back(angle);
				quad_bin[i][j].push_back(theta);
            }
        }
    }
    /*************************************************************/
    /*************boundary mat**********************************/

    int total=x_num*y_num;
    Mat B=Mat::zeros(total*2,1,CV_32FC1);
    Mat BI=Mat::zeros(total*2,1,CV_32FC1);
    for(int i=0;i<total*2;i+=x_num*2)
    {
        B.at<float>(i,0)=1;
        BI.at<float>(i,0)=1;
    }
    for(int i=1;i<x_num*2;i+=2)
    {
        B.at<float>(i,0)=1;
        BI.at<float>(i,0)=1;
    }
    for(int i=x_num*2-2;i<total*2;i+=x_num*2)
    {
        B.at<float>(i,0)=img.cols;
        BI.at<float>(i,0)=1;
    }
    for(int i=total*2-x_num*2+1;i<total*2;i+=2)
    {
        B.at<float>(i,0)=img.rows;
        BI.at<float>(i,0)=1;
    }
    Mat Dg=Mat::diag(BI);
    /**************************************************/

    /*********************optimization loop*********************/
    Mat R(2,2,CV_32FC1);
    Mat pst(2,1,CV_32FC1);
    Mat pen(2,1,CV_32FC1);
    Mat vx(2,2,CV_32FC1);
    Mat vy(2,2,CV_32FC1);
    Mat **Cmatrixes=new Mat*[quadrows];
    for (int i=0;i<quadrows;i++) Cmatrixes[i]=new Mat[quadcols];
    int iterN=1;
    double NL;
    int bad[quadrows][quadcols][110];
    Mat new_xgrid(y_num,x_num,CV_32FC1);
    Mat new_ygrid(y_num,x_num,CV_32FC1);
    for(int it=0;it<iterN;it++)//1mins/iter
    {
        NL=0;
        memset(bad,0,sizeof(bad));
        int Cmatrixes_flag[quadrows+100][quadcols+100];
        vector<Mat>TT[quadrows+100][quadcols+100];
        memset(Cmatrixes_flag,0,sizeof(Cmatrixes_flag));
        /*********************line mat*********************/
        for(int i=0;i<quadrows;i++)
        {
            for(int j=0;j<quadcols;j++)
            {
                int lineN=lineSeg[i][j].rows;
                NL+=lineN;
                for(int k=0;k<lineN;k++)
                {
                    getvertices(i,j,warp_ygrid,warp_xgrid,vx,vy);
                    pst.at<float>(0,0)=lineSeg[i][j].at<float>(k,0);
                    pst.at<float>(1,0)=lineSeg[i][j].at<float>(k,1);
                    pen.at<float>(0,0)=lineSeg[i][j].at<float>(k,2);
                    pen.at<float>(1,0)=lineSeg[i][j].at<float>(k,3);
                    Mat T1,T2;
                    //trans_mat(vx,vy,pst,T1);
                    //trans_mat(vx,vy,pen,T2);
                    int flgg=0;
                    getLinTrans(pst.at<float>(0,0),pst.at<float>(1,0),vy,vx,T1,flgg);
                    getLinTrans(pen.at<float>(0,0),pen.at<float>(1,0),vy,vx,T2,flgg);
                    TT[i][j].push_back(T1);
                    TT[i][j].push_back(T2);
                    double theta=lineSeg[i][j].at<float>(k,4);
                    R.at<float>(0,0)=cos(theta);R.at<float>(0,1)=-sin(theta);
                    R.at<float>(1,0)=sin(theta);R.at<float>(1,1)=cos(theta);
                    Mat e(2,1,CV_32FC1);
                    e.at<float>(0,0)=pen.at<float>(1,0)-pst.at<float>(1,0);
                    e.at<float>(1,0)=pen.at<float>(0,0)-pst.at<float>(0,0);
                    Mat I=Mat::eye(2,2,CV_32FC1);
                    Mat C=(R*e*(e.t()*e).inv()*e.t()*R.t()-I)*(T2-T1);//C*V
                    //cout<<Cmatrixes_flag[i][j];
                    //cout<<C<<endl;
                    if(Cmatrixes_flag[i][j]==0)
                    {
						Cmatrixes[i][j]=C;
						Cmatrixes_flag[i][j]++;
					}
                    else vconcat(Cmatrixes[i][j],C,Cmatrixes[i][j]);
                }
            }
        }
        Mat L;
		int L_flag=0,Li_flag=0;
		int n,m=0;
		for(int i=0;i<quadrows;i++)
        {
			Li_flag=0;
			n=0;
			Mat Li;
			for (int j=0;j<quadcols;j++)
			{
				int lineN=lineSeg[i][j].rows;
				if (lineN==0)
				{
					if(Li_flag!=0)
					{
						Mat x=Mat::zeros(Li.rows,8,CV_32FC1);
						hconcat(Li,x, Li);
					}
					else
					{
						n=n+8;
					}
				}
				else
				{
					if(Li_flag==0)
					{
						if(n!=0)
						{
              				Li=Mat::zeros(Cmatrixes[i][j].rows,n,CV_32FC1);
							hconcat(Li,Cmatrixes[i][j],Li);
						}
						else
						{
							Li=Cmatrixes[i][j].clone();
						}
						Li_flag++;
					}
					else
					{
						blkdiag(Li,Cmatrixes[i][j],Li);
					}
				}
			}
			if (L_flag==0&&Li_flag==0)
			{
				m=m+n;
			}
			else if(L_flag==0&&Li_flag!=0)
			{
				if(m!=0)
				{
					L=Mat::zeros(Li.rows,m,CV_32FC1);
					hconcat(L, Li, L);
				}
				else
				{
					L=Li;
				}
				L_flag++;
			}
			else
			{
				if(Li_flag==0)
				{
					Li=Mat::zeros(L.rows,n,CV_32FC1);
					hconcat(L,Li,L);
				}
				else
				{
					blkdiag(L,Li,L);
				}
			}
		}
		//cout<<Q<<endl;
        //cout<<sum(L)[0]<<endl;
        /***********************************************/
        /*************************update V*************/
        //cout<<"#";
/*
        double N=quadrows*quadcols;
        double lamdaL=0;
        double lamdaB=1e8;
        //cout<<NL<<endl;
        Mat ESC;
        Mat EB;
        //cout<<sum(lamdaB*Dg)[0]<<endl;
        //cout<<sum(1.0/N*S*Q)[0]<<endl;,E
        vconcat(1.0/N*S*Q,1.0*lamdaL/NL*L*Q,ESC);
        vconcat(ESC,lamdaB*Dg,ESC);
        //cout<<sum(ESC)[0]<<endl;
        Mat Z=Mat::zeros(ESC.rows-B.rows,1,CV_32FC1);
        vconcat(Z,lamdaB*B,EB);
        //cout<<sum(EB)[0];
        //cout<<ESC.size();
        //Mat x1=ESC.t()*ESC;
        //cout<<sum(x1)[0]<<endl;
        //cout<<sum(ESC.t()*EB)[0]<<endl;
        //Mat x=(x1.inv())*(ESC.t()*EB);//[1600*1]
        Mat x;
        solve(ESC.t()*ESC,ESC.t()*EB,x,CV_LU);*/
        //cout<<x<<endl;
        double Nq = quadrows*quadcols;
		double lambl = 1;
	    double lambB = 1e8;
		Mat BA;
		MatrixXd S_matrix, Q_matrix, L_matrix,Dg_matrix;
		MatrixXd x1_matrix, x2_matrix, x3_matrix;
		cv2eigen(S, S_matrix);
		cv2eigen(Q, Q_matrix);
		cv2eigen(L, L_matrix);
		cv2eigen(Dg, Dg_matrix);
		SparseMatrix<double> S1=S_matrix.sparseView();
		SparseMatrix<double> Q1=Q_matrix.sparseView();
		SparseMatrix<double> L1=L_matrix.sparseView();
		SparseMatrix<double> Dg1=Dg_matrix.sparseView();
		x1_matrix = (1.0/Nq)*S1*Q1;
	    x2_matrix = (lambl/NL)*L1*Q1;
	    x3_matrix = lambB*Dg1;
		Mat x1, x2, x3;
		eigen2cv(x1_matrix, x1);
		eigen2cv(x2_matrix, x2);
		eigen2cv(x3_matrix, x3);
		Mat K;
		vconcat(x1, x2,K);
		vconcat(K, x3, K);
		//cout<<sum(K)[0];
		cv::vconcat(Mat::zeros(K.rows - B.rows, 1, CV_32FC1), lambB*B, BA);
		//cout<<sum(BA)[0];
		MatrixXd K_matrix,BA_matrix, A_matrix,b_matrix;//
		cv2eigen(K,K_matrix);
		cv2eigen(BA, BA_matrix);
		SparseMatrix<double> K1=K_matrix.sparseView();
		SparseMatrix<double> BA1=BA_matrix.sparseView();
		A_matrix = K1.transpose()*K1;
		b_matrix = K1.transpose()*BA1;
/*
		MatrixXd Ainv_matrix = A_matrix.inverse();
		MatrixXd x_matrix = Ainv_matrix*b_matrix;*/

		//MatrixXd x_matrix=A_matrix.ldlt().solve(b_matrix);

		SparseMatrix<double> A=A_matrix.sparseView();
        SparseLU<SparseMatrix<double>> solver;
        solver.compute(A);
        if (solver.info()!=Success)
        {
            cout << "Compute Matrix is error" << endl;
            return;
        }
        MatrixXd x_matrix = solver.solve(b_matrix);
		cv::Mat x;
		eigen2cv(x_matrix, x);
        int cnt=0;
        for(int i=0;i<y_num;i++)
        {
            for(int j=0;j<x_num;j++)
            {
                new_xgrid.at<float>(i,j)=(int)x.at<double>(cnt,0)-1;
                new_ygrid.at<float>(i,j)=(int)x.at<double>(cnt+1,0)-1;
                cnt+=2;
            }
        }
        //cout<<new_ygrid<<new_xgrid<<endl;

        double bin_num[55];
        double rot_sum[55];
        for(int i=0;i<quadrows;i++)
        {
            for(int j=0;j<quadcols;j++)
            {
                int lineN=lineSeg[i][j].rows;
                getvertices(i,j,new_ygrid,new_xgrid,vx,vy);
                for(int k=0;k<lineN;k++)
                {
                    if(bad[i][j][k]) continue;
                    Mat T1=TT[i][j][k*2];
                    Mat T2=TT[i][j][k*2+1];
                    Mat V(8,1,CV_32FC1);
                    V.at<float>(0,0)=vx.at<float>(0,0);V.at<float>(1,0)=vy.at<float>(0,0);
                    V.at<float>(2,0)=vx.at<float>(0,1);V.at<float>(3,0)=vy.at<float>(0,1);
                    V.at<float>(4,0)=vx.at<float>(1,0);V.at<float>(5,0)=vy.at<float>(1,0);
                    V.at<float>(6,0)=vx.at<float>(1,1);V.at<float>(7,0)=vy.at<float>(1,1);
                    Mat st=T1*V;
                    Mat en=T2*V;
                    double oritheta=quad_theta[i][j][k];
                    double theta=atan((st.at<float>(1,0)-en.at<float>(1,0))/(st.at<float>(0,0)-en.at<float>(0,0)));
                    double delta_theta=theta-oritheta;
                    if(isnan(delta_theta)) continue;
                    if(delta_theta>pi/2) delta_theta-=pi;
                    if(delta_theta<-pi/2) delta_theta+=pi;
                    int bin=quad_bin[i][j][k];
                    //cout<<bin<<" ";
                    bin_num[bin]++;
                    rot_sum[bin]+=delta_theta;
                }
            }
        }
        for(int i=0;i<50;i++)
        {
            if(bin_num[i]==0) rot_sum[i]=0;
            else rot_sum[i]=1.0*rot_sum[i]/bin_num[i];
        }
        for(int i=0;i<quadrows;i++)
        {
            for(int j=0;j<quadcols;j++)
            {
                int lineN=lineSeg[i][j].rows;
                for(int k=0;k<lineN;k++)
                {
                    lineSeg[i][j].at<float>(k,4)=rot_sum[quad_bin[i][j][k]];
                }
            }
        }
        //for(int i=0;i<50;i++) cout<<rot_sum[i]<<endl;
        //cout<<bin_num[49]<<" "<<rot_sum[49]<<endl;
        /*Mat gridmask2,imageGrided2;
        drawGridmask(new_ygrid,new_xgrid,rows,cols,gridmask2);
        drawGrid(gridmask2,img,imageGrided2);
        imshow("imageGrided2",imageGrided2);
        waitKey(0);*/
    }
    /*
    Mat gridmask2,imageGrided2;
    drawGridmask(new_ygrid,new_xgrid,rows,cols,gridmask2);
    drawGrid(gridmask2,img,imageGrided2);
    imshow("imageGrided2",imageGrided2);
    waitKey(1000);*/

    Mat vx1(2,2,CV_32FC1);
    Mat vx2(2,2,CV_32FC1);
    Mat vy1(2,2,CV_32FC1);
    Mat vy2(2,2,CV_32FC1);
    Mat outimg(img.rows,img.cols,CV_32SC3);
    int **cnt=new int*[img.rows];
	for(int i=0;i<img.rows;i++)
		cnt[i]=new int[img.cols];
    for(int i=0;i<img.rows;i++)
        for(int j=0;j<img.cols;j++)
            cnt[i][j]=0;
    double sx=0,sy=0;
    //#pragma omp parallel for
    for(int i=0;i<quadrows;i++)
    {
        for(int j=0;j<quadcols;j++)
        {
            getvertices(i,j,new_ygrid,new_xgrid,vx1,vy1);
            Mat V1(8,1,CV_32FC1);
            V1.at<float>(0,0)=vx1.at<float>(0,0);V1.at<float>(1,0)=vy1.at<float>(0,0);
            V1.at<float>(2,0)=vx1.at<float>(0,1);V1.at<float>(3,0)=vy1.at<float>(0,1);
            V1.at<float>(4,0)=vx1.at<float>(1,0);V1.at<float>(5,0)=vy1.at<float>(1,0);
            V1.at<float>(6,0)=vx1.at<float>(1,1);V1.at<float>(7,0)=vy1.at<float>(1,1);
            getvertices(i,j,warp_ygrid,warp_xgrid,vx2,vy2);
            Mat V2(8,1,CV_32FC1);
            V2.at<float>(0,0)=vx2.at<float>(0,0);V2.at<float>(1,0)=vy2.at<float>(0,0);
            V2.at<float>(2,0)=vx2.at<float>(0,1);V2.at<float>(3,0)=vy2.at<float>(0,1);
            V2.at<float>(4,0)=vx2.at<float>(1,0);V2.at<float>(5,0)=vy2.at<float>(1,0);
            V2.at<float>(6,0)=vx2.at<float>(1,1);V2.at<float>(7,0)=vy2.at<float>(1,1);
            double minx=min(min(V1.at<float>(0,0),V1.at<float>(2,0)),min(V1.at<float>(4,0),V1.at<float>(6,0)));
            double maxx=max(max(V1.at<float>(0,0),V1.at<float>(2,0)),max(V1.at<float>(4,0),V1.at<float>(6,0)));
            double miny=min(min(V1.at<float>(1,0),V1.at<float>(3,0)),min(V1.at<float>(5,0),V1.at<float>(7,0)));
            double maxy=max(max(V1.at<float>(1,0),V1.at<float>(3,0)),max(V1.at<float>(5,0),V1.at<float>(7,0)));
            double lenx=maxx-minx;
            double leny=maxy-miny;
            sx+=1.0*(img.cols-1)/(x_num-1)/lenx;
            sy+=1.0*(img.rows-1)/(y_num-1)/leny;
            double tx=1.0/(2*lenx);
            double ty=1.0/(2*leny);
            for(double y=0;y<1;y+=ty)
            {
                for(double x=0;x<1;x+=tx)
                {
                    #pragma omp parallel for
                    double k1=1-y-x+y*x;
                    double k2=x-y*x;
                    double k3=y-y*x;
                    double k4=y*x;
                    Mat T(2,8,CV_32FC1);
                    T.at<float>(0,0)=k1;T.at<float>(1,0)=0;
					T.at<float>(0,1)=0;T.at<float>(1,1)=k1;
					T.at<float>(0,2)=k2;T.at<float>(1,2)=0;
					T.at<float>(0,3)=0;T.at<float>(1,3)=k2;
					T.at<float>(0,4)=k3;T.at<float>(1,4)=0;
					T.at<float>(0,5)=0;T.at<float>(1,5)=k3;
					T.at<float>(0,6)=k4;T.at<float>(1,6)=0;
					T.at<float>(0,7)=0;T.at<float>(1,7)=k4;
					Mat pout=T*V1;
					Mat ppre=T*V2;
					int x1=(int)pout.at<float>(0,0);
					int y1=(int)pout.at<float>(1,0);
					int x2=(int)ppre.at<float>(0,0);
					int y2=(int)ppre.at<float>(1,0);
					if(y1<0||x1<0||
                       y2<0||x2<0) continue;
                    if(y1>=img.rows||x1>=img.cols||
                       y2>=img.rows||x2>=img.cols) continue;
                    outimg.at<Vec3i>(y1,x1)+=img.at<Vec3b>(y2,x2);
    //cout<<y1<<" "<<x1<<"   "<<y2<<" "<<x2<<endl;
    //if(img.at<Vec3b>(y2,x2)[0]<255) cout<<img.at<Vec3b>(y2,x2)<<endl;
                    cnt[y1][x1]++;
                }
            }
        }
    }
    for(int i=0;i<img.rows;i++)
    {
        for(int j=0;j<img.cols;j++)
        {
            if(cnt[i][j]==0) continue;
            outimg.at<Vec3i>(i,j)[0]/=cnt[i][j];
            outimg.at<Vec3i>(i,j)[1]/=cnt[i][j];
            outimg.at<Vec3i>(i,j)[2]/=cnt[i][j];
            //if(outimg.at<Vec3i>(i,j)[0]<255) cout<<outimg.at<Vec3i>(i,j)<<endl;
        }
    }
    delete(cnt);
    sx/=(1.0*quadcols*quadrows);
    sy/=(1.0*quadcols*quadrows);
    //cout<<sx<<" "<<sy<<endl;
    int coll=(int)(outimg.cols*sx);
    int roww=(int)(outimg.rows*sy);
    outimg.convertTo(outimg,CV_8U);
    resize(outimg,outimg,Size(coll,roww));
    //imshow("final",outimg);
    //waitKey(0);
    output=outimg.clone();
}
