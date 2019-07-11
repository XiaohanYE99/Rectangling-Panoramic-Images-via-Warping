#include<bits/stdc++.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#define INF 1111111111
using namespace std ;
using namespace cv;
void get_energy(Mat& img,Mat& output,Mat& mask)
{

    Mat dx,dy;
    Sobel(img,dx,3,1,0);
    Sobel(img,dy,3,0,1);
    Mat out=Mat::zeros(dx.rows,dx.cols,CV_64FC1);
    for(int i=0;i<img.rows;i++)
        for(int j=0;j<img.cols;j++)
        {
            for(int k=0;k<3;k++)
            {
                out.at<double>(i,j)+=sqrt(dx.at<Vec3b>(i,j)[k]*dx.at<Vec3b>(i,j)[k]
                                          +dy.at<Vec3b>(i,j)[k]*dy.at<Vec3b>(i,j)[k]);
            }

            if(mask.at<uchar>(i,j)!=0) out.at<double>(i,j)+=10000000;
        }
    output=out.clone();
}
void update_energy(Mat& img,Mat& output,Mat& mask,int& st,int& en,int *to)
{
    int W=img.cols;
    int H=img.rows;
    Mat out=output.clone();
    for(int i=st;i<=en;i++)
    {
        for(int j=W-1;j>=to[i]-1&&j>=0;j--)
        {
            if(j>to[i]) out.at<double>(i,j)=out.at<double>(i,j-1);
            else
            {
                Vec3b z={0,0,0};
                Vec3b l = (j>0)?img.at<Vec3b>(i, j-1):z;
				Vec3b r = (j<W-1)?img.at<Vec3b>(i, j+1):z;
				Vec3b u = (i>0)?img.at<Vec3b>(i-1, j):z;
				Vec3b d = (i<H-1)?img.at<Vec3b>(i+1, j):z;
				int val = sqrt((l[0]-r[0])*(l[0]-r[0]) + (l[1]-r[1])*(l[1]-r[1])) + sqrt((l[2]-r[2])*(l[2]-r[2]) +
						(u[0]-d[0])*(u[0]-d[0])) + sqrt((u[1]-d[1])*(u[1]-d[1]) + (u[2]-d[2])*(u[2]-d[2]));
				out.at<double>(i, j) = val;
            }
            if(mask.at<uchar>(i,j)!=0) out.at<double>(i,j)=10000000;
        }
    }
    output=out.clone();
    //cout<<sum(output)[0]<<endl;
}
int *path(Mat& img,int& dir,int& st,int& en)
{
    int H=img.rows;
    int W=img.cols;
    if(dir==2||dir==3)
    {
        int t;
        t=st;
        st=en;
        en=t;
        st=H-st-1;
        en=H-en-1;
    }
    int **dp = new int*[H];
	for (int i = 0; i < H; i++)
		dp[i] = new int[W];
    for(int i=0;i<W;i++) dp[st][i]=(int)img.at<double>(st,i);
    #pragma omp parallel for
    for(int i=st+1;i<=en;i++)
    {
        for(int j=0;j<W;j++)
        {
            if(j==0) dp[i][j]=min(dp[i-1][j],dp[i-1][j+1]);
            else if(j==W-1) dp[i][j]=min(dp[i-1][j],dp[i-1][j-1]);
            else dp[i][j]=min(min(dp[i-1][j-1],dp[i-1][j]),dp[i-1][j+1]);
            dp[i][j]+=(int)img.at<double>(i,j);
        }
    }
    int *to=new int[H];
    int minn=INF,tmp=-1;
    for(int i=0;i<W;i++)
        if(dp[en][i]<minn)
        {
            minn=dp[en][i];
            tmp=i;
        }
    to[en]=tmp;
    Point pos(en,tmp);
    //cout<<minn<<endl;
    while(pos.x>st)
    {
        int x=pos.x;
        int y=pos.y;
        int res=dp[x][y]-(int)img.at<double>(x,y);
        if(y==0)
        {
            if(res==dp[x-1][y]) pos=Point(x-1,y);
            else if(res==dp[x-1][y+1]) pos=Point(x-1,y+1);
            else cout<<"error"<<endl;
        }
        else if(y==W-1)
        {
            if(res==dp[x-1][y]) pos=Point(x-1,y);
            else if(res==dp[x-1][y-1]) pos=Point(x-1,y-1);
            else cout<<"error"<<endl;
        }
        else
        {
            if(res==dp[x-1][y]) pos=Point(x-1,y);
            else if(res==dp[x-1][y+1]) pos=Point(x-1,y+1);
            else if(res==dp[x-1][y-1]) pos=Point(x-1,y-1);
            else cout<<"error"<<endl;
        }
        to[pos.x]=pos.y;
    }
    delete(dp);
    return to;
}
void add_seam(Mat& img,int *to,int dir,Mat& mask,int& st,int& en,Mat& disimg)
{
    int W=img.cols;
    int H=img.rows;
    for(int i=st;i<=en;i++)
    {
        for(int k=0;k<3;k++)
            img.at<Vec3b>(i,to[i])[k]=(img.at<Vec3b>(i,to[i]-1)[k]+img.at<Vec3b>(i,to[i])[k])/2+0.5;
        if(mask.at<uchar>(i,to[i])==0) mask.at<uchar>(i,to[i])=2;
        else if(mask.at<uchar>(i,to[i])==1) mask.at<uchar>(i,to[i])=3;
    }
    for(int i=st;i<=en;i++)
    {
        for(int j=W-1;j>to[i];j--)
        {
            img.at<Vec3b>(i,j)=img.at<Vec3b>(i,j-1);
            mask.at<uchar>(i,j)=mask.at<uchar>(i,j-1);
            Vec2f dis;
            if(dir==1)
            {
                dis[0]=0;dis[1]=1;
            }
            else if(dir==2)
            {
                dis[0]=1;dis[1]=0;
            }
            else if(dir==3)
            {
                dis[0]=0;dis[1]=-1;
            }
            else
            {
                dis[0]=-1;dis[1]=0;
            }
            disimg.at<Vec2f>(i,j)+=dis;
        }
    }
}
void rot(Mat& img,int flag)
{
    if(flag==4)
    {
        transpose(img,img);
        flip(img,img,1);
    }
    if(flag==3)
    {
        flip(img,img,-1);
    }
    if(flag==2)
    {
        transpose(img,img);
        flip(img,img,0);
    }
}
void invrot(Mat& img,int flag)
{
    if(flag==4)
    {
        flip(img,img,1);
        transpose(img,img);
    }
    if(flag==3)
    {
        flip(img,img,-1);
    }
    if(flag==2)
    {
        flip(img,img,0);
        transpose(img,img);
    }
}
void get_len(Mat& bor,int flag,int& len,int& dir,int& st,int& en)
{
    int dif,l=0,r=0;
    if(flag==1||flag==3)
    {
        for(int i=0;i<=bor.rows;i++)
        {
            if(bor.at<uchar>(i,0)==2) bor.at<uchar>(i,0)=1;
            else if(bor.at<uchar>(i,0)==3) bor.at<uchar>(i,0)==1;
            if(i==0) dif=bor.at<uchar>(i,0);
            else if(i==bor.rows) dif=-bor.at<uchar>(i-1,0);
            else dif=bor.at<uchar>(i,0)-bor.at<uchar>(i-1,0);
            if(dif==1) l=i;
            if(dif==-1)
            {
                r=i-1;
                if(r-l+1>len)
                {
                    len=r-l+1;
                    dir=flag;
                    st=l;
                    en=r;
                }
            }
        }
    }
    else
    {
        for(int i=0;i<=bor.cols;i++)
        {
            if(bor.at<uchar>(0,i)==2) bor.at<uchar>(0,i)=1;
            else if(bor.at<uchar>(0,i)==3) bor.at<uchar>(0,i)=1;
            if(i==0) dif=bor.at<uchar>(0,i);
            else if(i==bor.cols) dif=-bor.at<uchar>(0,i-1);
            else dif=bor.at<uchar>(0,i)-bor.at<uchar>(0,i-1);
            if(dif==1) l=i;
            if(dif==-1)
            {
                r=i-1;
                if(r-l+1>len)
                {
                    len=r-l+1;
                    dir=flag;
                    st=l;
                    en=r;
                }
            }
        }
    }
}
void find_dir(Mat& mask,int& dir,int& st,int& en)
{
    int W=mask.cols;
    int H=mask.rows;
    int len=0;
    dir=0;st=0;en=0;
    for(int i=1;i<=4;i++)
    {
        if(i==1)
        {
            Mat bor=mask.col(W-1).clone();
            get_len(bor,1,len,dir,st,en);
        }
        else if(i==2)
        {
            Mat bor=mask.row(H-1).clone();
            get_len(bor,2,len,dir,st,en);
        }
        else if(i==3)
        {
            Mat bor=mask.col(0).clone();
            get_len(bor,3,len,dir,st,en);
        }
        else
        {
            Mat bor=mask.row(0).clone();
            get_len(bor,4,len,dir,st,en);
        }
    }
    if(len<28) dir=0;
    //cout<<len<<endl;
}
void localwrap(Mat& oriimg,Mat& orimask,Mat& disimg,Mat& new_img)
{
    int dir=0,st=0,en=0;
    Mat grad,mask;
    mask=orimask.clone();
    Mat img=oriimg.clone();
    get_energy(img,grad,mask);
    while(1)
    {
        find_dir(mask,dir,st,en);
        if(dir==0) break;
        rot(img,dir);
        rot(grad,dir);
        rot(mask,dir);
        rot(disimg,dir);
        int *to=path(grad,dir,st,en);
        add_seam(img,to,dir,mask,st,en,disimg);
        update_energy(img,grad,mask,st,en,to);
        //get_energy(img,grad,mask);
        delete(to);
        //cout<<dir<<" "<<st<<" "<<en<<endl;
        invrot(img,dir);
        invrot(grad,dir);
        invrot(mask,dir);
        invrot(disimg,dir);
        //imshow("1",img);
        //waitKey(1);

    }
    /*imshow("1",img);
    waitKey(0);
    new_img=img.clone();*/
}



