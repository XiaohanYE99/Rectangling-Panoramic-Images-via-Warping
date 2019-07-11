# Rectangling-Panoramic-Images-via-Warping

运行main.cpp即可将输入的不规则全景图片矩形化。

add_seam.hpp:该程序运用seam carving算法完成图片的初次变形，用动态规划算法找出一小块子图像中能量最小的缝隙进行图片resize，记录每个像素的位移量。
global_warp.hpp:利用seam carving得到的位移量计算原图像的grid mesh，运用warping的算法实现图像全局扭曲。
