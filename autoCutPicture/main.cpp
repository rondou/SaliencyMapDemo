//
//  main.cpp
//  autoCutPicture
//
//  Created by rondou.chen on 2014/10/18.
//  Copyright (c) 2014年 rondou chen. All rights reserved.
// CmCode/CmLib/Basic/CmDefinition.h

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include "cvaux.h"
#include "highgui.h"
#include <stdio.h>
#include <math.h>
#include "PrefixHeader.pch"

#include <stdio.h>
#include <sys/stat.h> //include for stat
#include <string.h>
#include <errno.h>

#include <sstream>

using namespace cv;
using namespace std;

void SmoothSaliency();

typedef unsigned char byte;
typedef vector<int> vecI;
typedef vector<unsigned char> vecB;
typedef vector<float> vecF;
typedef vector<double> vecD;

typedef const Mat CMat;
typedef const SparseMat CSMat;
typedef pair<double, int> CostIdx;
typedef pair<float, int> CostfIdx;
typedef pair<int, int> CostiIdx;
typedef vector<CostIdx> CostIdxV;
typedef vector<CostfIdx> CostfIdxV;
typedef vector<CostiIdx> CostiIdxV;
typedef complex<double> complexD;
typedef complex<float> complexF;
const double EPS = 1e-200;		// Epsilon (zero value)
const double INF = 1e200;

#define ForPoints2(pnt, xS, yS, xE, yE)	for (Point pnt(0, (yS)); pnt.y != (yE); pnt.y++) for (pnt.x = (xS); pnt.x != (xE); pnt.x++)

typedef struct {
    float w;
    int a, b;
}edge;

bool operator<(const edge &a, const edge &b) {
    return a.w < b.w;
}

struct Region{
    Region() { pixNum = 0; ad2c = Point2d(0, 0);}
    int pixNum;  // Number of pixels
    vector<CostfIdx> freIdx;  // Frequency of each color and its index
    Point2d centroid;
    Point2d ad2c; // Average distance to image center
};
template<typename T> inline T sqr(T x) { return x * x; }

// Region Contrast
static Mat GetRC(CMat &img3f);
static Mat GetRC(CMat &img3f, CMat &idx1i, int regNum, double sigmaDist = 0.4);
static Mat GetRC(CMat &img3f, double sigmaDist, double segK, int segMinSize, double segSigma);
static void BuildRegions(CMat& regIdx1i, vector<Region> &regs, CMat &colorIdx1i, int colorNum);
static void RegionContrast(const vector<Region> &regs, CMat &color3fv, Mat& regSal1d, double sigmaDist);
static void SmoothByHist(CMat &img3f, Mat &sal1f, float delta);
static void SmoothByRegion(Mat &sal1f, CMat &idx1i, int regNum, bool bNormalize = true);
static Mat GetBorderReg(CMat &idx1i, int regNum, double ratio = 0.02, double thr = 0.3);
template<class T> inline T pntSqrDist(const Point_<T> &p1, const Point_<T> &p2) {return sqr(p1.x - p2.x) + sqr(p1.y - p2.y);} // out of range risk for T = byte, ...
template<class T> inline double pntDist(const Point_<T> &p1, const Point_<T> &p2) {return sqrt((double)pntSqrDist(p1, p2));}

static Mat GetHC(CMat &img3f);
static void GetHC(CMat &binColor3f, CMat &colorNums1i, Mat &colorSaliency);
const int DefaultNums[3] = {12, 12, 12};
static int Quantize(CMat& img3f, Mat &idx1i, Mat &_color3f, Mat &_colorNum, double ratio = 0.95, const int colorNums[3] = DefaultNums);
static void SmoothSaliency(Mat &sal1f, float delta, const vector<vector<CostfIdx>> &similar);
static void SmoothSaliency(CMat &colorNum1i, Mat &sal1f, float delta, const vector<vector<CostfIdx>> &similar);

//template<class T> inline T pntSqrDist(const Point_<T> &p1, const Point_<T> &p2) {return sqr(p1.x - p2.x) + sqr(p1.y - p2.y);}
template<class T, int D> inline T vecSqrDist(const Vec<T, D> &v1, const Vec<T, D> &v2) {T s = 0; for (int i=0; i<D; i++) s += sqr(v1[i] - v2[i]); return s;}
template<class T, int D> inline T vecDist(const Vec<T, D> &v1, const Vec<T, D> &v2) { return sqrt(vecSqrDist(v1, v2)); }

//Mat origin_image;
//Mat src_gary;

#define THRESHOLD(size, c) (c/size)
universe *segment_graph(int nu_vertices, int nu_edges, edge *edges, float c) {
    // sort edges by weight
    std::sort(edges, edges + nu_edges);
    
    // make a disjoint-set forest
    universe *u = new universe(nu_vertices);
    
    // init thresholds
    float *threshold = new float[nu_vertices];
    for (int i = 0; i < nu_vertices; i++)
        threshold[i] = THRESHOLD(1,c);
    
    // for each edge, in non-decreasing weight order...
    for (int i = 0; i < nu_edges; i++) {
        edge *pedge = &edges[i];
        
        // components conected by this edge
        int a = u->find(pedge->a);
        int b = u->find(pedge->b);
        if (a != b) {
            if ((pedge->w <= threshold[a]) &&
                (pedge->w <= threshold[b])) {
                u->join(a, b);
                a = u->find(a);
                threshold[a] = pedge->w + THRESHOLD(u->size(a), c);
            }
        }
    }
    
    // free up
    delete threshold;
    return u;
}


//string origin_file("/Users/rondouchen/Pictures/summerwar.jpg");//OK
//string origin_file("/Users/rondouchen/Pictures/BenchmarkIMAGES/i3.jpg");
//string origin_file("/Users/rondouchen/Pictures/testjump.jpg");//OK
//string origin_file("/Users/rondouchen/Pictures/hideGirl2.jpg");
//string origin_file("/Users/rondouchen/Pictures/testjump2.jpg");//OK
//string origin_file("/Users/rondouchen/Pictures/rain.jpg");//OK
//string origin_file("/Users/rondouchen/Pictures/rain2.jpg");//OK
//string origin_file("/Users/rondouchen/Pictures/rain3.jpg");//OK
//string origin_file("/Users/rondouchen/Pictures/sleepbuitty.jpg");//OK
//string origin_file("/Users/rondouchen/Pictures/Miyazaki_Hayao.jpg");
//string origin_file("/Users/rondouchen/Pictures/wreckitralph.jpeg");//OK
//string origin_file("/Users/rondouchen/Pictures/test.jpg");//OK



//string origin_file("/Users/rondouchen/Pictures/BroomRiding.jpg");//OK
//string origin_file("/Users/rondouchen/Pictures/HorseRiding.jpg");//OK
//string origin_file("/Users/rondouchen/Pictures/rain4.jpg");//OK
//string origin_file("/Users/rondouchen/Pictures/rain5.jpg");//OK
//string origin_file("/Users/rondouchen/Pictures/wind.jpg");//OK
//string origin_file("/Users/rondouchen/Pictures/wind1.jpg");//OK

//string origin_file("/Users/rondouchen/Pictures/SaintSeiya.jpg");//OK
//string origin_file("/Users/rondouchen/Pictures/SaintSeiya1.jpg");//OK
//string origin_file("/Users/rondouchen/Pictures/SaintSeiya2.jpg");
//string origin_file("/Users/rondouchen/Pictures/moschrid.jpg");
//string origin_file("/Users/rondouchen/Pictures/maho.jpg");



void SmoothSaliency(Mat &sal1f, float delta, const vector<vector<CostfIdx>> &similar)
{
    Mat colorNum1i = Mat::ones(sal1f.size(), CV_32SC1);
    SmoothSaliency(colorNum1i, sal1f, delta, similar);
}

void SmoothSaliency(CMat &colorNum1i, Mat &sal1f, float delta, const vector<vector<CostfIdx>> &similar)
{
    if (sal1f.cols < 2)
        return;
    CV_Assert(sal1f.rows == 1 && sal1f.type() == CV_32FC1);
    CV_Assert(colorNum1i.size() == sal1f.size() && colorNum1i.type() == CV_32SC1);
    
    int binN = sal1f.cols;
    Mat newSal1d= Mat::zeros(1, binN, CV_64FC1);
    float *sal = (float*)(sal1f.data);
    double *newSal = (double*)(newSal1d.data);
    int *pW = (int*)(colorNum1i.data);
    
    // Distance based smooth
    int n = max(cvRound(binN * delta), 2);
    vecD dist(n, 0), val(n), w(n);
    for (int i = 0; i < binN; i++){
        const vector<CostfIdx> &similari = similar[i];
        double totalDist = 0, totoalWeight = 0;
        for (int j = 0; j < n; j++){
            int ithIdx =similari[j].second;
            dist[j] = similari[j].first;
            val[j] = sal[ithIdx];
            w[j] = pW[ithIdx];
            totalDist += dist[j];
            totoalWeight += w[j];
        }
        double valCrnt = 0;
        for (int j = 0; j < n; j++)
            valCrnt += val[j] * (totalDist - dist[j]) * w[j];
        
        newSal[i] =  valCrnt / (totalDist * totoalWeight);
    }
    normalize(newSal1d, sal1f, 0, 1, NORM_MINMAX, CV_32FC1);
}

int Quantize(CMat& img3f, Mat &idx1i, Mat &_color3f, Mat &_colorNum, double ratio, const int clrNums[3])
{
    float clrTmp[3] = {clrNums[0] - 0.0001f, clrNums[1] - 0.0001f, clrNums[2] - 0.0001f};
    int w[3] = {clrNums[1] * clrNums[2], clrNums[2], 1};
    
    CV_Assert(img3f.data != NULL);
    idx1i = Mat::zeros(img3f.size(), CV_32S);
    int rows = img3f.rows, cols = img3f.cols;
    if (img3f.isContinuous() && idx1i.isContinuous()){
        cols *= rows;
        rows = 1;
    }
    
    // Build color pallet
    map<int, int> pallet;
    for (int y = 0; y < rows; y++)
    {
        const float* imgData = img3f.ptr<float>(y);
        int* idx = idx1i.ptr<int>(y);
        for (int x = 0; x < cols; x++, imgData += 3)
        {
            idx[x] = (int)(imgData[0]*clrTmp[0])*w[0] + (int)(imgData[1]*clrTmp[1])*w[1] + (int)(imgData[2]*clrTmp[2]);
            pallet[idx[x]] ++;
        }
    }
    
    // Find significant colors
    int maxNum = 0;
    {
        int count = 0;
        vector<pair<int, int>> num; // (num, color) pairs in num
        num.reserve(pallet.size());
        for (map<int, int>::iterator it = pallet.begin(); it != pallet.end(); it++)
            num.push_back(pair<int, int>(it->second, it->first)); // (color, num) pairs in pallet
        sort(num.begin(), num.end(), std::greater<pair<int, int>>());
        
        maxNum = (int)num.size();
        int maxDropNum = cvRound(rows * cols * (1-ratio));
        for (int crnt = num[maxNum-1].first; crnt < maxDropNum && maxNum > 1; maxNum--)
            crnt += num[maxNum - 2].first;
        maxNum = min(maxNum, 256); // To avoid very rarely case
        if (maxNum <= 10)
            maxNum = min(10, (int)num.size());
        
        pallet.clear();
        for (int i = 0; i < maxNum; i++)
            pallet[num[i].second] = i;
        
        vector<Vec3i> color3i(num.size());
        for (unsigned int i = 0; i < num.size(); i++)
        {
            color3i[i][0] = num[i].second / w[0];
            color3i[i][1] = num[i].second % w[0] / w[1];
            color3i[i][2] = num[i].second % w[1];
        }
        
        for (unsigned int i = maxNum; i < num.size(); i++)
        {
            int simIdx = 0, simVal = INT_MAX;
            for (int j = 0; j < maxNum; j++)
            {
                int d_ij = vecSqrDist<int, 3>(color3i[i], color3i[j]);
                if (d_ij < simVal)
                    simVal = d_ij, simIdx = j;
            }
            pallet[num[i].second] = pallet[num[simIdx].second];
        }
    }
    
    _color3f = Mat::zeros(1, maxNum, CV_32FC3);
    _colorNum = Mat::zeros(_color3f.size(), CV_32S);
    
    Vec3f* color = (Vec3f*)(_color3f.data);
    int* colorNum = (int*)(_colorNum.data);
    for (int y = 0; y < rows; y++) 
    {
        const Vec3f* imgData = img3f.ptr<Vec3f>(y);
        int* idx = idx1i.ptr<int>(y);
        for (int x = 0; x < cols; x++)
        {
            idx[x] = pallet[idx[x]];
            color[idx[x]] += imgData[x];
            colorNum[idx[x]] ++;
        }
    }
    for (int i = 0; i < _color3f.cols; i++)
        color[i] /= (float)colorNum[i];
    
    return _color3f.cols;
}


void GetCmplx(CMat& mag32F, CMat& ang32F, Mat& cmplx32FC2)
{
    CV_Assert(mag32F.type() == CV_32FC1 && ang32F.type() == CV_32FC1 && mag32F.size() == ang32F.size());
    cmplx32FC2.create(mag32F.size(), CV_32FC2);
    for (int y = 0; y < mag32F.rows; y++)
    {
        float* cmpD = cmplx32FC2.ptr<float>(y);
        const float* dataA = ang32F.ptr<float>(y);
        const float* dataM = mag32F.ptr<float>(y);
        for (int x = 0; x < mag32F.cols; x++, cmpD += 2)
        {
            cmpD[0] = dataM[x] * cos(dataA[x]);
            cmpD[1] = dataM[x] * sin(dataA[x]);
        }
    }
}


void AbsAngle(CMat& cmplx32FC2, Mat& mag32FC1, Mat& ang32FC1)
{
    CV_Assert(cmplx32FC2.type() == CV_32FC2);
    mag32FC1.create(cmplx32FC2.size(), CV_32FC1);
    ang32FC1.create(cmplx32FC2.size(), CV_32FC1);
    
    for (int y = 0; y < cmplx32FC2.rows; y++)
    {
        const float* cmpD = cmplx32FC2.ptr<float>(y);
        float* dataA = ang32FC1.ptr<float>(y);
        float* dataM = mag32FC1.ptr<float>(y);
        for (int x = 0; x < cmplx32FC2.cols; x++, cmpD += 2)
        {
            dataA[x] = atan2(cmpD[1], cmpD[0]);
            dataM[x] = sqrt(cmpD[0] * cmpD[0] + cmpD[1] * cmpD[1]);
        }
    }
}

void BuildRegions(CMat& regIdx1i, vector<Region> &regs, CMat &colorIdx1i, int colorNum)
{
    int rows = regIdx1i.rows, cols = regIdx1i.cols, regNum = (int)regs.size();
    double cx = cols/2.0, cy = rows / 2.0;
    Mat_<int> regColorFre1i = Mat_<int>::zeros(regNum, colorNum); // region color frequency
    for (int y = 0; y < rows; y++){
        const int *regIdx = regIdx1i.ptr<int>(y);
        const int *colorIdx = colorIdx1i.ptr<int>(y);
        for (int x = 0; x < cols; x++, regIdx++, colorIdx++){
            Region &reg = regs[*regIdx];
            reg.pixNum ++;
            reg.centroid.x += x;
            reg.centroid.y += y;
            regColorFre1i(*regIdx, *colorIdx)++;
            reg.ad2c += Point2d(abs(x - cx), abs(y - cy));
        }
    }
    
    for (int i = 0; i < regNum; i++){
        Region &reg = regs[i];
        reg.centroid.x /= reg.pixNum * cols;
        reg.centroid.y /= reg.pixNum * rows;
        reg.ad2c.x /= reg.pixNum * cols;
        reg.ad2c.y /= reg.pixNum * rows;
        int *regColorFre = regColorFre1i.ptr<int>(i);
        for (int j = 0; j < colorNum; j++){
            float fre = (float)regColorFre[j]/(float)reg.pixNum;
            if (regColorFre[j] > EPS)
                reg.freIdx.push_back(make_pair(fre, j));
        }
    }
}

void RegionContrast(const vector<Region> &regs, CMat &color3fv, Mat& regSal1d, double sigmaDist)
{
    Mat_<float> cDistCache1f = Mat::zeros(color3fv.cols, color3fv.cols, CV_32F);{
        Vec3f* pColor = (Vec3f*)color3fv.data;
        for(int i = 0; i < cDistCache1f.rows; i++)
            for(int j= i+1; j < cDistCache1f.cols; j++)
                cDistCache1f[i][j] = cDistCache1f[j][i] = vecDist<float, 3>(pColor[i], pColor[j]);
    }
    
    int regNum = (int)regs.size();
    Mat_<double> rDistCache1d = Mat::zeros(regNum, regNum, CV_64F);
    regSal1d = Mat::zeros(1, regNum, CV_64F);
    double* regSal = (double*)regSal1d.data;
    for (int i = 0; i < regNum; i++){
        const Point2d &rc = regs[i].centroid;
        for (int j = 0; j < regNum; j++){
            if(i<j) {
                double dd = 0;
                const vector<CostfIdx> &c1 = regs[i].freIdx, &c2 = regs[j].freIdx;
                for (size_t m = 0; m < c1.size(); m++)
                    for (size_t n = 0; n < c2.size(); n++)
                        dd += cDistCache1f[c1[m].second][c2[n].second] * c1[m].first * c2[n].first;
                rDistCache1d[j][i] = rDistCache1d[i][j] = dd * exp(-pntSqrDist(rc, regs[j].centroid)/sigmaDist);
            }
            regSal[i] += regs[j].pixNum * rDistCache1d[i][j];
        }
        regSal[i] *= exp(-9.0 * (sqr(regs[i].ad2c.x) + sqr(regs[i].ad2c.y)));
    }
}

void SmoothByHist(CMat &img3f, Mat &sal1f, float delta)
{
    //imshow("Before", sal1f); imshow("Src", img3f);
    
    // Quantize colors
    CV_Assert(img3f.size() == sal1f.size() && img3f.type() == CV_32FC3 && sal1f.type() == CV_32FC1);
    Mat idx1i, binColor3f, colorNums1i;
    int binN = Quantize(img3f, idx1i, binColor3f, colorNums1i);
    //CmShow::HistBins(binColor3f, colorNums1i, "Frequency");
    
    // Get initial color saliency
    Mat _colorSal =  Mat::zeros(1, binN, CV_64FC1);
    int rows = img3f.rows, cols = img3f.cols;{
        double* colorSal = (double*)_colorSal.data;
        if (img3f.isContinuous() && sal1f.isContinuous())
            cols *= img3f.rows, rows = 1;
        for (int y = 0; y < rows; y++){
            const int* idx = idx1i.ptr<int>(y);
            const float* initialS = sal1f.ptr<float>(y);
            for (int x = 0; x < cols; x++)
                colorSal[idx[x]] += initialS[x];
        }
        const int *colorNum = (int*)(colorNums1i.data);
        for (int i = 0; i < binN; i++)
            colorSal[i] /= colorNum[i];
        normalize(_colorSal, _colorSal, 0, 1, NORM_MINMAX, CV_32F);
    }
    // Find similar colors & Smooth saliency value for color bins
    vector<vector<CostfIdx>> similar(binN); // Similar color: how similar and their index
    Vec3f* color = (Vec3f*)(binColor3f.data);
    cvtColor(binColor3f, binColor3f, CV_BGR2Lab);
    for (int i = 0; i < binN; i++){
        vector<CostfIdx> &similari = similar[i];
        similari.push_back(make_pair(0.f, i));
        for (int j = 0; j < binN; j++)
            if (i != j)
                similari.push_back(make_pair(vecDist<float, 3>(color[i], color[j]), j));
        sort(similari.begin(), similari.end());
    }
    cvtColor(binColor3f, binColor3f, CV_Lab2BGR);
    //CmShow::HistBins(binColor3f, _colorSal, "BeforeSmooth", true);
    SmoothSaliency(colorNums1i, _colorSal, delta, similar);
    //CmShow::HistBins(binColor3f, _colorSal, "AfterSmooth", true);
    
    // Reassign pixel saliency values
    float* colorSal = (float*)(_colorSal.data);
    for (int y = 0; y < rows; y++){
        const int* idx = idx1i.ptr<int>(y);
        float* resSal = sal1f.ptr<float>(y);
        for (int x = 0; x < cols; x++)
            resSal[x] = colorSal[idx[x]];
    }
    //imshow("After", sal1f);
    //waitKey(0);
}

void SmoothByRegion(Mat &sal1f, CMat &segIdx1i, int regNum, bool bNormalize)
{
    vecD saliecy(regNum, 0);
    vecI counter(regNum, 0);
    for (int y = 0; y < sal1f.rows; y++){
        const int *idx = segIdx1i.ptr<int>(y);
        float *sal = sal1f.ptr<float>(y);
        for (int x = 0; x < sal1f.cols; x++){
            saliecy[idx[x]] += sal[x];
            counter[idx[x]] ++;
        }
    }
    
    for (size_t i = 0; i < counter.size(); i++)
        saliecy[i] /= counter[i];
    Mat rSal(1, regNum, CV_64FC1, &saliecy[0]);
    if (bNormalize)
        normalize(rSal, rSal, 0, 1, NORM_MINMAX);
    
    for (int y = 0; y < sal1f.rows; y++){
        const int *idx = segIdx1i.ptr<int>(y);
        float *sal = sal1f.ptr<float>(y);
        for (int x = 0; x < sal1f.cols; x++)
            sal[x] = (float)saliecy[idx[x]];
    }	
}

Mat GetBorderReg(CMat &idx1i, int regNum, double ratio, double thr)
{
    // Variance of x and y
    vecD vX(regNum), vY(regNum);
    int w = idx1i.cols, h = idx1i.rows;{
        vecD mX(regNum), mY(regNum), n(regNum); // Mean value of x and y, pixel number of region
        for (int y = 0; y < idx1i.rows; y++){
            const int *idx = idx1i.ptr<int>(y);
            for (int x = 0; x < idx1i.cols; x++, idx++)
                mX[*idx] += x, mY[*idx] += y, n[*idx]++;
        }
        for (int i = 0; i < regNum; i++)
            mX[i] /= n[i], mY[i] /= n[i];
        for (int y = 0; y < idx1i.rows; y++){
            const int *idx = idx1i.ptr<int>(y);
            for (int x = 0; x < idx1i.cols; x++, idx++)
                vX[*idx] += abs(x - mX[*idx]), vY[*idx] += abs(y - mY[*idx]);
        }
        for (int i = 0; i < regNum; i++)
            vX[i] = vX[i]/n[i] + EPS, vY[i] = vY[i]/n[i] + EPS;
    }
    
    // Number of border pixels in x and y border region
    vecI xbNum(regNum), ybNum(regNum);
    int wGap = cvRound(w * ratio), hGap = cvRound(h * ratio);
    
    vector<Point> bPnts; {
        ForPoints2(pnt, 0, 0, w, hGap) // Top region
        ybNum[idx1i.at<int>(pnt)]++, bPnts.push_back(pnt);
        ForPoints2(pnt, 0, h - hGap, w, h) // Bottom region
        ybNum[idx1i.at<int>(pnt)]++, bPnts.push_back(pnt);
        ForPoints2(pnt, 0, 0, wGap, h) // Left region
        xbNum[idx1i.at<int>(pnt)]++, bPnts.push_back(pnt);
        ForPoints2(pnt, w - wGap, 0, w, h)
        xbNum[idx1i.at<int>(pnt)]++, bPnts.push_back(pnt);
    }
    
    Mat bReg1u(idx1i.size(), CV_8U);{  // likelihood map of border region
        double xR = 1.0/(4*hGap), yR = 1.0/(4*wGap);
        vector<unsigned char> regL(regNum); // likelihood of each region belongs to border background
        for (int i = 0; i < regNum; i++) {
            double lk = xbNum[i] * xR / vY[i] + ybNum[i] * yR / vX[i];
            regL[i] = lk/thr > 1 ? 255 : 0; //saturate_cast<byte>(255 * lk / thr);
        }
        
        for (int r = 0; r < h; r++)	{
            const int *idx = idx1i.ptr<int>(r);
            unsigned char* maskData = bReg1u.ptr<byte>(r);
            for (int c = 0; c < w; c++, idx++)
                maskData[c] = regL[*idx];
        }
    }
    
    for (size_t i = 0; i < bPnts.size(); i++)
        bReg1u.at<byte>(bPnts[i]) = 255;
    return bReg1u;
}

Mat GetRC(CMat &img3f)
{
    return GetRC(img3f, 0.4, 50, 200, 0.5);
}

Mat GetRC(CMat &img3f, CMat &regIdx1i, int regNum, double sigmaDist)
{
    Mat colorIdx1i, regSal1v, tmp, color3fv;
    int QuatizeNum = Quantize(img3f, colorIdx1i, color3fv, tmp);
    if (QuatizeNum == 2){
        printf("QuatizeNum == 2, %d: %s\n", __LINE__, __FILE__);
        Mat sal;
        compare(colorIdx1i, 1, sal, CMP_EQ);
        sal.convertTo(sal, CV_32F, 1.0/255);
        return sal;
    }
    if (QuatizeNum <= 2) // Color quantization
        return Mat::zeros(img3f.size(), CV_32F);
    
    cvtColor(color3fv, color3fv, CV_BGR2Lab);
    vector<Region> regs(regNum);
    BuildRegions(regIdx1i, regs, colorIdx1i, color3fv.cols);
    RegionContrast(regs, color3fv, regSal1v, sigmaDist);
    
    Mat sal1f = Mat::zeros(img3f.size(), CV_32F);
    cv::normalize(regSal1v, regSal1v, 0, 1, NORM_MINMAX, CV_32F);
    float* regSal = (float*)regSal1v.data;
    for (int r = 0; r < img3f.rows; r++){
        const int* regIdx = regIdx1i.ptr<int>(r);
        float* sal = sal1f.ptr<float>(r);
        for (int c = 0; c < img3f.cols; c++)
            sal[c] = regSal[regIdx[c]];
    }
    
    Mat bdReg1u = GetBorderReg(regIdx1i, regNum, 0.02, 0.4);
    sal1f.setTo(0, bdReg1u);
    SmoothByHist(img3f, sal1f, 0.1f);
    SmoothByRegion(sal1f, regIdx1i, regNum);
    sal1f.setTo(0, bdReg1u);
    
    GaussianBlur(sal1f, sal1f, Size(3, 3), 0);
    return sal1f;
}

static inline float diff(CMat &img3f, int x1, int y1, int x2, int y2)
{
    const Vec3f &p1 = img3f.at<Vec3f>(y1, x1);
    const Vec3f &p2 = img3f.at<Vec3f>(y2, x2);
    return sqrt(sqr(p1[0] - p2[0]) + sqr(p1[1] - p2[1]) + sqr(p1[2] - p2[2]));
}

int SegmentImage(CMat &_src3f, Mat &pImgInd, double sigma, double c, int min_size)
{
    CV_Assert(_src3f.type() == CV_32FC3);
    int width(_src3f.cols), height(_src3f.rows);
    Mat smImg3f;
    GaussianBlur(_src3f, smImg3f, Size(), sigma, 0, BORDER_REPLICATE);
    
    // build graph
    edge *edges = new edge[width*height*4];
    int num = 0;
    {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (x < width-1) {
                    edges[num].a = y * width + x;
                    edges[num].b = y * width + (x+1);
                    edges[num].w = diff(smImg3f, x, y, x+1, y);
                    num++;
                }
                
                if (y < height-1) {
                    edges[num].a = y * width + x;
                    edges[num].b = (y+1) * width + x;
                    edges[num].w = diff(smImg3f, x, y, x, y+1);
                    num++;
                }
                
                if ((x < width-1) && (y < height-1)) {
                    edges[num].a = y * width + x;
                    edges[num].b = (y+1) * width + (x+1);
                    edges[num].w = diff(smImg3f, x, y, x+1, y+1);
                    num++;
                }
                
                if ((x < width-1) && (y > 0)) {
                    edges[num].a = y * width + x;
                    edges[num].b = (y-1) * width + (x+1);
                    edges[num].w = diff(smImg3f, x, y, x+1, y-1);
                    num++;
                }
            }
        }
    }
    
    // segment
    universe *u = segment_graph(width*height, num, edges, (float)c);
    
    // post process small components
    for (int i = 0; i < num; i++) {
        int a = u->find(edges[i].a);
        int b = u->find(edges[i].b);
        if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
            u->join(a, b);
    }
    delete [] edges;
    
    // pick random colors for each component
    map<int, int> marker;
    pImgInd.create(smImg3f.size(), CV_32S);
    
    int idxNum = 0;
    for (int y = 0; y < height; y++) {
        int *imgIdx = pImgInd.ptr<int>(y);
        for (int x = 0; x < width; x++) {
            int comp = u->find(y * width + x);
            if (marker.find(comp) == marker.end())
                marker[comp] = idxNum++;
            
            int idx = marker[comp];
            imgIdx[x] = idx;
        }
    }  
    delete u;
    
    return idxNum;
}

Mat GetRC(CMat &img3f, double sigmaDist, double segK, int segMinSize, double segSigma)
{
    Mat imgLab3f, regIdx1i;
    cvtColor(img3f, imgLab3f, CV_BGR2Lab);
    int regNum = SegmentImage(imgLab3f, regIdx1i, segSigma, segK, segMinSize);
    return GetRC(img3f, regIdx1i, regNum, sigmaDist);
}

Mat GetHC(CMat &img3f)
{
    // Quantize colors and
    Mat idx1i, binColor3f, colorNums1i, _colorSal;
    Quantize(img3f, idx1i, binColor3f, colorNums1i);
    cvtColor(binColor3f, binColor3f, CV_BGR2Lab);
    
    GetHC(binColor3f, colorNums1i, _colorSal);
    float* colorSal = (float*)(_colorSal.data);
    Mat salHC1f(img3f.size(), CV_32F);
    for (int r = 0; r < img3f.rows; r++){
        float* salV = salHC1f.ptr<float>(r);
        int* _idx = idx1i.ptr<int>(r);
        for (int c = 0; c < img3f.cols; c++)
            salV[c] = colorSal[_idx[c]];
    }
    GaussianBlur(salHC1f, salHC1f, Size(3, 3), 0);
    normalize(salHC1f, salHC1f, 0, 1, NORM_MINMAX);
    return salHC1f;
}


void GetHC(CMat &binColor3f, CMat &colorNums1i, Mat &_colorSal)
{
    Mat weight1f;
    normalize(colorNums1i, weight1f, 1, 0, NORM_L1, CV_32F);
    
    int binN = binColor3f.cols;
    _colorSal = Mat::zeros(1, binN, CV_32F);
    float* colorSal = (float*)(_colorSal.data);
    vector<vector<CostfIdx>> similar(binN); // Similar color: how similar and their index
    Vec3f* color = (Vec3f*)(binColor3f.data);
    float *w = (float*)(weight1f.data);
    for (int i = 0; i < binN; i++){
        vector<CostfIdx> &similari = similar[i];
        similari.push_back(make_pair(0.f, i));
        for (int j = 0; j < binN; j++){
            if (i == j)
                continue;
            float dij = vecDist<float, 3>(color[i], color[j]);
            similari.push_back(make_pair(dij, j));
            colorSal[i] += w[j] * dij;
        }
        sort(similari.begin(), similari.end());
    }
    
    SmoothSaliency(_colorSal, 0.25f, similar);
}


Mat GetFT(CMat &img3f)
{
    //CV_Assert(img3f.data != NULL && img3f.type() == CV_32FC3);
    Mat sal(img3f.size(), CV_32F), tImg;
    GaussianBlur(img3f, tImg, Size(3, 3), 0);
    cvtColor(tImg, tImg, CV_BGR2Lab);
    Scalar colorM = mean(tImg);
    for (int r = 0; r < tImg.rows; r++) {
        float *s = sal.ptr<float>(r);
        float *lab = tImg.ptr<float>(r);
        for (int c = 0; c < tImg.cols; c++, lab += 3)
            s[c] = (float)(sqr(colorM[0] - lab[0]) + sqr(colorM[1] - lab[1]) + sqr(colorM[2] - lab[2]));
    }
    normalize(sal, sal, 0, 1, NORM_MINMAX);
    return sal;
}

Mat GetSR(CMat &img3f)
{
    Size sz(64, 64);
    Mat img1f[2], sr1f, cmplxSrc2f, cmplxDst2f;
    cvtColor(img3f, img1f[1], CV_BGR2GRAY);
    resize(img1f[1], img1f[0], sz, 0, 0, CV_INTER_AREA);
    
    img1f[1] = Mat::zeros(sz, CV_32F);
    merge(img1f, 2, cmplxSrc2f);
    dft(cmplxSrc2f, cmplxDst2f);
    AbsAngle(cmplxDst2f, img1f[0], img1f[1]);
    
    log(img1f[0], img1f[0]);
    blur(img1f[0], sr1f, Size(3, 3));
    sr1f = img1f[0] - sr1f;
    
    exp(sr1f, sr1f);
    GetCmplx(sr1f, img1f[1], cmplxDst2f);
    dft(cmplxDst2f, cmplxSrc2f, DFT_INVERSE | DFT_SCALE);
    split(cmplxSrc2f, img1f);
    
    pow(img1f[0], 2, img1f[0]);
    pow(img1f[1], 2, img1f[1]);
    img1f[0] += img1f[1];
    
    GaussianBlur(img1f[0], img1f[0], Size(3, 3), 0);
    normalize(img1f[0], img1f[0], 0, 1, NORM_MINMAX);
    resize(img1f[0], img1f[1], img3f.size(), 0, 0, INTER_CUBIC);
    
    return img1f[1];
}

Mat GetLC(CMat &img3f)
{
    Mat img;
    cvtColor(img3f, img, CV_BGR2GRAY);
    img.convertTo(img, CV_8U, 255);
    double f[256], s[256];
    memset(f, 0, 256*sizeof(double));
    memset(s, 0, 256*sizeof(double));
    for (int r = 0; r < img.rows; r++){
        //byte* data;
        unsigned char* data = img.ptr<unsigned char>(r);
        for (int c = 0; c < img.cols; c++)
            f[data[c]] += 1;
    }
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 256; j++)
            s[i] += abs(i - j) * f[j];
    Mat sal1f(img3f.size(), CV_64F);
    for (int r = 0; r < img.rows; r++){
        unsigned char* data = img.ptr<unsigned char>(r);
        double* sal = sal1f.ptr<double>(r);
        for (int c = 0; c < img.cols; c++)
            sal[c] = s[data[c]];
    }
    normalize(sal1f, sal1f, 0, 1, NORM_MINMAX, CV_32F);
    return sal1f;
}

string int2str(int &i) {
    string s;
    stringstream ss(s);
    ss << i;
    
    return ss.str();
}

int main(int argc, const char * argv[])
{
    // insert code here...
    for (int i = 6; i<=297; i++) {
        string s;
        s = int2str(i);
        std::string q = "/Users/rondouchen/Pictures/BenchmarkIMAGES/i" + s;
    string origin_file("/Users/rondouchen/Pictures/BenchmarkIMAGES/i" + s + ".jpg");
    Mat origin_image = imread(origin_file, 21); // resoure image
    //cv::imshow("show_origin", origin_image);
    //cvtColor(origin_image, src_gray, CV_BGR2GRAY); /// 轉灰
    //cvtColor(origin_image, src_ycbcr, CV_RGB2YCrCb, 0); /// TO YcBcR
    //cvtColor(origin_image, src_ycbcr, CV_RGB2YCrCb); /// TO YcBcR
    //cvtColor(origin_image, src_lab, CV_RGB2Lab);
    
    //Mat sal_FT;
    //Mat sal_SR;
    //Mat sal_LC;
    //Mat sal_HC;
    //Mat sal_RC;
    Mat origin_image_CV_32FC3;

    mkdir(q.c_str(), S_IRWXU);
    
    origin_image.convertTo(origin_image_CV_32FC3, CV_32FC3, 1/255.0);
    Mat sal_FT = GetFT(origin_image_CV_32FC3);
    //imshow("getFT", sal_FT);
    imwrite( "/Users/rondouchen/Pictures/BenchmarkIMAGES/i" + s + "/sal_FT.jpg", sal_FT*255);
    
    Mat sal_SR = GetSR(origin_image_CV_32FC3);
    //imshow("getSR", sal_SR);
    imwrite( "/Users/rondouchen/Pictures/BenchmarkIMAGES/i" + s + "/sal_SR.jpg", sal_SR*255);
    
    Mat sal_LC = GetLC(origin_image_CV_32FC3);
    //imshow("getLC", sal_LC);
    imwrite( "/Users/rondouchen/Pictures/BenchmarkIMAGES/i" + s + "/sal_LC.jpg", sal_LC*255);
    
    Mat sal_HC = GetHC(origin_image_CV_32FC3);
    //imshow("getHC", sal_HC);
    imwrite( "/Users/rondouchen/Pictures/BenchmarkIMAGES/i" + s + "/sal_HC.jpg", sal_HC*255);
    
    Mat sal_RC = GetRC(origin_image_CV_32FC3);
    //imshow("getRC", sal_RC);
    imwrite( "/Users/rondouchen/Pictures/BenchmarkIMAGES/i" + s + "/sal_RC.jpg", sal_RC*255);
    }
    //cv::waitKey(0);
}

