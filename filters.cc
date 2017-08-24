//
// Image processing filters
//

#include "spt.h"

// Compute the laplacian filter of image src into dst.
//
// src -- source image
// dst -- destination image (output)
//
void
laplacian(const Mat &src, Mat &dst)
{
	 Mat kern = (Mat_<double>(3,3) <<
	 	0,     1/4.0,  0,
		1/4.0, -4/4.0, 1/4.0,
		0,     1/4.0,  0);
	filter2D(src, dst, -1, kern);
}

void
maxnorm3(const Mat &_src1, const Mat &_src2, const Mat &_src3, Mat &_dst)
{
	CHECKMAT(_src1, CV_32FC1);
	CHECKMAT(_src2, CV_32FC1);
	CHECKMAT(_src3, CV_32FC1);
	CV_Assert(_src1.rows == _src2.rows && _src1.rows == _src3.rows);
	CV_Assert(_src1.cols == _src2.cols && _src1.cols == _src3.cols);
	_dst.create(_src1.size(), CV_32FC1);
	
	auto src1 = _src1.ptr<float>(0);
	auto src2 = _src2.ptr<float>(0);
	auto src3 = _src3.ptr<float>(0);
	auto dst = _dst.ptr<float>(0);
 
	const double E[3][3] = {
		{-7.05298894649560, 3.21224182031700, 3.23523481656600},
		{3.21224182031700, -6.49865666437750, 3.52640569975500},
		{3.23523481656600, 3.52640569975500, -6.44835254151000},
	};
	for(size_t i = 0; i < _src1.total(); i++){
		double r1 = fabs(E[0][0]*src1[i] + E[0][1]*src2[i] + E[0][2]*src3[i]);
		double r2 = fabs(E[1][0]*src1[i] + E[1][1]*src2[i] + E[1][2]*src3[i]);
		double r3 = fabs(E[2][0]*src1[i] + E[2][1]*src2[i] + E[2][2]*src3[i]);
		dst[i] = max(r1, max(r2, r3));
	}
}


// Separable blur implementation that can handle images containing NaN.
// OpenCV blur does not correctly handle such images.
//
// src -- source image
// dst -- destination image (output)
// ksize -- kernel size
//
void
nanblur(const Mat &src, Mat &dst, int ksize)
{
	Mat kernel = Mat::ones(ksize, 1, CV_64FC1)/ksize;
	sepFilter2D(src, dst, -1, kernel, kernel);
}

// Compute the gradient magnitude of image src into dst.
//
// src -- source image
// dst -- destination image of gradient magnitude (output)
// dX, dY -- derivative in the X and Y directions (output)
//
void
gradientmag(const Mat &src, Mat &dst, Mat1f &dX, Mat1f &dY)
{
	partial_derivative(src, dX, dY);
	sqrt(dX.mul(dX) + dY.mul(dY), dst);
}

// Compute the gradient magnitude of image src into dst.
//
// src -- source image
// dst -- destination image of gradient magnitude (output)
//
void
gradientmag(const Mat &src, Mat &dst)
{
	Mat1f dX, dY;
	partial_derivative(src, dX, dY);
	sqrt(dX.mul(dX) + dY.mul(dY), dst);
}


void
partial_derivative(const Mat &src, Mat1f &dX, Mat1f &dY)
{
	Mat h = (Mat_<double>(5,1) <<
		0.036420, 0.248972, 0.429217, 0.248972, 0.036420);
	Mat hp = (Mat_<double>(5,1) <<
		0.108415, 0.280353, 0, -0.280353, -0.108415);

	dX.create(src.size());
	dY.create(src.size());
	sepFilter2D(src, dX, -1, h, hp);
	// We negate h here to fix the sign of Dy
	sepFilter2D(src, dY, -1, hp, -h);
}


void
laplace_operator(const Mat &dX,const Mat1f &dY, Mat1f &dst, Mat1f &dXX, Mat1f &dYY)
{
	Mat h = (Mat_<double>(5,1) <<
		0.036420, 0.248972, 0.429217, 0.248972, 0.036420);
	Mat hp = (Mat_<double>(5,1) <<
		0.108415, 0.280353, 0, -0.280353, -0.108415);

	dXX.create(dX.size());
	dYY.create(dX.size());
	sepFilter2D(dX, dXX, -1, h, hp);
	// We negate h here to fix the sign of Dy
	sepFilter2D(dY, dYY, -1, hp, -h);
	sqrt(dXX.mul(dXX) + dYY.mul(dYY), dst);
}

void
laplace_operator2(const Mat1f &dXX,const Mat1f &dYY, Mat &dst)
{
	sqrt(dXX.mul(dXX) + dYY.mul(dYY), dst);
}

void
meangradientmag(const Mat1f &dx, const Mat1f &dy, int ksize, Mat1f &dst)
{
	Mat dxblur, dyblur;
	nanblur(dx, dxblur, ksize);
	nanblur(dy, dyblur, ksize);
	sqrt(dxblur.mul(dxblur) + dyblur.mul(dyblur), dst);
}

// Find local maximum.
// Prototype: matlab/localmax.m
//
// sstmag -- SST gradient magnitude
// high, low -- (output)
// sigma -- standard deviation
//
void
localmax(const Mat &sstmag, Mat &high, double sigma, bool debug)
{
	Mat low;
	
	enum {
		NStd = 4,
	};
	int i, j, x, y, winsz;
	double e, a, dd, mu1, mu2;
	Mat DGaussxx, DGaussxy, DGaussyy,
		Dxx, Dxy, Dyy;
	
	CHECKMAT(sstmag, CV_32FC1);

	const int width = ceil(NStd*sigma);
	winsz = 2*width + 1;
	DGaussxx = Mat::zeros(winsz, winsz, CV_64FC1);
	DGaussxy = Mat::zeros(winsz, winsz, CV_64FC1);
	DGaussyy = Mat::zeros(winsz, winsz, CV_64FC1);

	for(i = 0; i < winsz; i++){
		x = i - width;
		for(j = 0; j < winsz; j++){
			y = j - width;

			e = exp(-(SQ(x) + SQ(y)) / (2.0*SQ(sigma)));
			DGaussxx.at<double>(i, j) =
				DGaussyy.at<double>(j, i) =
				1/(2*M_PI*pow(sigma, 4)) *
				(SQ(x)/SQ(sigma) - 1) * e;
			DGaussxy.at<double>(i, j) =
				1/(2*M_PI*pow(sigma, 6)) * (x*y) * e;
		}
	}
if(debug){
	SAVENC(DGaussxx);
	SAVENC(DGaussxy);
	SAVENC(DGaussyy);
}
	filter2D(sstmag, Dxx, -1, DGaussxx);
	filter2D(sstmag, Dxy, -1, DGaussxy);
	filter2D(sstmag, Dyy, -1, DGaussyy);
if(debug){
	SAVENC(Dxx);
	SAVENC(Dxy);
	SAVENC(Dyy);
}	

	CHECKMAT(Dxx, CV_32FC1);
	CHECKMAT(Dxy, CV_32FC1);
	CHECKMAT(Dyy, CV_32FC1);
	
	high.create(Dxx.rows, Dxx.cols, CV_32FC1);
	low.create(Dxx.rows, Dxx.cols, CV_32FC1);
	auto highp = high.ptr<float>(0);
	auto lowp = low.ptr<float>(0);
	auto Dxxp = Dxx.ptr<float>(0);
	auto Dxyp = Dxy.ptr<float>(0);
	auto Dyyp = Dyy.ptr<float>(0);
	for(i = 0; i < Dxx.rows*Dxx.cols; i++){
		a = Dxxp[i] + Dyyp[i];
		dd = sqrt(SQ(Dxxp[i] - Dyyp[i]) + 4*SQ(Dxyp[i]));
		mu1 = 0.5*(a + dd);
		mu2 = 0.5*(a - dd);
		if(fabs(mu1) > fabs(mu2)){
			highp[i] = mu1;
			lowp[i] = mu2;
		}else{
			highp[i] = mu2;
			lowp[i] = mu1;
		}
	}
}

// Find local maximum.
// Prototype: matlab/localmax.m
//
// sstmag -- SST gradient magnitude
// high, low -- (output)
// sigma -- standard deviation
//
void
localmax2(Mat &mu2,const Mat &Dxx,const Mat &Dyy, const Mat &Dxy, bool debug)
{
	
	int i, j;
	double a, dd;
	
	CHECKMAT(Dxx, CV_32FC1);
	CHECKMAT(Dxy, CV_32FC1);
	CHECKMAT(Dyy, CV_32FC1);
	
	mu2.create(Dxx.rows, Dxx.cols, CV_32FC1);
	
	auto mu2p = mu2.ptr<float>(0);
	auto Dxxp = Dxx.ptr<float>(0);
	auto Dxyp = Dxy.ptr<float>(0);
	auto Dyyp = Dyy.ptr<float>(0);

	for(i = 0; i < Dxx.rows*Dxx.cols; i++){
		a = Dxxp[i] + Dyyp[i];
		dd = sqrt(SQ(Dxxp[i] - Dyyp[i]) + 4*SQ(Dxyp[i])); // SQ(x) = x^2
		/*
		mu1 = 0.5*(a + dd);
		mu2 = 0.5*(a - dd);
		*/
		mu2p[i] =  0.5*(a - dd);
		/*
		if(fabs(mu1) > fabs(mu2)){
			highp[i] = mu1;
		}else{
			highp[i] = mu2;
		}
		*/
	}
}

// Standard deviation filter, implemented as
//	dst = sqrt(blur(src^2) - blur(src)^2)
//
// src -- source image
// dst -- destination image (output)
// ksize -- kernel size
//
void
stdfilter(const Mat &src, Mat &dst, int ksize)
{
	Mat b, bs, _tmp;
	
	nanblur(src.mul(src), bs, ksize);
	nanblur(src, b, ksize);

	// avoid sqrt of nagative number
	_tmp = bs - b.mul(b);
	CHECKMAT(_tmp, CV_32FC1);
	auto tmp = _tmp.ptr<float>(0);
	for(size_t i = 0; i < _tmp.total(); i++){
		if(tmp[i] < 0){
			tmp[i] = 0;
		}
	}
	sqrt(_tmp, dst);
}

void
stdfilter1(const Mat &src, Mat &dst, int ksize)
{
	CHECKMAT(src, CV_32FC1);
	if(ksize < 0){
		abort();
	}
	
	dst = Mat::zeros(src.size(), CV_32FC1);
	
	for(int y = 0; y < src.rows-ksize+1; y++){
		for(int x = 0; x < src.cols-ksize+1; x++){
			// Two-pass algorithm is more numerically stable:
			// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass_algorithm
			double avg1 = 0;
			int n = 0;
			for(int i = y; i < y+ksize; i++){
				for(int j = x; j < x+ksize; j++){
					double p = src.at<float>(i, j);
					if(!std::isnan(p)){
						avg1 += p;
						n++;
					}
				}
			}
			if(n == 0){
				dst.at<float>(y, x) = 0;
				continue;
			}
			avg1 /= n;
			
			double avg2 = 0;
			for(int i = y; i < y+ksize; i++){
				for(int j = x; j < x+ksize; j++){
					double p = src.at<float>(i, j);
					if(!std::isnan(p)){
						avg2 += SQ(p - avg1);
					}
				}
			}
			avg2 /= n;
			
			dst.at<float>(y+ksize/2, x+ksize/2) = sqrt(avg2);
		}
	}
}

void
atan2mat(const Mat &_X, const Mat &_Y, Mat &_dst)
{
	CHECKMAT(_X, CV_32FC1);
	CHECKMAT(_Y, CV_32FC1);
	CV_Assert(_X.rows == _Y.rows && _X.cols == _Y.cols);
	
	_dst = Mat::zeros(_X.size(), CV_32FC1);
	
	auto X = _X.ptr<float>(0);
	auto Y = _Y.ptr<float>(0);
	auto dst = _dst.ptr<float>(0);
	
	for(size_t i = 0; i < _X.total(); i++){
		dst[i] = atan2(X[i], Y[i]);
	}
}

void
cosmat(const Mat &_src, Mat &_dst)
{
	CHECKMAT(_src, CV_32FC1);
	
	_dst = Mat::zeros(_src.size(), CV_32FC1);
	
	auto src = _src.ptr<float>(0);
	auto dst = _dst.ptr<float>(0);
	
	for(size_t i = 0; i < _src.total(); i++){
		dst[i] = cos(src[i]);
	}
}


// Range filter.
//
// src -- source image
// dst -- destination image (output)
// ksize -- kernel size
//
void
rangefilter(const Mat &src, Mat &dst, int ksize)
{
	Mat min, max;
	
	Mat elem = getStructuringElement(MORPH_RECT, Size(ksize, ksize));
	erode(src, min, elem);
	dilate(src, max, elem);
	dst = max - min;
}

// Compute Laplacian of Gaussian kernel.
// See http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
// This function is equivalent to fspecial('log', n, sigma) in MATLAB.
//
// n -- width/height of kernel
// sigma -- standard deviation of the Gaussian
// dst -- the kernel (output)
//
void
logkernel(int n, double sigma, Mat &dst)
{
	dst.create(n, n, CV_64FC1);
	int h = n/2;
	double ss = sigma*sigma;
	
	for(int i = 0; i < n; i++){
		double y = i - h;
		for(int j = 0; j < n; j++){
			double x = j - h;
			dst.at<double>(i, j) = exp(-(x*x + y*y) / (2*ss));
		}
	}
	double total = sum(dst)[0];
	
	for(int i = 0; i < n; i++){
		double y = i - h;
		for(int j = 0; j < n; j++){
			double x = j - h;
			dst.at<double>(i, j) *= x*x + y*y - 2*ss;
		}
	}

	dst /= pow(ss, 2) * total;
	dst -= mean(dst)[0];	// make sum of filter equal to 0
}

// Create a mask of where source image is NaN.
//
// _src -- source image
// _dst -- NaN mask (output)
//
void
nanmask(const Mat &_src, Mat &_dst)
{
	CHECKMAT(_src, CV_32FC1);
	_dst = Mat::zeros(_src.rows, _src.cols, CV_8UC1);
	auto src = _src.ptr<float>(0);
	uchar *dst = _dst.data;

	for(size_t i = 0; i < _src.total(); i++){
		if(std::isnan(src[i])){
			dst[i] = 255;
		}
	}
}

// Apply Laplacian of Gaussian (LoG) filter to an image containing NAN.
//
// src -- source image
// size -- width/height of LoG kernel
// sigma -- standard deviation of the Gaussian used in LoG kernel
// factor -- kernel is multiplied by this number before application
// dst -- destination image (output)
//
void
nanlogfilter(const Mat &src, const int size, const int sigma, const int factor, Mat &dst)
{
	Mat kern, mask;
	
	logkernel(size, sigma, kern);
	kern *= factor;
	
	src.copyTo(dst);
	nanmask(dst, mask);
	dst.setTo(0, mask);
	filter2D(dst, dst, -1, kern);
	dst.setTo(NAN, mask);
}

void
morphclosing(const Mat &src, int winsize, Mat &dst)
{
	Mat tmp;
	
	Mat elem = getStructuringElement(MORPH_RECT, Size(winsize, winsize));
	dilate(src, tmp, elem);
	erode(tmp, dst, elem);
}

void
gaussianblurnan(const Mat &src, Mat &dst, int ksize, double sigma)
{
	CHECKMAT(src, CV_32FC1);
	
	Mat _tmp = src.clone();
	auto tmp = _tmp.ptr<float>(0);
	for(size_t i = 0; i < _tmp.total(); i++){
		if(std::isnan(tmp[i])){
			tmp[i] = 0;
		}
	}
	GaussianBlur(_tmp, dst, Size(ksize, ksize), sigma, sigma);
}

void
adaptivethreshold(const Mat &_src, float low, float high, int winsize, double rat, Mat &_dst)
{
	Mat _B;
	
	CHECKMAT(_src, CV_32FC1);
	Mat _tmp = _src.clone();
	auto tmp = _tmp.ptr<float>(0);
	
	for(size_t i = 0; i < _tmp.total(); i++){
		if(std::isnan(tmp[i]) || tmp[i] < low){
			tmp[i] = 0;
		}else if(tmp[i] > high){
			tmp[i] = high;
		}
	}
	
	nanblur(_tmp, _B, winsize);
	CHECKMAT(_B, CV_32FC1);
	_dst.create(_src.size(), CV_8UC1);
	
	auto B = _B.ptr<float>(0);
	uchar *dst = _dst.data;
	
	for(size_t i = 0; i < _tmp.total(); i++){
		dst[i] = tmp[i] <= B[i]*rat ? 0 : 255;
	}
}

void
compute_eigenvals(const Mat1f &bt08,const Mat1f &bt11,const Mat1f &bt12,
                  const Mat1b border_mask, Mat1f &eigen)
{
  int y,x,i,j;
  int y_delta = 5;
  int x_delta = 5;
  int height = bt08.rows;
  int width = bt08.cols;
  eigen.create(height,width);
  
  int min_num = (2*y_delta +1) *(2*x_delta + 1)/2;
  int count_dim = 0;
  float bt08_sum,bt11_sum,bt12_sum,count,window_sum,row_sum, res_mean;
  float temp_bt08;
  float temp_bt11;
  float temp_bt12;


  float bt08_mean;
  float bt11_mean;
  float bt12_mean;

  vector<float> valid_bt08;
  vector<float> valid_bt11;
  vector<float> valid_bt12;

  vector<int> left_inds;


  Vector3f ones(1,1,1);
  Vector3f e1;
  MatrixXf r;
  Matrix3f A;

  for(y=y_delta;y<height-y_delta;y++){
    for(x=x_delta;x<width-x_delta;x++){
      if(isfinite(bt11(y,x))){//&& border_mask(y,x) == 0){
        // calc first window
        // we know that first left are nans so we don't calculate left inds     
        bt08_sum=bt11_sum=bt12_sum=0;
        valid_bt08.clear();
        valid_bt11.clear();
        valid_bt12.clear();
        for(i=-y_delta;i<y_delta+1;i++){
          for(j=-x_delta;j<x_delta+1;j++){              
           
	          //t = ((((cur_ind+k)%FILTER_TIME_SIZE)+FILTER_TIME_SIZE) % FILTER_TIME_SIZE);
	          temp_bt08 = bt08(y+i,x+j);
	          temp_bt11 = bt11(y+i,x+j);
	          temp_bt12 = bt12(y+i,x+j);

	          if(!std::isnan(temp_bt08) && !std::isnan(temp_bt11) && !std::isnan(temp_bt12)){
	            valid_bt08.push_back(temp_bt08);
	            valid_bt11.push_back(temp_bt11);
	            valid_bt12.push_back(temp_bt12);

	            bt08_sum+= temp_bt08;
	            bt11_sum+=temp_bt11;
	            bt12_sum+=temp_bt12;
	          }
          }
        }
  
        //if numberof pixels in window is greater tan threshold
        // calculate the mean of the norm of the pixels
        // projected into the second eigenvector
        count = valid_bt08.size();
        count_dim = valid_bt08.size();
        //printf("count = %f\n",count);
        //printf("min_num %d\n",min_num);
        if(count > min_num){
        
          //printf("in count\n");
          bt08_mean =bt08_sum/count;
          bt11_mean =bt11_sum/count;
          bt12_mean =bt12_sum/count;

          MatrixXf window(count_dim,3);
          for(i = 0; i < count; ++i){
            window(i,0) = valid_bt08[i] - bt08_mean;
            window(i,1) = valid_bt11[i] - bt11_mean;
            window(i,2) = valid_bt12[i] - bt12_mean;
          }
          
          A = (window.transpose()*window);
          e1 = A*(A*ones);
          e1/=sqrt(e1.transpose()*e1);
          r = window - (window*e1)*e1.transpose();
          window_sum =0;
          /*
          for(i = 0;i < count; ++i){
            row_sum = 0;
            row_sum+=r(i,0)*r(i,0);
            row_sum+=r(i,1)*r(i,1);
            row_sum+=r(i,2)*r(i,2);
            row_sum = sqrt(row_sum);
            window_sum += row_sum;
          }
          
          res_mean = window_sum/(float)valid_bt08.size();
          */
          for(i = 0; i < 3; ++i){
          	row_sum = 0;
          	for(j = 0; j < count; ++j){
          		row_sum += r(j,i) * r(j,i);
          	}
          	row_sum = sqrt(row_sum);
          	window_sum += row_sum;
          }
          res_mean = window_sum/ 3.0;
          eigen(y,x) = res_mean;
          
        }
      }
    }
  }
}