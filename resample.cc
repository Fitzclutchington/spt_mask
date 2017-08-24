//
// Resampling of image based on latitude
//

#include "spt.h"

template <class T>
static Mat
resample_unsort_(const Mat &sind, const Mat &img)
{
	Mat newimg;
	int i, j, k;

	CHECKMAT(sind, CV_32SC1);
	CV_Assert(img.channels() == 1);

	newimg = Mat::zeros(img.rows, img.cols, img.type());
	auto sp = sind.ptr<int>(0);
	auto ip = img.ptr<T>(0);
	k = 0;
	for(i = 0; i < newimg.rows; i++){
		for(j = 0; j < newimg.cols; j++){
			newimg.at<T>(sp[k], j) = ip[k];
			k++;
		}
	}
	return newimg;
}

// Returns the unsorted image of the sorted image img.
// Sind is the image of sort indices.
Mat
resample_unsort(const Mat &sind, const Mat &img)
{
	switch(img.type()){
	default:
		eprintf("unsupported type %s\n", type2str(img.type()));
		break;
	case CV_8UC1:
		return resample_unsort_<uchar>(sind, img);
	case CV_32FC1:
		return resample_unsort_<float>(sind, img);
	case CV_64FC1:
		return resample_unsort_<double>(sind, img);
	}
	// not reached
	abort();
}

template <class T>
static Mat
resample_sort_(const Mat &sind, const Mat &img)
{
	Mat newimg;
	int i, j, k;
	T *np;

	CHECKMAT(sind, CV_32SC1);
	CV_Assert(img.channels() == 1);

	newimg = Mat::zeros(img.rows, img.cols, img.type());
	auto sp = sind.ptr<int>(0);
	np = newimg.ptr<T>(0);
	k = 0;
	for(i = 0; i < newimg.rows; i++){
		for(j = 0; j < newimg.cols; j++){
			np[k] = img.at<T>(sp[k], j);
			k++;
		}
	}
	return newimg;
}

// Returns the sorted image of the unsorted image img.
// Sind is the image of sort indices.
Mat
resample_sort(const Mat &sind, const Mat &img)
{
	switch(img.type()){
	default:
		eprintf("unsupported type %s\n", type2str(img.type()));
		break;
	case CV_8UC1:
		return resample_sort_<uchar>(sind, img);
		break;
	case CV_32FC1:
		return resample_sort_<float>(sind, img);
		break;
	case CV_64FC1:
		return resample_sort_<double>(sind, img);
		break;
	}
	// not reached
	return Mat();
}

// Returns the average of 3 pixels.
static double
avg3(double a, double b, double c)
{
	if(std::isnan(b)){
		return NAN;
	}
	if(std::isnan(a) || std::isnan(c)){
		return b;
	}
	return (a+b+c)/3.0;
}

// Returns the average filter of image 'in' with a window of 3x1
// where sorted order is not the same as the original order.
// Sind is the sort indices giving the sort order.
static Mat
avgfilter3(const Mat &in, const Mat &sind)
{
	const int *sindp;
	const float *ip;
	Mat out;
	int i, j, rows, cols;
	float *op;

	CHECKMAT(in, CV_32FC1);
	CHECKMAT(sind, CV_32SC1);
	rows = in.rows;
	cols = in.cols;

	out.create(rows, cols, CV_32FC1);
	in.row(0).copyTo(out.row(0));
	in.row(rows-1).copyTo(out.row(rows-1));

	for(i = 1; i < rows-1; i++){
		ip = in.ptr<float>(i);
		op = out.ptr<float>(i);
		sindp = sind.ptr<int>(i);
		for(j = 0; j < cols; j++){
			if(sindp[j] != i){
				op[j] = avg3(ip[j-cols], ip[j], ip[j+cols]);
			}else{
				op[j] = ip[j];
			}
		}
	}
	return out;
}

// Interpolate the missing values in image simg and returns the result.
// Slat is the latitude image, and slandmask is the land mask image.
// All input arguments must already be sorted.
static Mat
resample_interp(const Mat &simg, const Mat &slat, const Mat &slandmask)
{
	int i, j, k, nbuf;
	Mat newimg, bufmat;
	double x, llat, rlat, lval, rval;

	CHECKMAT(simg, CV_32FC1);
	CHECKMAT(slat, CV_32FC1);
	CHECKMAT(slandmask, CV_8UC1);

	newimg = simg.clone();
	bufmat = Mat::zeros(simg.rows, 1, CV_32SC1);
	auto buf = bufmat.ptr<int>(0);

	for(j = 0; j < simg.cols; j++){
		nbuf = 0;
		llat = -999;
		lval = NAN;
		for(i = 0; i < simg.rows; i++){
			// land pixel, nothing to do
			if(slandmask.at<unsigned char>(i, j) != 0){
				continue;
			}

			// valid pixel
			if(!std::isnan(simg.at<float>(i, j))){
				// first pixel is not valid, so extrapolate
				if(llat == -999){
					for(k = 0; k < nbuf; k++){
						newimg.at<float>(buf[k],j) = simg.at<float>(i, j);
					}
					nbuf = 0;
				}

				// interpolate pixels in buffer
				for(k = 0; k < nbuf; k++){
					rlat = slat.at<float>(i, j);
					rval = simg.at<float>(i, j);
					x = slat.at<float>(buf[k], j);
					newimg.at<float>(buf[k],j) =
						lval + (rval - lval)*(x - llat)/(rlat - llat);
				}

				llat = slat.at<float>(i, j);
				lval = simg.at<float>(i, j);
				nbuf = 0;
				continue;
			}

			// not land and no valid pixel
			buf[nbuf++] = i;
		}
		// extrapolate the last pixels
		if(llat != -999){
			for(k = 0; k < nbuf; k++){
				newimg.at<float>(buf[k],j) = lval;
			}
		}
	}
	return newimg;
}

enum Pole {
	NORTHPOLE,
	SOUTHPOLE,
	NOPOLE,
};
typedef enum Pole Pole;

// Argsort latitude image 'lat' with given swath size.
// Image of sort indices are return in 'sortidx'.
static void
argsortlat(const Mat &lat, int swathsize, Mat &sortidx)
{
	int i, j, off, width, height, dir, d, split;
	Pole pole;
	Mat col, idx, botidx;
	Range colrg, toprg, botrg;
	
	CHECKMAT(lat, CV_32FC1);
	CV_Assert(swathsize >= 2);
	CV_Assert(lat.data != sortidx.data);
	
	width = lat.cols;
	height = lat.rows;
	sortidx.create(height, width, CV_32SC1);
	
	// For a column in latitude image, look at every 'swathsize' pixels
	// starting from 'off'. If they increases and then decreases, or
	// decreases and then increases, we're at the polar region.
	off = swathsize/2;
	
	pole = NOPOLE;
	
	for(j = 0; j < width; j++){
		col = lat.col(j);
		
		// find initial direction -- increase, decrease or no change
		dir = 0;
		for(i = off+swathsize; i < height; i += swathsize){
			dir = SGN(col.at<float>(i) - col.at<float>(i-swathsize));
			if(dir != 0){
				break;
			}
		}
		
		// find change in direction if there is one
		for(; i < height; i += swathsize){
			d = SGN(col.at<float>(i) - col.at<float>(i-swathsize));
			if(dir == 1 && d == -1){
				CV_Assert(pole == NOPOLE || pole == NORTHPOLE);
				pole = NORTHPOLE;
				break;
			}
			if(dir == -1 && d == 1){
				CV_Assert(pole == NOPOLE || pole == SOUTHPOLE);
				pole = SOUTHPOLE;
				break;
			}
		}
		
		if(i >= height){
			pole = NOPOLE;
			if(dir >= 0){
				sortIdx(col, sortidx.col(j), CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
			}else{
				sortIdx(col, sortidx.col(j), CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
			}
			continue;
		}
		
		split = i-swathsize;	// split before change in direction
		colrg = Range(j, j+1);
		toprg = Range(0, split);
		botrg = Range(split, height);
		
		if(pole == NORTHPOLE){
			botidx = sortidx(botrg, colrg);
			sortIdx(col.rowRange(toprg), sortidx(toprg, colrg),
				CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
			sortIdx(col.rowRange(botrg), botidx,
				CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
			botidx += split;
		}else{	// pole == SOUTHPOLE
			botidx = sortidx(botrg, colrg);
			sortIdx(col.rowRange(toprg), sortidx(toprg, colrg),
				CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
			sortIdx(col.rowRange(botrg), botidx,
				CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
			botidx += split;
		}
	}
}

/*
static void
benchmark_avgfilter3(Mat &img, Mat &sind, int N)
{
	int i;
	struct timespec tstart, tend;
	double secs;
	
	clock_gettime(CLOCK_MONOTONIC, &tstart);
	for(i = 0; i < N; i++){
		avgfilter3(img, sind);
	}
	clock_gettime(CLOCK_MONOTONIC, &tend);
	
	secs = ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - 
		((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec);
	printf("avgfilter3 took about %.5f seconds; %f ops/sec\n", secs, N/secs);
}
*/

void
resample_init(Resample &r, const Mat &lat, const Mat &acspo)
{
	CHECKMAT(lat, CV_32FC1);
	CHECKMAT(acspo, CV_8UC1);

	argsortlat(lat, VIIRS_SWATH_SIZE, r.sind);

	r.slat = resample_sort(r.sind, lat);
	r.sacspo = resample_sort(r.sind, acspo);
	r.slandmask = (r.sacspo & MaskLand) != 0;
}

// Resample VIIRS swatch image img with corresponding
// latitude image lat, and ACSPO mask acspo.
void
resample_float32(const Resample &r, const Mat &src, Mat &dst, bool sort)
{
	CHECKMAT(src, CV_32FC1);

	dst = sort ? resample_sort(r.sind, src) : src;
//benchmark_avgfilter3(src, r.sind, 100);
	dst = avgfilter3(dst, r.sind);
//dumpmat("avgfilter3_new.bin", dst);

	dst = resample_interp(dst, r.slat, r.slandmask);
}
