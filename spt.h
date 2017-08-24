//
// SST Pattern Test
//

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <netcdf.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <sstream>
#include <omp.h>
#include <unistd.h>	// access(3)
#include <cmath>

#include <Eigen/Dense>
using namespace Eigen;
#include "connectedcomponents.h"

using namespace cv;

#define USED(x)	((void)(x))
#define SQ(x)    ((x)*(x))
#define SGN(A)   ((A) > 0 ? 1 : ((A) < 0 ? -1 : 0 ))
#define nelem(x) (sizeof(x)/sizeof((x)[0]))
#define SAVENPY(X)	savenpy(#X ".npy", (X))
#define SAVENC(X)	if(DEBUG)savenc("../data/thermalfronts/" #X ".nc", (X))
#define SAVENCF(X, Y)	if(DEBUG)savenc("../data/thermalfronts/" #Y ".nc", (X))
#define SAVENCN(X, Y)	if(DEBUG)savenc(Y, (X))
#define CHECKMAT(M, T)	CV_Assert((M).type() == (T) && (M).isContinuous())
#define CLAMP(A,L,H)    ((A)<=(L) ? (L) : (A)<=(H) ? (A) : (H))
#define isfinite(x)	(std::isfinite(x))

#define GRAD_THRESH 0.2
#define GRAD_LOW 0.1
#define DELTARANGE_THRESH 0.3
#define DELTAMAG_LOW (GRAD_LOW/2.0)
#define LAM2_THRESH	-0.05
#define SST_LOW 270
#define SST_HIGH 309
#define DELTA_LOW -1
#define DELTA_HIGH 4
#define OMEGA_LOW -5
#define OMEGA_HIGH 3
#define ANOMALY_HIGH 10
#define ANOMALY_LOW -10
#define ANOMALY_THRESH -8
#define ALBEDO_LOW 3
#define ALBEDO_HIGH 4
#define EDGE_THRESH 1
#define STD_THRESH 0.5

#define TQ_STEP 1
#define DQ_STEP 0.5	// 0.5 doesn't work on some examples
#define OQ_STEP 0.5
#define AQ_STEP 1

#define TQ_HIST_STEP 1
#define DQ_HIST_STEP 0.25
#define OQ_HIST_STEP OQ_STEP

enum {
	DEBUG = 1,
	
	LUT_INVALID = -1,
	LUT_OCEAN = 0,
	LUT_CLOUD = 1,
	
	LUT_LAT_SPLIT = 4,
	
	COMP_INVALID = -1,	// invalid component
	COMP_SPECKLE = -2,	// component that is too small
	
	FRONT_SIDE_DIST = 7,	// multiplied to gradient dx/dy to obtain front sides
	MIN_FRONT_SIZE = 10,//100,
	MIN_FRONT_FRAGMENT = 10,
	MIN_CLUSTER_SIZE = 100,
};

enum {
	MaskInvalid       = (1<<0),    		// or Valid
	MaskDay           = (1<<1),         // or Night
	MaskLand          = (1<<2),         // or Ocean
	MaskTwilightZone  = (1<<3),         // or No Twilight Zone
	MaskGlint         = (1<<4),         // or No Sun Glint
	MaskIce           = (1<<5),         // or No Ice
	
	MaskCloudOffset   = 6,              // first bit of cloud mask
	MaskCloud         = ((1<<7)|(1<<6)),
	MaskCloudClear    = 0,              // 0 - Clear
	MaskCloudProbably = (1<<6),         // 1 - Probably cloudy
	MaskCloudSure     = (1<<7),         // 2 - Confidently  cloudy
	MaskCloudInvalid  = ((1<<7)|(1<<6)),  // 3 - Irrelevant to SST (which includes land, ice and invalid pixels)
};

enum {
	LAND = -12,
	ICE_TEST = -11,
	COLD_CLOUD = -10,
	STD_TEST = -8,
	NN_TEST = -7,
	BT12_TEST = -6,
	TEST_LAPLACIAN = -5,
	RATIO_TEST=-4,
	EIG_TEST = -3,
	TEST_UNIFORMITY = -2,
	TEST_CLOUD_BOUNDARY = -1,
	TEST_GRADMAG_LOW = 0,
	TEST_LOCALMAX,
	FRONT_GUESS,
};

enum {
	VIIRS_SWATH_SIZE = 16,
	MODIS_SWATH_SIZE = 10,
};

typedef struct Resample Resample;
struct Resample {
	Mat sind, slat, sacspo, slandmask;
};

// resample.cc
Mat resample_unsort(const Mat &sind, const Mat &img);
Mat resample_sort(const Mat &sind, const Mat &img);
void resample_init(Resample &r, const Mat &lat, const Mat &acspo);
void resample_float32(const Resample &r, const Mat &src, Mat &dst, bool sort);

// utils.cc
void eprintf(const char *fmt, ...);
void logprintf(const char *fmt, ...);
char* estrdup(const char *s);
void *emalloc(size_t n);
const char *type2str(int type);
void diffcloudmask(const Mat &_old, const Mat &_new, Mat &_rgb);
string outpath(const char *inpath, const char *suffix);
double meann(const float *x, int n);
double maxn(const float *x, int n);
double corrcoef(const float *x, const float *y, int n);
void generate_sst_histogram_4d(const Mat1f &sst, const Mat1f &diff_11_12, const Mat_<schar> &mask, Mat1f &hist);
void histogram_check_4d(const Mat1f &sst, const Mat1f &diff_11_12, Mat_<schar> &frontmask, const Mat1f hist);
void histogram_check_3d(const Mat1f &sst,  Mat_<schar> &frontmask, const Mat1f hist, int flag1, int flag2);
void generate_sst_histogram_3d(const Mat1f &sst,  const Mat_<schar> &mask, Mat1f &hist, int flag);
void get_landborders(const Mat &land_mask, Mat &border_mask,int kernel_size);
void apply_land_mask(const Mat1b &landmask,Mat1f &clear);
void generate_cloud_histogram_3d(const Mat1f &d1, const Mat1f &d2, const Mat1f &d3, const float *lows, const float *highs, const float *steps, const Mat1b &mask, Mat1f &hist);
void check_cloud_histogram_3d(const Mat1f &d1, const Mat1f &d2, const Mat1f &d3, const float *lows, const float *highs, const float *steps, const Mat1b &mask,const Mat1f &hist, int flag, Mat_<schar> &frontmask , Mat1f &hist_count);
void row_neighbor_diffs(const Mat1f &src, Mat1f &dst);

// io.cc
char	*fileprefix(const char *path);
int	readvar(int ncid, const char *name, Mat&);
void ncfatal(int n, const char *fmt, ...);
int open_resampled(const char *path, Resample &r, int omode);
Mat	readvar_resampled(int ncid, Resample &r, const char *name);
void createvar(int ncid, const char *varname, const char *varunits, const char *vardescr, const Mat &data);
void writevar(int ncid, const char *varname, const Mat &data);
void	savenc(const char*, const Mat&, bool compress=false);
void loadnc(const char*, Mat&);

// npy.cc
void savenpy(const char *filename, Mat &mat);
void loadnpy(const char *filename, Mat &mat);

// filters.cc
void	laplacian(Mat &src, Mat &dst);
void maxnorm3(const Mat &_src1, const Mat &_src2, const Mat &_src3, Mat &_dst);
void	nanblur(const Mat &src, Mat &dst, int ksize);
void	gradientmag(const Mat &src, Mat &dst, Mat1f &dX, Mat1f &dY);
void	gradientmag(const Mat &src, Mat &dst);
void	meangradientmag(const Mat1f &dx, const Mat1f &dy, int ksize, Mat1f &dst);
void	localmax(const Mat &sstmag, Mat &high, double sigma, bool debug=false);
void	stdfilter(const Mat &src, Mat &dst, int ksize);
void	stdfilter1(const Mat &src, Mat &dst, int ksize);
void	rangefilter(const Mat &src, Mat &dst, int ksize);
void	logkernel(int n, double sigma, Mat &dst);
void	nanlogfilter(const Mat &src, const int size, const int sigma, const int factor, Mat &dst);
void	atan2mat(const Mat &_X, const Mat &_Y, Mat &_dst);
void	cosmat(const Mat &_src, Mat &_dst);
void morphclosing(const Mat &src, int winsize, Mat &dst);
void gaussianblurnan(const Mat &src, Mat &dst, int ksize, double sigma);
void adaptivethreshold(const Mat &_src, float low, float high, int winsize, double rat, Mat &_dst);
void laplace_operator(const Mat &dX,const Mat1f &dY, Mat1f &dst, Mat1f &dXX, Mat1f &dYY);
void compute_eigenvals(const Mat1f &bt08,const Mat1f &bt11,const Mat1f &bt12, const Mat1b border_mask, Mat1f &eigen);
void nanmask(const Mat &_src, Mat &_dst);
void partial_derivative(const Mat &src, Mat1f &dX, Mat1f &dY);
void laplace_operator2(const Mat1f &dXX,const Mat1f &dYY, Mat &dst);
void localmax2(Mat &mu2, const Mat &Dxx, const Mat &Dyy, const Mat &Dxy, bool debug);

class ACSPOFile {
	bool resample;
	Resample r;
	
public:
	int ncid;
	
	ACSPOFile(const char *path, int omode, bool _resample);
	void close();
	void _readvar(const char *name, Mat &data, int type);
	template <class T> void readvar(const char *name, Mat_<T> &data);
};
