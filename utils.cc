//
// Utility functions
//

#include "spt.h"

void
eprintf(const char *fmt, ...)
{
	va_list args;

	fflush(stdout);
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);

	if(fmt[0] != '\0' && fmt[strlen(fmt)-1] == ':'){
		fprintf(stderr, " %s", strerror(errno));
	}
	fprintf(stderr, "\n");

	exit(2);
}

void
logprintf(const char *fmt, ...)
{
	va_list args;
	time_t now;

	static time_t t0 = 0;
	if(t0 == 0){
		time(&t0);
	}
	time(&now);
	int dt = now - t0;
	printf("# [%02dm%02ds] ", dt/60, dt%60);
	
	fflush(stdout);
	va_start(args, fmt);
	vprintf(fmt, args);
	va_end(args);
	fflush(stdout);
}

char*
estrdup(const char *s)
{
	char *dup;

	dup = strdup(s);
	if(dup == nullptr){
		eprintf("strdup of \"%s\" failed:", s);
	}
	return dup;
}

void*
emalloc(size_t n)
{
	void *buf;

	buf = malloc(n);
	if(buf == nullptr){
		eprintf("malloc failed:");
	}
	return buf;
}

const char*
type2str(int type)
{
	switch(type){
	default:       return "UnknownType";
	case CV_8UC1:  return "CV_8UC1";
	case CV_8SC1:  return "CV_8SC1";
	case CV_16UC1: return "CV_16UC1";
	case CV_16SC1: return "CV_16SC1";
	case CV_32SC1: return "CV_32SC1";
	case CV_32FC1: return "CV_32FC1";
	case CV_64FC1: return "CV_64FC1";
	}
}

// Cloud mask values
enum {
	CMClear,
	CMProbably,
	CMSure,
	CMInvalid,
};

// Number of bits in cloud mask
enum {
	CMBits = 2,
};

enum {
	White	= 0xFFFFFF,
	Red		= 0xFF0000,
	Green	= 0x00FF00,
	Blue	= 0x0000FF,
	Yellow	= 0xFFFF00,
	JetRed	= 0x7F0000,
	JetBlue	= 0x00007F,
	JetGreen	= 0x7CFF79,
};

#define SetColor(v, c) do{ \
		(v)[0] = ((c)>>16) & 0xFF; \
		(v)[1] = ((c)>>8) & 0xFF; \
		(v)[2] = ((c)>>0) & 0xFF; \
	}while(0);

// Compute RGB diff image of cloud mask.
//
// _old -- old cloud mask (usually ACSPO cloud mask)
// _new -- new cloud mask (usually SPT cloud mask)
// _rgb -- RGB diff image (output)
//
void
diffcloudmask(const Mat &_old, const Mat &_new, Mat &_rgb)
{
	uchar *old, *new1, *rgb, oval, nval;
	
	CHECKMAT(_old, CV_8UC1);
	CHECKMAT(_new, CV_8UC1);
	
	_rgb.create(_old.size(), CV_8UC3);
	rgb = _rgb.data;
	old = _old.data;
	new1 = _new.data;
	
	for(size_t i = 0; i < _old.total(); i++){
		oval = old[i]>>MaskCloudOffset;
		nval = new1[i] & 0x03;
		
		if(oval == CMProbably){
			oval = CMSure;
		}
		if(nval == CMProbably){
			nval = CMSure;
		}
		
		switch((oval<<CMBits) | nval){
		default:
			SetColor(rgb, Yellow);
			break;
		
		case (CMInvalid<<CMBits) | CMInvalid:
			SetColor(rgb, White);
			break;
		
		case (CMClear<<CMBits) | CMClear:
			SetColor(rgb, JetBlue);
			break;
		
		case (CMSure<<CMBits) | CMSure:
			SetColor(rgb, JetRed);
			break;
		
		case (CMSure<<CMBits) | CMClear:
		case (CMInvalid<<CMBits) | CMClear:
			SetColor(rgb, JetGreen);
			break;
		}
		rgb += 3;
	}
}

// outpath("quux.nc", "_frontstats.nc") -> "quux_frontstats.nc"
// outpath("/foo/bar/quux.nc", "_frontstats.nc") -> "/foo/bar/quux_frontstats.nc"
string
outpath(const char *inpath, const char *suffix)
{
	auto p = string(inpath);
	auto slash = p.find_last_of("/");
	auto dot = p.find_last_of(".");

	if((slash == string::npos && dot != string::npos)
	|| (slash != string::npos && dot != string::npos && dot > slash)){
		p = p.substr(0, dot);
	}
	return p + suffix;
}

// Return mean of first n elements of x.
//
double
meann(const float *x, int n)
{
	if(n <= 0){
		return 0;
	}
	
	double sum = 0;
	for(int i = 0; i < n; i++){
		sum += x[i];
	}
	return sum/n;
}

// Return maximum of first n elements of x.
//
double
maxn(const float *x, int n)
{
	if(n <= 0){
		return -999;
	}
	
	double max = x[0];
	for(int i = 1; i < n; i++){
		if(x[i] > max){
			max = x[i];
		}
	}
	return max;
	
}

// Return sample correlation coefficient of first n elements of x and y.
// Computed as:
// sum((x - mean(x)) * (y - mean(y))) / (sqrt(sum((x - sx)**2)) * sqrt(sum((y - sy)**2)))
//
// See http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#For_a_sample
//
double
corrcoef(const float *x, const float *y, int n)
{
	if(n <= 0){
		return 0;
	}

	double mx = meann(x, n);
	double my = meann(y, n);
	
	// top = sum((x-mean(x)) * (y-mean(y)))
	double top = 0;
	for(int i = 0; i < n; i++){
		top += (x[i]-mx) * (y[i]-my);
	}
	
	// sx = sum((x-mean(x))**2)
	double sx = 0;
	for(int i = 0; i < n; i++){
		sx += SQ(x[i] - mx);
	}
	
	// sy = sum((y-mean(y))**2)
	double sy = 0;
	for(int i = 0; i < n; i++){
		sy += SQ(y[i] - my);
	}
	
	// return top / (sqrt(sx) * sqrt(sy))
	double bot = sqrt(sx) * sqrt(sy);
	if(bot == 0){
		return 0;
	}
	return top/bot;
}

/*
void
generate_sst_histogram_4d(const Mat1f &sst, const Mat1f &diff_11_12, const Mat_<schar> &mask, Mat1f &hist)
{
	printf("in function\n");
	int x,y,m,n,s,d,i,j,k;
	int height = sst.rows;
	int width = sst.cols;
	float step = 20;
	float d_sst = 0.2;
	float d_diff = 0.1;

	float sst_low = 271.15;
	float sst_high = 310;
	float diff_high = 5;
	float diff_low = -2;

	printf("height = %d width = %d\n",height,width);

	int y_size = ceil(height/float(step)) + 1;

	int x_size = ceil(width/float(step)) + 1;

	int sst_range = ceil(sst_high - sst_low);
	int sst_size = ceil(sst_range/d_sst) + 1;
	int diff_range = ceil(diff_high - diff_low);
	int diff_size = ceil(diff_range/d_diff) + 1;
	printf("y_size = %d x_size = %d sst_size = %d diff_size = %d\n", y_size, x_size, sst_size, diff_size);
	int dims[4] = {y_size,x_size, sst_size ,  diff_size};
	printf("set up dimensions\n");
	hist.create( 4,dims);
	hist.setTo(0);

	printf("initialized variables\n");
	for(y = 0; y < height; ++y){
		for(x = 0; x < width; ++x){
			if((mask(y,x) == TEST_LAPLACIAN || mask(y,x) == COLD_CLOUD) && isfinite(sst(y,x)) && isfinite(diff_11_12(y,x))){ 
				m = round(y/step);
				n = round(x/step);
				s = round((sst(y,x)-sst_low)/d_sst);
				d = round(diff_11_12(y,x)/d_diff);
				s < 0 ? s = 0 : s = s;
				s > sst_size - 1 ? s = sst_size -1 : s = s;

				d < 0 ? d = 0 : d = d;
				d > diff_size - 1 ? d = diff_size - 1 : d = d;


				if(m >= 0 && m < y_size && n >= 0 && n < x_size && s >= 0 && s < sst_size && d >= 0 && d < diff_size){
					//printf("val of time = %d, val of dt = %d",n,m);
					for(i = -1; i <= 1; ++i){
						for(j = -1; j <= 1; ++j){
							for(k = -s; k <=0 ; ++k){
									if(n+j >= 0 && m+i>=0 && n+j < x_size && m+i < y_size && s+k <sst_size ){
										int idx[4] = {m+i,n+j,s+k,d};
										hist.at<float>(idx)++;
									} 
								
							}
						}
					}						
				}
				
					
				else{
					printf("OUT OF BOUNDS\n");
					printf("m = %d n = %d s = %d\n, d=%d\n",m,n,s,d);
				}
				
			}
		}
	}
}

void
histogram_check_4d(const Mat1f &sst, const Mat1f &diff_11_12, Mat_<schar> &frontmask, const Mat1f hist)
{
	printf("in function\n");
	int x,y,m,n,s,d;
	int height = sst.rows;
	int width = sst.cols;
	float step = 20;
	float d_sst = 0.2;
	float d_diff = 0.1;

	float sst_low = 271.15;
	float sst_high = 310;
	float diff_high = 5;
	float diff_low = -2;


	int y_size = ceil(height/float(step)) + 1;

	int x_size = ceil(width/float(step)) + 1;

	int sst_range = ceil(sst_high - sst_low);
	int sst_size = ceil(sst_range/d_sst) + 1;
	int diff_range = ceil(diff_high - diff_low);
	int diff_size = ceil(diff_range/d_diff) + 1;

	for(y = 0; y < height; ++y){
		for(x = 0; x < width; ++x){
			if(frontmask(y,x) == FRONT_GUESS && isfinite(sst(y,x)) && isfinite(diff_11_12(y,x))){ //| mask(y,x) == 3{
				m = round(y/step);
				n = round(x/step);
				s = round((sst(y,x)-sst_low)/d_sst);
				d = round(diff_11_12(y,x)/d_diff);
				s < 0 ? s = 0 : s = s;
				s > sst_size - 1 ? s = sst_size -1 : s = s;

				d < 0 ? d = 0 : d = d;
				d > diff_size - 1 ? d = diff_size - 1 : d = d;


				if(m >= 0 && m < y_size && n >= 0 && n < x_size && s >= 0 && s < sst_size && d >= 0 && d < diff_size){
					int idx[4] = {m, n, s, d};
					if(hist.at<float>(idx)){
						frontmask(y,x) = TEST_LAPLACIAN_HIST_FRONT;
					}
				}
				
					
				else{
					printf("OUT OF BOUNDS\n");
					printf("m = %d n = %d s = %d\n, d=%d\n",m,n,s,d);
				}
				
			}
			
			if(frontmask(y,x) == TEST_GRADMAG_LOW  && isfinite(sst(y,x)) && isfinite(diff_11_12(y,x))){ //| mask(y,x) == 3{
				m = round(y/step);
				n = round(x/step);
				s = round((sst(y,x)-sst_low)/d_sst);
				d = round(diff_11_12(y,x)/d_diff);
				s < 0 ? s = 0 : s = s;
				s > sst_size - 1 ? s = sst_size -1 : s = s;

				d < 0 ? d = 0 : d = d;
				d > diff_size - 1 ? d = diff_size - 1 : d = d;


				if(m >= 0 && m < y_size && n >= 0 && n < x_size && s >= 0 && s < sst_size && d >= 0 && d < diff_size){
					int idx[4] = {m, n, s, d};
					if(hist.at<float>(idx)){
						frontmask(y,x) = TEST_LAPLACIAN_HIST;
					}
				}
				
					
				else{
					printf("OUT OF BOUNDS\n");
					printf("m = %d n = %d s = %d\n, d=%d\n",m,n,s,d);
				}
				
			}
			
		}
	}
}
*/
void
generate_sst_histogram_3d(const Mat1f &sst,  const Mat_<schar> &mask, Mat1f &hist, int flag)
{
	printf("in function\n");
	int x,y,m,n,s,i,j,k;
	int height = sst.rows;
	int width = sst.cols;
	float step = 10;
	float d_sst = 0.2;


	float sst_low = 271.15;
	float sst_high = 310;


	printf("height = %d width = %d\n",height,width);

	int y_size = ceil(height/float(step)) + 1;

	int x_size = ceil(width/float(step)) + 1;

	int sst_range = ceil(sst_high - sst_low);
	int sst_size = ceil(sst_range/d_sst) + 1;

	printf("y_size = %d x_size = %d sst_size = %d \n", y_size, x_size, sst_size);
	int dims[3] = {y_size,x_size, sst_size };
	printf("set up dimensions\n");
	hist.create( 3,dims);
	hist.setTo(0);

	printf("initialized variables\n");
	for(y = 0; y < height; ++y){
		for(x = 0; x < width; ++x){
			if((mask(y,x) == flag || mask(y,x) == COLD_CLOUD) && isfinite(sst(y,x)) ){ 
				m = round(y/step);
				n = round(x/step);
				s = round((sst(y,x)-sst_low)/d_sst);
		
				s < 0 ? s = 0 : s = s;
				s > sst_size - 1 ? s = sst_size -1 : s = s;


				if(m >= 0 && m < y_size && n >= 0 && n < x_size && s >= 0 && s < sst_size ){
				
					for(i = -1; i <= 1; ++i){
						for(j = -1; j <= 1; ++j){
							for(k = -s; k <=0 ; ++k){
									if(n+j >= 0 && m+i>=0 && n+j < x_size && m+i < y_size && s+k <sst_size ){
										hist(m+i,n+j,s+k)++;
									} 
								
							}
						}
					}						
				}
				
					
				else{
					printf("OUT OF BOUNDS\n");
					printf("m = %d n = %d s = %d\n",m,n,s);
				}
				
			}
		}
	}
}

void
histogram_check_3d(const Mat1f &sst,  Mat_<schar> &frontmask, const Mat1f hist, int flag2)
{

	int x,y,m,n,s;
	int height = sst.rows;
	int width = sst.cols;
	float step = 10;
	float d_sst = 0.2;

	float sst_low = 271.15;
	float sst_high = 310;


	int y_size = ceil(height/float(step)) + 1;

	int x_size = ceil(width/float(step)) + 1;

	int sst_range = ceil(sst_high - sst_low);
	int sst_size = ceil(sst_range/d_sst) + 1;


	for(y = 0; y < height; ++y){
		for(x = 0; x < width; ++x){

			if(frontmask(y,x) == TEST_GRADMAG_LOW  && isfinite(sst(y,x))){ //| mask(y,x) == 3{
				m = round(y/step);
				n = round(x/step);
				s = round((sst(y,x)-sst_low)/d_sst);
		
				s < 0 ? s = 0 : s = s;
				s > sst_size - 1 ? s = sst_size -1 : s = s;



				if(m >= 0 && m < y_size && n >= 0 && n < x_size && s >= 0 && s < sst_size){
					if(hist(m,n,s)){
						//frontmask(y,x) = TEST_LAPLACIAN_HIST;
						frontmask(y,x) = flag2;
					}
				}
				
					
				else{
					printf("OUT OF BOUNDS\n");
					printf("m = %d n = %d s = %d\n\n",m,n,s);
				}
				
			}
			
		}
	}
}

void
get_landborders(const Mat &land_mask, Mat &border_mask,int kernel_size)
{
  
  	CHECKMAT(land_mask, CV_8UC1);
  	border_mask.create(land_mask.size(),CV_8UC1);

	int x,y;
	int height = land_mask.size[0];
	int width = land_mask.size[1];

	Mat element = getStructuringElement( MORPH_RECT,
	                                   Size( kernel_size, kernel_size ) );
	dilate(land_mask, border_mask,element);
	for(y=0;y<height;y++){
		for(x=0;x<width;x++){
		  border_mask.at<uchar>(y,x) -= land_mask.at<uchar>(y,x);
		}
	}
}

void
apply_land_mask(const Mat1b &landmask,Mat1f &clear)
{
    int x,y;
    int height = landmask.size[0];
    int width = landmask.size[1];
    for(y=0;y<height;y++){
        for(x=0;x<width;x++){
            if( landmask(y,x) != 0 ){
                clear(y,x) = NAN;
            }
        }
    }
    
}

void
generate_cloud_histogram_3d(const Mat1f &d1, const Mat1f &d2, const Mat1f &d3, const float *lows,
	                        const float *highs, const float *steps, const Mat1b &mask, Mat1f &hist)
{
	printf("in function\n");
	int x,y,m,n,s,i,j,k;
	float range;

	int height = d1.rows;
	int width =  d1.cols;
	
	int sizes[3];

	// determine the size of each dimension of the histogram
	for(i = 0; i < 3; ++i){
		range = ceil(highs[i] - lows[i]);
		sizes[i] = ceil(range/steps[i]);
	}

	

	printf("x_size = %d y_size = %d z_size = %d \n", sizes[0], sizes[1], sizes[2]);

	hist.create( 3,sizes);
	hist.setTo(0);
	printf("set up histogram\n");

	printf("initialized variables\n");
	for(y = 0; y < height; ++y){
		for(x = 0; x < width; ++x){
			if(mask(y,x)  && isfinite(d1(y,x)) && isfinite(d2(y,x)) && isfinite(d3(y,x))){ 
				m = round(d1(y,x)/steps[0]);
				n = round(d2(y,x)/steps[1]);
				s = round(d3(y,x)/steps[2]);
				
				
				if(m > 0 && m < (sizes[0] -1) && n > 0 && n < (sizes[1] -1)&& s > 0 && s < (sizes[2] -1)){
					hist(m,n,s)++;
				}						
								
			}
		}
	}
}

void
check_cloud_histogram_3d(const Mat1f &d1, const Mat1f &d2, const Mat1f &d3, const float *lows, const float *highs,
	                     const float *steps, const Mat1b &mask,const Mat1f &hist, int flag, Mat_<schar> &frontmask, Mat1f &hist_count )
{
	printf("in function\n");
	int x,y,m,n,s,i;
	float range;

	int height = d1.rows;
	int width =  d1.cols;
	
	int sizes[3];

	// determine the size of each dimension of the histogram
	for(i = 0; i < 3; ++i){
		range = ceil(highs[i] - lows[i]);
		sizes[i] = ceil(range/steps[i]);
	}


	printf("initialized variables\n");
	for(y = 0; y < height; ++y){
		for(x = 0; x < width; ++x){
			if(mask(y,x) && isfinite(d1(y,x)) && isfinite(d2(y,x)) && isfinite(d3(y,x))){ 
				m = round(d1(y,x)/steps[0]);
				n = round(d2(y,x)/steps[1]);
				s = round(d3(y,x)/steps[2]);
				
				if(m > 0 && m < (sizes[0] -1) && n > 0 && n < (sizes[1] -1)&& s > 0 && s < (sizes[2] -1)){
					hist_count(y,x) = hist(m,n,s);
					if(hist(m,n,s) < 1){
						frontmask(y,x) = flag;
					}				
				}	
				else{
					frontmask(y,x) = flag;
				}			
			}
		}
	}
}

void
row_neighbor_diffs(const Mat1f &src, Mat1f &dst)
{	
	int y,x;
	dst.create(src.size());
	dst.setTo(0);

	int height = src.rows;
	int width = src.cols;
	for(y=0;y<height;++y){
		for(x = 0; x < width-1; ++x){
			dst(y,x) = src(y,x) - src(y,x+1);
		}
	}
}