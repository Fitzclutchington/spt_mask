// Connected component functions backported from OpenCV 3.0,
// which has not been released yet as of 2014-10-15.

#include <opencv2/opencv.hpp>

namespace cv {

//! connected components algorithm output formats
enum { CC_STAT_LEFT   = 0,
       CC_STAT_TOP    = 1,
       CC_STAT_WIDTH  = 2,
       CC_STAT_HEIGHT = 3,
       CC_STAT_AREA   = 4,
       CC_STAT_MAX    = 5
     };

int connectedComponents(InputArray _img, OutputArray _labels,
	int connectivity, int ltype);
int connectedComponentsWithStats(InputArray _img, OutputArray _labels,
	OutputArray statsv, OutputArray centroids, int connectivity, int ltype);

}	// namespace cv
