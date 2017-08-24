//
// SST Pattern Test
//

#include "spt.h"
#include "fastBilateral.hpp"


// thernal fronts and their sides
enum {
	FRONT_INVALID = -1,
	FRONT_INIT = 0,	// initial fronts
	FRONT_THIN,	// thinned fronts
	FRONT_LEFT,	// left side
	FRONT_RIGHT,	// right side
};



void
rectdilate(InputArray src, OutputArray dst, int ksize)
{
	dilate(src, dst, getStructuringElement(MORPH_RECT, Size(ksize, ksize)));
}

void
recterode(InputArray src, OutputArray dst, int ksize)
{
	erode(src, dst, getStructuringElement(MORPH_RECT, Size(ksize, ksize)));
}





// Connected components wrapper that limits the minimum size of components.
//
// mask -- the image to be labeled
// connectivity -- 8 or 4 for 8-way or 4-way connectivity respectively
// lim -- limit on the minimum size of components
// cclabels -- destination labeled image (output)
//
static int
connectedComponentsWithLimit(const Mat &mask, int connectivity, int lim, Mat1i &cclabels)
{
	Mat stats, centoids;
	int ncc = connectedComponentsWithStats(mask, cclabels, stats, centoids, connectivity, CV_32S);
	
	// Remove small connected components and rename labels to be contiguous.
	// Also, set background label 0 (where mask is 0) to -1.
	vector<int> ccrename(ncc);
	ccrename[0] = COMP_INVALID;
	int newlab = 0;
	for(int lab = 1; lab < ncc; lab++){
		if(stats.at<int>(lab, CC_STAT_AREA) >= lim){
			ccrename[lab] = newlab++;
		}else{
			ccrename[lab] = COMP_SPECKLE;
		}
	}
	ncc = newlab;
	for(size_t i = 0; i < mask.total(); i++){
		cclabels(i) = ccrename[cclabels(i)];
	}

	return ncc;
}



// Thin fronts based on SST gradient.
// Prototype: matlab/front_thinning.m
//
// frontsimg -- front labels
// dY -- gradient in Y direction
// dX -- gradient in X direction
// sstmag -- gradient magnitude
// thinnedf -- front labels containing thinned fronts (output)
//
static void
thinfronts(const Mat_<schar> &frontsimg, const Mat1f &dY, const Mat1f &dX, const Mat1f &sstmag, Mat_<schar> &thinnedf)
{
	thinnedf.create(frontsimg.size());
	thinnedf = FRONT_INVALID;
	int i = 0;
	int alpha_step = 2;
	for(int y = 0; y < frontsimg.rows; y++){
		for(int x = 0; x < frontsimg.cols; x++){
			if(frontsimg(i) == FRONT_INIT){
				double dy = dY(i) / sstmag(i);
				double dx = dX(i) / sstmag(i);

				int maxy = y;
				int maxx = x;
				float maxg = sstmag(maxy, maxx);
				for(int alpha = -alpha_step; alpha <= alpha_step; alpha++){
					int yy = round(y + alpha*dx);
					int xx = round(x - alpha*dy);
					
					if(0 <= yy && yy < frontsimg.rows
					&& 0 <= xx && xx < frontsimg.cols
					&& sstmag(yy, xx) > maxg){
						maxy = yy;
						maxx = xx;
						maxg = sstmag(yy, xx);
					}
				}
				if(frontsimg(maxy, maxx) == FRONT_INIT){
					thinnedf(maxy, maxx) = FRONT_THIN;
				}
			}
			i++;
		}
	}
}


enum {
	ACSPOTestsEasycloudMask = 0xc6
};

// Attempt to connect broken fronts by using cos-similarity of gradient vectors.
// Each pixels within a window is compared to the pixel at the center of the
// window.
// Prototype: matlab/front_connect.m
//
// fronts -- fronts containing only initial fronts (FRONT_INIT) (intput & output)
// dX -- gradient in x direction
// dY -- gradient in y direction
// sstmag -- SST gradient magnitude
// easyclouds -- guaranteed cloud based on various thresholds
// lam2 -- local max
//
static void
connectfronts(Mat_<schar> &fronts, const Mat1f &dX, const Mat1f &dY, const Mat1f &sst, const Mat1f &logmag, 
	          const Mat1b &easyclouds, const Mat1f &lam2)
{
	Mat_<schar> thinnedf;
	
	const int W = 11;	// window width/height
	const int mid = fronts.cols*(W/2) + (W/2);	// pixel at center of window
	
	Mat1b rejected(fronts.size());
	rejected = 0;
	
	Mat1f tmp;
	exp(100*(lam2+0.01), tmp);
	Mat1f llam = 1.0/(1+tmp);

	Mat1b valid =  (fronts != FRONT_INIT) & (logmag > GRAD_LOW) & ~easyclouds;
	
	for(int iter = 0; iter < 10; iter++){
		int i = 0;
		// For each pixel with full window, where the pixel
		// is top left corner of the window
		for(int y = 0; y < fronts.rows-W+1; y++){
			for(int x = 0; x < fronts.cols-W+1; x++){
				if(fronts(i + mid) == FRONT_INIT){
					double cdY = dY(i + mid);
					double cdX = dX(i + mid);
					double csst = sst(i + mid);
					double max = 0;
					int k = i;
					int argmax = i + mid;
					for(int yy = y; yy < y+W; yy++){
						for(int xx = x; xx < x+W; xx++){
							if(valid(k) != 0 && fabs(sst(k) - csst) < 0.5 && rejected(k) == 0){
								// cos-similarity
								double sim = llam(k) * (dY(k)*cdY + dX(k)*cdX);
								if(sim > max){
									max = sim;
									argmax = k;
								}
							}
							k++;
						}
						k += fronts.cols - W;
					}
					fronts(argmax) = FRONT_INIT;
				}
				i++;
			}
		}
		thinfronts(fronts, dY, dX, logmag, thinnedf);
		
		for(size_t i = 0; i < thinnedf.total(); i++){
			switch(thinnedf(i)){
			default:
				if(fronts(i) == FRONT_INIT){
					fronts(i) = FRONT_INVALID;
					rejected(i) = 255;
				}
				break;
			case FRONT_THIN:
				fronts(i) = FRONT_INIT;
				break;
			}
		}
	}

}



// Write spt into NetCDF dataset ncid as variable named "spt_mask".
//
static void
writesptmask(int ncid, const Mat1b &newacspo, const Mat_<schar> &frontsimg)
{
	int n, varid, ndims, dimids[2];
	nc_type xtype;
	size_t len;
	
	const char varname[] = "spt_mask";
	const char varunits[] = "none";
	const char vardescr[] = "SPT mask packed into 1 byte: bits1-2 (00=clear; 01=probably clear; 10=cloudy; 11=clear-sky mask undefined); bit3 (0=no thermal front; 1=thermal front)";

	// chunk sizes used by acspo_mask
	const size_t chunksizes[] = {1024, 3200};
	
	// It's not possible to delete a NetCDF variable, so attempt to use
	// the variable if it already exists. Create the variable if it does not exist.
	n = nc_inq_varid(ncid, varname, &varid);
	if(n != NC_NOERR){
		n = nc_inq_dimid(ncid, "scan_lines_along_track", &dimids[0]);
		if(n != NC_NOERR){
			ncfatal(n, "nc_inq_dimid failed");
		}
		n = nc_inq_dimid(ncid, "pixels_across_track", &dimids[1]);
		if(n != NC_NOERR){
			ncfatal(n, "nc_inq_dimid failed");
		}
		
		n = nc_def_var(ncid, varname, NC_UBYTE, nelem(dimids), dimids, &varid);
		if(n != NC_NOERR){
			ncfatal(n, "nc_def_var failed");
		}
		n = nc_def_var_chunking(ncid, varid, NC_CHUNKED, chunksizes);
		if(n != NC_NOERR){
			ncfatal(n, "nc_def_var_chunking failed");
		}
		n = nc_def_var_deflate(ncid, varid, 0, 1, 1);
		if(n != NC_NOERR){
			ncfatal(n, "setting deflate parameters failed");
		}
		
		n = nc_put_att_text(ncid, varid, "UNITS", nelem(varunits)-1, varunits);
		if(n != NC_NOERR){
			ncfatal(n, "setting attribute UNITS failed");
		}
		n = nc_put_att_text(ncid, varid, "Description", nelem(vardescr)-1, vardescr);
		if(n != NC_NOERR){
			ncfatal(n, "setting attribute Description failed");
		}
	}
	
	// Varify that the netcdf variable has correct type and dimensions.
	n = nc_inq_var(ncid, varid, nullptr, &xtype, &ndims, dimids, nullptr);
	if(n != NC_NOERR){
		ncfatal(n, "nc_inq_var failed");
	}
	if(xtype != NC_UBYTE){
		eprintf("variable type is %d, want %d\n", xtype, NC_UBYTE);
	}
	if(ndims != 2){
		eprintf("variable dims is %d, want 2\n", ndims);
	}
	for(int i = 0; i < 2; i++){
		n = nc_inq_dimlen(ncid, dimids[i], &len);
		if(n != NC_NOERR){
			ncfatal(n, "nc_inq_dimlen failed");
		}
		if(len != static_cast<size_t>(newacspo.size[i])){
			eprintf("dimension %d is %d, want %d\n", i, len, newacspo.size[i]);
		}
	}
	
	Mat1b sptmask(newacspo.size());
	sptmask = 0;

	// add fronts to sptmask
	for(size_t i = 0; i < sptmask.total(); i++){
		sptmask(i) = (newacspo(i)&MaskCloud) >> MaskCloudOffset;
		if((newacspo(i)&MaskCloud) == MaskCloudClear && frontsimg(i) == FRONT_INIT){
			sptmask(i) |= (1<<2);
		}
	}

	// Write data into netcdf variable.
	n = nc_put_var_uchar(ncid, varid, sptmask.data);
	if(n != NC_NOERR){
		ncfatal(n, "nc_putvar_uchar failed");
	}
}


static void
maskfronts(const Mat1f &sst, const Mat1f &magsst, const Mat1f &magbt11, const Mat1f &bt12, const Mat1f &eigen,const Mat1f &laplacian,
	       const Mat1f &lam2, const Mat1f &medianSST, const Mat1b icemask, const Mat1b &landmask, Mat_<schar> &frontmask)
{
	float delta_n = 0.1;
	float T_low = 271.15;
	float delta_Lam = 0.01;
	float thresh_mag = 0.1;//0.2;

	float thresh_L = 0.8;
	float eigen_thresh = 2;
	float median_thresh = 0.5;
	float mag_ratio_thresh = 0.5;
	float std_thresh = 0.2;

	Mat1f magdiff = magsst - magbt11 ;
	Mat1f sst_row_diff, median_row_diff;
	row_neighbor_diffs(sst, sst_row_diff);
	row_neighbor_diffs(medianSST, median_row_diff);
	Mat1f std_median(sst.size());
	Mat1f std_sst(sst.size());
    stdfilter(median_row_diff, std_median, 7);
    stdfilter(sst_row_diff, std_sst, 7);


	frontmask.setTo(TEST_GRADMAG_LOW, (magsst <= thresh_mag));
	frontmask.setTo(FRONT_GUESS, (magsst > thresh_mag));	
	frontmask.setTo(TEST_LOCALMAX,((lam2>-delta_Lam) & (frontmask==FRONT_GUESS)));
	frontmask.setTo(STD_TEST, (std_sst - std_median >std_thresh));
	frontmask.setTo(BT12_TEST, (sst<bt12));
	frontmask.setTo(TEST_UNIFORMITY, (abs(sst - medianSST) > median_thresh));
	frontmask.setTo(TEST_CLOUD_BOUNDARY, (magdiff < -delta_n));
	frontmask.setTo(TEST_LAPLACIAN,(laplacian>thresh_L));
	frontmask.setTo(COLD_CLOUD, (sst<T_low));
	frontmask.setTo(RATIO_TEST, ((magsst>thresh_mag) & ((magdiff/magsst) > mag_ratio_thresh)));	
	frontmask.setTo(ICE_TEST,icemask);
	frontmask.setTo(LAND,landmask);
	frontmask.setTo(EIG_TEST,( eigen>eigen_thresh));
	

	printf("front guess = %d\n",FRONT_GUESS);
}


class SPT {
	int ncid;
	Mat1b acspo;
	Mat1b landmask;
	Mat1b border_mask;
	Mat1b markup;
	Mat1b icemask;
	Mat1f dY;
	Mat1f dX;
	Mat1f logmag;
	Mat1f lat;
	Mat1f lon;
	Mat1f bt12;
	Mat1f magbt12;
	Mat1f bt11;
	Mat1f magbt11;
	Mat1f bt03;
	Mat1f bt08;
	Mat1f sstmag;
	Mat1f sst;
	Mat laplacian;
	Mat residuals;
	Mat lam2;
	Mat1f diff_11_12;
	Mat1b easyclouds;
	Mat1b acspotests;
	Mat1f eigen;
	Mat1f medSST;
	Mat1b nan_mask;
	Mat1f sst_ref;

public:
	SPT(ACSPOFile &f);
	int quantizationcluster(Mat1i&);
	void run();
};

SPT::SPT(ACSPOFile &f)
{
	ncid = f.ncid;
	
	logprintf("reading data ...\n");
	f.readvar<float>("sst_regression", sst);
	f.readvar<float>("brightness_temp_chM15", bt11);
	f.readvar<float>("brightness_temp_chM16", bt12);
	f.readvar<float>("brightness_temp_chM12", bt03);
	f.readvar<float>("brightness_temp_chM14", bt08);
	f.readvar<float>("sst_reynolds", sst_ref);
	f.readvar<uchar>("acspo_mask", acspo);
	f.readvar<uchar>("individual_clear_sky_tests_results", acspotests);

	Mat1f dX_mag, dY_mag, dXX, dYY, dYX, dXY;

	logprintf("computing sstmag, etc....\n");
	
	medianBlur(sst, medSST, 5);

	gradientmag(bt11, magbt11);
	gradientmag(bt12, magbt12);
	gradientmag(sst, sstmag, dX, dY);
	

	logprintf("Laplacian of Gaussican...\n");
	nanlogfilter(sstmag, 17, 2, -17, logmag);
	logmag.setTo(0, logmag < 0);

	partial_derivative(dX, dXX, dXY);
	partial_derivative(dY, dYX, dYY);

	laplace_operator2(dXX, dYY,laplacian);


	partial_derivative(sstmag, dX_mag, dY_mag);
	partial_derivative(dX_mag, dXX, dXY);
	partial_derivative(dY_mag, dYX, dYY);
	localmax2( lam2, dXX, dYY, dXY, false);
	


	diff_11_12 = bt11 - bt12;
	easyclouds.create(sst.size());

	icemask = (acspo&(1<<5));
	landmask = (acspo & 4) != 0;

	acspo = (acspo & (128+64)) != 0;
	
	
	acspotests = ((acspotests&(1<<6)) != 0);

	
	get_landborders(landmask, border_mask,7);

	apply_land_mask( landmask, bt03);
	apply_land_mask( landmask, bt11);
	apply_land_mask( landmask, bt12);
	apply_land_mask( landmask, sst);

	compute_eigenvals(bt03, bt11, bt12, border_mask, eigen);

	
	nanmask(sst, nan_mask);

}


void
SPT::run()
{
	Mat_<schar> frontmask(sst.size());
	Mat_<schar> glabelsnn;
	Mat_<schar> frontsimg(sst.size());
	Mat1i labels;
	Mat1i glabels;
	Mat1f hist;
	Mat1b easyclouds_D, easyclouds_temp,landmask_D;
	Mat1b newacspo(sst.size());

	int ref_thresh = -8;

	maskfronts(sst,sstmag, magbt11, bt12, eigen, laplacian, lam2, medSST, icemask, landmask, frontmask);
	SAVENC(frontmask);


	easyclouds=0;
	easyclouds.setTo(1,(frontmask<0));
	

	Mat1b ind_ocean = ((nan_mask==0) & (easyclouds == 0) & (acspo == 0));
	Mat1b ind_test = ((nan_mask==0) & (easyclouds == 0) & (acspo));
	
	double bt11_low,bt11_high, cluster_high, cluster_low;
	Point min_loc, max_loc;

	minMaxLoc(bt11, &bt11_low, &bt11_high, &min_loc, &max_loc, ind_ocean);

	float lows[3] = {(float)bt11_low,-1,0};
	float highs[3] = {(float)bt11_high,5,5};
	float steps[3] = {0.2,0.025,0.025};

	generate_cloud_histogram_3d(bt11-bt11_low, bt11-bt12+1, bt11-bt08, lows, highs, steps, ind_ocean, hist);
	Mat1f hist_count(sst.size());
	check_cloud_histogram_3d(bt11-bt11_low, bt11-bt12+1, bt11-bt08, lows, highs, steps, ind_test, hist, NN_TEST, frontmask, hist_count);

	easyclouds.setTo(1,frontmask < 0);

	/* New code  */
	Mat1i cluster_labels(sst.size());

	connectedComponentsWithLimit(((acspo) & (easyclouds==0)), 4, MIN_CLUSTER_SIZE, cluster_labels);
	easyclouds.setTo(1,(cluster_labels==COMP_SPECKLE));



	Mat1b ref_diff = ((sst - sst_ref) < ref_thresh);

	minMaxLoc(cluster_labels, &cluster_low, &cluster_high, &min_loc, &max_loc);
	for(int i=1;i<cluster_high;++i){
		Scalar tempVal = mean( (sst - sst_ref), (cluster_labels==i) );
		if(tempVal[0] < ref_thresh){
			easyclouds.setTo(1,(cluster_labels==i));
			cluster_labels.setTo(-1,(cluster_labels==i));
		}
	}
	/* end new code */

	frontsimg.setTo(FRONT_INVALID,frontmask!=FRONT_GUESS);
	frontsimg.setTo(FRONT_INIT,frontmask==FRONT_GUESS);

	//TODO: pass logmag to thinfronts
	connectfronts(frontsimg, dX, dY, sst, logmag, easyclouds, lam2);

	rectdilate(landmask,landmask_D,7);
	frontsimg.setTo(FRONT_INVALID,landmask_D);

	connectedComponentsWithLimit(frontsimg == FRONT_INIT, 8, MIN_FRONT_SIZE, labels);
	labels.setTo(FRONT_INVALID,easyclouds);

	frontsimg.setTo(FRONT_INVALID,labels < 1);
	frontsimg.setTo(FRONT_INIT, labels > 0);

	newacspo = ((acspo) & (easyclouds==1));
	writesptmask(ncid, newacspo, frontsimg);
}



int
main(int argc, const char **argv)
{
	if(argc < 2){
		eprintf("usage: %s granule [markup]\n", argv[0]);
	}
	auto path = argv[1];
	
	logprintf("granule: %s\n", path);
	ACSPOFile f(path, NC_WRITE, false);
	

	auto spt = SPT(f);

	spt.run();

	f.close();
	logprintf("done\n");
	return 0;
}
