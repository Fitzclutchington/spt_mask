//
// I/O utility functions
//

#include "spt.h"

enum {
	MAXDIMS = 5,
};

char*
fileprefix(const char *path)
{
	const char *b;
	char *p, *s;
	
	b = strrchr(path, '/');
	if(b == nullptr){
		b = path;
	}else{
		b++;
	}
	p = strdup(b);
	s = strrchr(p, '.');
	if(s != nullptr){
		*s = '\0';
	}
	return p;
}

int
readvar(int ncid, const char *name, Mat &img)
{
	int i, varid, n, ndims, dimids[MAXDIMS], ishape[MAXDIMS], cvt;
	size_t shape[MAXDIMS] = {};
	nc_type nct;
	
	n = nc_inq_varid(ncid, name, &varid);
	if(n != NC_NOERR){
		ncfatal(n, "nc_inq_varid failed for variable %s", name);
	}
	n = nc_inq_var(ncid, varid, nullptr, &nct, &ndims, dimids, nullptr);
	if(n != NC_NOERR){
		ncfatal(n, "nc_inq_var failed for variable %s", name);
	}
	if(ndims > MAXDIMS){
		eprintf("number of dimensions %d > MAXDIMS=%d\n", ndims, MAXDIMS);
	}
	
	for(i = 0; i < ndims; i++){
		n = nc_inq_dimlen(ncid, dimids[i], &shape[i]);
		if(n != NC_NOERR){
			ncfatal(n, "nc_inq_dimlen failed for dim %d", dimids[i]);
		}
	}
	
	cvt = -1;
	switch(nct){
	default:
		eprintf("unknown netcdf data type");
		break;
	case NC_BYTE:	cvt = CV_8SC1; break;
	case NC_UBYTE:	cvt = CV_8UC1; break;
	case NC_SHORT:	cvt = CV_16SC1; break;
	case NC_USHORT:	cvt = CV_16UC1; break;
	case NC_INT:	cvt = CV_32SC1; break;
	case NC_FLOAT:	cvt = CV_32FC1; break;
	case NC_DOUBLE:	cvt = CV_64FC1; break;
	}
	
	for(i = 0; i < MAXDIMS; i++){
		ishape[i] = shape[i];
	}
	img = Mat(ndims, ishape, cvt);
	n = nc_get_var(ncid, varid, img.data);
	if(n != NC_NOERR){
		ncfatal(n, "readvar: nc_get_var '%s' failed", name);
	}
	return varid;
}

void
savenc(const char *path, const Mat &mat, bool compress)
{
	int i, n, ncid, dims, varid, xtype;
	int dimids[MAXDIMS] = {};
	char *name;
	const char *dimnames[MAXDIMS] = {
		"dim0",
		"dim1",
		"dim2",
		"dim3",
		"dim4",
	};
	
	dims = mat.dims;
	if(mat.channels() > 1){
		dims++;
	}
	if(dims > MAXDIMS){
		eprintf("savenc: too many dimensions %d\n", dims);
	}
	
	n = nc_create(path, NC_NETCDF4, &ncid);
	if(n != NC_NOERR){
		ncfatal(n, "savenc: creating %s failed", path);
	}
	for(i = 0; i < mat.dims; i++){
		n = nc_def_dim(ncid, dimnames[i], mat.size[i], &dimids[i]);
		if(n != NC_NOERR){
			ncfatal(n, "savenc: creating dim %d failed", i);
		}
	}
	if(mat.channels() > 1){
		n = nc_def_dim(ncid, dimnames[i], mat.channels(), &dimids[i]);
		if(n != NC_NOERR){
			ncfatal(n, "savenc: creating dim %d failed", i);
		}
	}
	
	xtype = -1;
	switch(mat.depth()){
	default:
		eprintf("savenc: unsupported type %s\n", type2str(mat.type()));
		break;
	case CV_8U:	xtype = NC_UBYTE; break;
	case CV_8S:	xtype = NC_BYTE; break;
	case CV_16U:	xtype = NC_USHORT; break;
	case CV_16S:	xtype = NC_SHORT; break;
	case CV_32S:	xtype = NC_INT; break;
	case CV_32F:	xtype = NC_FLOAT; break;
	case CV_64F:	xtype = NC_DOUBLE; break;
	}
	
	n = nc_def_var(ncid, "data", xtype, dims, dimids, &varid);
	if(n != NC_NOERR){
		ncfatal(n, "savenc: creating variable failed");
	}
	if(compress){	// enable compression?
		n = nc_def_var_deflate(ncid, varid, 0, 1, 1);
		if(n != NC_NOERR){
			ncfatal(n, "savenc: setting deflate parameters failed");
		}
	}

	n = nc_put_var(ncid, varid, mat.data);
	if(n != NC_NOERR){
		ncfatal(n, "savenc: writing variable failed");
	}
	n = nc_close(ncid);
	if(n != NC_NOERR){
		ncfatal(n, "savenc: closing %s failed", path);
	}

	name = fileprefix(path);
	printf("%s = loadnc(\"%s\")\n", name, path);
	free(name);
}

void
loadnc(const char *path, Mat &mat)
{
	int n, ncid;
	
	n = nc_open(path, NC_NOWRITE, &ncid);
	if(n != NC_NOERR){
		ncfatal(n, "loadnc: opening %s failed", path);
	}
	readvar(ncid, "data", mat);
	nc_close(n);
}

// Print out error for NetCDF error number n and exit the program.
void
ncfatal(int n, const char *fmt, ...)
{
	va_list args;

	fflush(stdout);
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);

	fprintf(stderr, ": %s\n", nc_strerror(n));
	exit(2);
}

// Open NetCDF file at path and initialize r.
int
open_resampled(const char *path, Resample &r, int omode)
{
	int ncid, n;
	Mat lat, acspo;
	
	n = nc_open(path, omode, &ncid);
	if(n != NC_NOERR){
		ncfatal(n, "nc_open failed for %s", path);
	}
	readvar(ncid, "acspo_mask", acspo);
	readvar(ncid, "latitude", lat);
	
	resample_init(r, lat, acspo);
	return ncid;
}

// Read a variable named name from NetCDF dataset ncid,
// resample the image if necessary and return it.
Mat
readvar_resampled(int ncid, Resample &r, const char *name)
{
	Mat img;
	
	if(strcmp(name, "latitude") == 0){
		return r.slat;
	}
	if(strcmp(name, "acspo_mask") == 0){
		return r.sacspo;
	}

	readvar(ncid, name, img);
	if(strcmp(name, "longitude") == 0
	|| strcmp(name, "sst_reynolds") == 0){
		resample_sort(r.sind, img);
		return img;
	}
	
	logprintf("resampling %s...\n", name);
	resample_float32(r, img, img, true);
	return img;
}

// Create a new variable and write data into it.
//
void
createvar(int ncid, const char *varname, const char *varunits, const char *vardescr, const Mat &data)
{
	int i, n, varid, ndims, dimids[2];
	nc_type xtype, xtype1;
	size_t len;
	
	xtype = -1;
	switch(data.depth()){
	default:
		eprintf("createvar: unsupported type %s\n", type2str(data.type()));
		break;
	case CV_8U:	xtype = NC_UBYTE; break;
	case CV_8S:	xtype = NC_BYTE; break;
	case CV_16U:	xtype = NC_USHORT; break;
	case CV_16S:	xtype = NC_SHORT; break;
	case CV_32S:	xtype = NC_INT; break;
	case CV_32F:	xtype = NC_FLOAT; break;
	case CV_64F:	xtype = NC_DOUBLE; break;
	}

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
		
		n = nc_def_var(ncid, varname, xtype, nelem(dimids), dimids, &varid);
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
		
		n = nc_put_att_text(ncid, varid, "UNITS", strlen(varunits), varunits);
		if(n != NC_NOERR){
			ncfatal(n, "setting attribute UNITS failed");
		}
		n = nc_put_att_text(ncid, varid, "Description", strlen(vardescr), vardescr);
		if(n != NC_NOERR){
			ncfatal(n, "setting attribute Description failed");
		}
	}
	
	// Varify that the netcdf variable has correct type and dimensions.
	n = nc_inq_var(ncid, varid, nullptr, &xtype1, &ndims, dimids, nullptr);
	if(n != NC_NOERR){
		ncfatal(n, "nc_inq_var failed");
	}
	if(xtype1 != xtype){
		eprintf("variable type is %d, want %d\n", xtype1, xtype);
	}
	if(ndims != 2){
		eprintf("variable dims is %d, want 2\n", ndims);
	}
	for(i = 0; i < 2; i++){
		n = nc_inq_dimlen(ncid, dimids[i], &len);
		if(n != NC_NOERR){
			ncfatal(n, "nc_inq_dimlen failed");
		}
		if(len != static_cast<size_t>(data.size[i])){
			eprintf("dimension %d is %d, want %d\n", i, len, data.size[i]);
		}
	}
	
	// Write data into netcdf variable.
	n = nc_put_var(ncid, varid, data.data);
	if(n != NC_NOERR){
		ncfatal(n, "nc_putvar_uchar failed");
	}
}

void
writevar(int ncid, const char *varname, const Mat &data)
{
	int n, varid, ndims, dimids[2];
	nc_type xtype;
	size_t len;
	
	CV_Assert(data.isContinuous());

	n = nc_inq_varid(ncid, varname, &varid);
	if(n != NC_NOERR){
		ncfatal(n, "nc_inq_varid failed");
	}

	// Varify that the netcdf variable has correct type and dimensions.
	n = nc_inq_var(ncid, varid, nullptr, &xtype, &ndims, dimids, nullptr);
	if(n != NC_NOERR){
		ncfatal(n, "nc_inq_var failed");
	}
	switch(xtype){
	default:
		eprintf("invalid variable type %d\n", xtype);
		break;
	case NC_UBYTE:
		if(data.type() != CV_8UC1){
			eprintf("invalid Mat type %s", type2str(data.type()));
		}
		break;
	case NC_FLOAT:
		if(data.type() != CV_32FC1){
			eprintf("invalid Mat type %s", type2str(data.type()));
		}
		break;
	}
	if(ndims != 2){
		eprintf("variable dims is %d, want 2\n", ndims);
	}
	for(int i = 0; i < 2; i++){
		n = nc_inq_dimlen(ncid, dimids[i], &len);
		if(n != NC_NOERR){
			ncfatal(n, "nc_inq_dimlen failed");
		}
		if(len != static_cast<size_t>(data.size[i])){
			eprintf("dimension %d is %d, want %d\n", i, len, data.size[i]);
		}
	}

	// Write data into netcdf variable.
	n = nc_put_var(ncid, varid, data.data);
	if(n != NC_NOERR){
		ncfatal(n, "nc_put_var failed");
	}
}



ACSPOFile::ACSPOFile(const char *path, int omode, bool _resample)
{
	if(_resample){
		ncid = open_resampled(path, r, omode);
	}else{
		int n = nc_open(path, omode, &ncid);
		if(n != NC_NOERR){
			ncfatal(n, "nc_open failed for %s", path);
		}
	}
	resample = _resample;
}

void
ACSPOFile::_readvar(const char *name, Mat &data, int type)
{
	if(resample){
		data = readvar_resampled(ncid, r, name);
	}else{
		::readvar(ncid, name, data);
	}
	CHECKMAT(data, type);
}


template <> void
ACSPOFile::readvar<float>(const char *name, Mat1f &data)
{
	_readvar(name, data, CV_32FC1);
}

template <> void
ACSPOFile::readvar<uchar>(const char *name, Mat1b &data)
{
	_readvar(name, data, CV_8UC1);
}

void
ACSPOFile::close()
{
	int n = nc_close(ncid);
	if(n != NC_NOERR){
		ncfatal(n, "nc_close failed");
	}
}
