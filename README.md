## SST Pattern Test

This is a work-in-progress rewrite of the implementation described in
this publication:

	Gladkova, I., Kihai, Y., Ignatov, A., Shahriar, F., & Petrenko, B. (2015).
	SST Pattern Test in ACSPO clear-sky mask for VIIRS. Remote Sensing of
	Environment.

### Dependencies

This program requires a C++ toolchain (e.g. GCC), NetCDF C library,
and OpenCV 2.x.

On Debian/Ubuntu, you can install them by running:

	apt-get install build-essential libnetcdf-dev libopencv-dev

### Usage

Build the program executable `spt` by running `make`. The program takes
the ACSPO granule as its only argument. After running it, it'll save
the output in a new variable named `spt_mask` in the granule. For example:

	$ make -j		# build the program
	...
	$ f=ACSPO_V2.31b02_NPP_VIIRS_2014-10-30_0400-0409_20141111.005748.nc
	$ cp $f SPT_$f	# make a copy of the granule
	$ ./spt SPT_$f	# run program
	...
	$ ncdump -h SPT_$f | grep spt_mask
	        ubyte spt_mask(scan_lines_along_track, pixels_across_track) ;
	                spt_mask:UNITS = "none" ;
	                spt_mask:Description = "SPT mask packed into 1 byte: bits1-2 (00=clear; 01=probably clear; 10=cloudy; 11=clear-sky mask undefined); bit3 (0=no thermal front; 1=thermal front)" ;

### License

Unless otherwise noted, the source files are distributed under the MIT
license found in the LICENSE file.
