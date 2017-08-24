SPT = spt
CMDIFF = cmdiff
CXX = g++
CXXFLAGS = -g -Wall -Wpedantic -Wfatal-errors -Wunused-parameter -march=native -O2 -fopenmp -std=c++11
LD = g++
LDFLAGS_SPT =\
	-fopenmp\
	-lm\
	-lnetcdf\
	-lopencv_core\
	-lopencv_imgproc\
	-lopencv_flann\
	-lopencv_highgui\

LDFLAGS_CMDIFF =\
	-lnetcdf\
	-lopencv_core\
	-lopencv_imgproc\
	-lopencv_highgui\

OFILES_SPT = \
	utils.o\
	io.o\
	resample.o\
	connectedcomponents.o\
	filters.o\
	spt.o\

HFILES =\
	spt.h\
	connectedcomponents.h\
	fastBilateral.hpp\

LOC =\
	eigen-eigen-26667be4f70b\

all: $(SPT) tags

$(SPT): $(OFILES_SPT)
	$(LD) -o $(SPT) $(OFILES_SPT) $(LDFLAGS_SPT)


%.o: %.cc $(HFILES)
	$(CXX) -I $(LOC)  -c $(CXXFLAGS) $<

tags: $(OFILES_SPT:.o=.cc) $(HFILES)
	ctags -n $^

clean:
	rm -f $(SPT) \
		$(OFILES_SPT) \
		tags

tidy:
	clang-tidy -checks=*,-google-runtime-int,-cppcoreguidelines-pro-type-vararg,-cppcoreguidelines-pro-bounds-pointer-arithmetic,-cppcoreguidelines-pro-bounds-array-to-pointer-decay,-cppcoreguidelines-pro-bounds-constant-array-index,-cert-dcl50-cpp \
		filters.cc \
		io.cc \
		resample.cc \
		spt.cc \
		utils.cc \
		-- $(CXXFLAGS)
