// shwfs_one.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "tiffio.h"
#include "fftw3.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;

int readTIFFHeader(char* fn, uint32 *nx, uint32 *ny);
int readTIFFImage(char* fn, uint16 *img, int nx, int ny);
int getStats(uint16 *baseimg, int nx, int ny, double *max, double *min, double *mean);
int findDots(uint16 *baseimg, int nx, int ny, int hsp, int *loc_x, int *loc_y, int N, int *Nspots);
int getGrads(uint16 *img, int nx, int ny, int hsp, int *loc_x, int *loc_y, int N, double *gradx, double *grady);
int getMask(int *loc_x, int *loc_y, int N, int sz);
int gradsFill(double *phix, double *phiy, int *loc_x, int *loc_y, int N, int sz, double *gradx0, double *grady0, double *gradx, double *grady);
int writetiff32(char* fn, double *img, uint32 width, uint32 height);
int hudgins_extend(double *phix, double *phiy, int sz);
int shiftdouble(double *data, int N);

const double spacing = 22.75;

int main(int argc, char *argv[])
{
	// get start time
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	// constants related to the SHWFS details
	const int half_lenslet = 11; // in pixels
	const int N = 850; // number of spaces in arrays
	const int mask_size = 32; // size of final wavefront image
	const int N_mask = 1024;
	// other variables
	int ret;
	uint32 nx, ny; // image size
	int N_spots = 0;
	int loc_x[N], loc_y[N];
	double gradx0[N], grady0[N];
	double gradx[N], grady[N];
	double phix[N_mask];
	double phiy[N_mask];
	if (argc != 3)
	{
		cout << "Wrong number of arguments\n";
		return -1;
	}
	ret = readTIFFHeader(argv[1], &nx, &ny);
	if (ret == -1) cout << "Couldn't open file\n";
	//if (ret == -2) cout << "Image is not uint16\n";
	//if (ret != 0) return -1;
	// now read image
	uint16 *img = NULL;
	img = (uint16 *)malloc(nx*ny * sizeof(uint16));
	cout << "Read image\n";
	readTIFFImage(argv[1], img, nx, ny);
	// read offset image
	uint16 *offsetimg = NULL;
	offsetimg = (uint16 *)malloc(nx*ny * sizeof(uint16));
	readTIFFImage(argv[2], offsetimg, nx, ny);
	// do stuff
	findDots(img, nx, ny, half_lenslet, loc_x, loc_y, N, &N_spots);
	cout << "spots: " << N_spots << endl;
	cout << "first spot: " << loc_x[0] << ", " << loc_y[0] << endl;
	getGrads(img, nx, ny, half_lenslet, loc_x, loc_y, N_spots, gradx0, grady0);
	cout << "first spot: " << gradx0[0] << ", " << grady0[0] << endl;
	getMask(loc_x, loc_y, N_spots, mask_size);
	getGrads(offsetimg, nx, ny, half_lenslet, loc_x, loc_y, N_spots, gradx, grady);
	gradsFill(phix, phiy, loc_x, loc_y, N_spots, mask_size, gradx0, grady0, gradx, grady);
	//hudgins_extend(phix, phiy, mask_size);
	//shiftdouble (phix, mask_size);
	writetiff32("phix.tif", phix, mask_size, mask_size);
	// cleanup
	free(img);
	free(offsetimg);
	// get end time
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	//std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	//std::cout << "finished computation at " << std::ctime(&end_time)
	cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    return 0;
}

int readTIFFHeader(char* fn, uint32 *nx, uint32 *ny)
{
	TIFF* tif = TIFFOpen(fn, "r");
	if (tif) {
		uint16 datatype = 0, bitspersample = 0, sampleformat = 0, samplesperpixel = 0;
		uint32 imagelength = 0, imagewidth = 0, imagedepth = 0;

		TIFFGetField(tif, TIFFTAG_DATATYPE, &datatype);
		TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitspersample);
		TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sampleformat);
		TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesperpixel);
		TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imagelength);
		TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imagewidth);
		TIFFGetField(tif, TIFFTAG_IMAGEDEPTH, &imagedepth);

		cout << "Datatype: " << datatype << " Bits per sample: " << bitspersample << endl;
		cout << "Sample format: " << sampleformat << " Samples per pixel: " << samplesperpixel << endl;
		cout << "Length: " << imagelength << " Width: " << imagewidth << " Depth: " << imagedepth << endl;
		*ny = imagewidth;
		*nx = imagelength;
		TIFFClose(tif);
		if (bitspersample == 16 && samplesperpixel == 0)
			return 0;
		else
			return -2;
	}
	else
		return -1;
}

int readTIFFImage(char* fn, uint16 *img, int nx, int ny)
{
	TIFF* tif = TIFFOpen(fn, "r");
	if (tif) {
		uint16 datatype = 0, bitspersample = 0, sampleformat = 0, samplesperpixel = 0;
		uint32 imagelength = 0, imagewidth = 0, imagedepth = 0;
		tsize_t scanline;
		tdata_t buf;
		uint32 row;
		uint32 col;

		TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imagelength);
		TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imagewidth);
		TIFFGetField(tif, TIFFTAG_IMAGEDEPTH, &imagedepth);

		scanline = TIFFScanlineSize(tif);
		buf = _TIFFmalloc(scanline);
		cout << imagelength << " " << scanline << "\n";
		for (row = 0; row < nx; row++) // max row = imagelength
		{
			TIFFReadScanline(tif, buf, row);
			memcpy(&img[row*ny], buf, scanline);
			//for (col = 0; col < 5; col++)
			//	cout << ((uint32*)buf)[col] << " ";
		}
		_TIFFfree(buf);
		TIFFClose(tif);
	}
	return 0;
}

int writetiff32(char* fn, double *img, uint32 width, uint32 height)
{
	// img is double to be compatible with ffts, needs to be written as 32 bit float to work with ImageJ
	// create float copy
	float *tempimg;
	tempimg = (float *)malloc(width * height * sizeof(float));
	for (int ii = 0; ii < (width*height); ++ii) tempimg[ii] = (float)img[ii];
	// TIFF header stuff
	TIFF *out = TIFFOpen(fn, "w");
	int sampleperpixel = 1;    // or 3 if there is no alpha channel, you should get a understanding of alpha in class soon.
							   // Now we need to set the tags in the new image file, and the essential ones are the following :
	TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);  // set the width of the image
	TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);    // set the height of the image
	TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, sampleperpixel);   // set number of channels per pixel
	TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 32);    // set the size of the channels
	TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
	TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // set the origin of the image.
																	//   Some other essential fields to set that you do not have to understand for now.
	TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

	// We will use most basic image data storing method provided by the library to write the data into 
	// the file, this method uses strips, and we are storing a line(row) of pixel at a time.
	// This following code writes the data from the uint32 array image into the file :

	tsize_t linebytes = sampleperpixel * width * sizeof(float);     // length in memory of one row of pixel in the image.
	cout << "linebytes " << linebytes << endl;

	// now we prepare for writing
	float *buf = NULL;        // buffer used to store the row of pixel information for writing to file
							   //    Allocating memory to store the pixels of current row
	if (TIFFScanlineSize(out) == linebytes)
		buf = (float *)_TIFFmalloc(linebytes);
	else
		buf = (float *)_TIFFmalloc(TIFFScanlineSize(out));

	// We set the strip size of the file to be size of one row of pixels
	TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, width*sampleperpixel));

	//Now writing image to the file one strip at a time
	for (uint32 row = 0; row < height; row++)
	{
		memcpy(buf, &tempimg[(height - row - 1)*width], linebytes);
		//memcpy(buf, (img+row*width), linebytes);
		// check the index here, and figure out why not using h*linebytes
		if (TIFFWriteScanline(out, buf, row, 0) < 0)
			break;
	}
	// Finally we close the output file, and destroy the buffer
	(void)TIFFClose(out);
	if (buf)
		_TIFFfree(buf);
	free(tempimg);
	return 0;
}

int writetiff16(char* fn, uint16 *img, uint32 width, uint32 height)
{
	TIFF *out = TIFFOpen(fn, "w");
	int sampleperpixel = 1;    // or 3 if there is no alpha channel, you should get a understanding of alpha in class soon.
							   // Now we need to set the tags in the new image file, and the essential ones are the following :
	TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);  // set the width of the image
	TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);    // set the height of the image
	TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, sampleperpixel);   // set number of channels per pixel
	TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 16);    // set the size of the channels
	TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
	TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // set the origin of the image.
																	//   Some other essential fields to set that you do not have to understand for now.
	TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

	// We will use most basic image data storing method provided by the library to write the data into 
	// the file, this method uses strips, and we are storing a line(row) of pixel at a time.
	// This following code writes the data from the uint32 array image into the file :

	tsize_t linebytes = sampleperpixel * width * sizeof(uint16);     // length in memory of one row of pixel in the image.
	cout << "linebytes " << linebytes << endl;

	uint16 *buf = NULL;        // buffer used to store the row of pixel information for writing to file
							   //    Allocating memory to store the pixels of current row
	if (TIFFScanlineSize(out) == linebytes)
		buf = (uint16 *)_TIFFmalloc(linebytes);
	else
		buf = (uint16 *)_TIFFmalloc(TIFFScanlineSize(out));

	// We set the strip size of the file to be size of one row of pixels
	TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, width*sampleperpixel));

	//Now writing image to the file one strip at a time
	for (uint32 row = 0; row < height; row++)
	{
		memcpy(buf, &img[(height - row - 1)*width], linebytes);
		//memcpy(buf, (img+row*width), linebytes);
		// check the index here, and figure out why not using h*linebytes
		if (TIFFWriteScanline(out, buf, row, 0) < 0)
			break;
	}
	// Finally we close the output file, and destroy the buffer
	(void)TIFFClose(out);
	if (buf)
		_TIFFfree(buf);
	return 0;
}

int getStats(uint16 *baseimg, int nx, int ny, double *max, double *min, double *mean)
{
	uint16 temp = 0;
	*max = 0.0;
	*min = UINT32_MAX;
	*mean = 0.0;
	for (int j=0; j<nx; j++)
		for (int k = 0; k < ny; k++)
		{
			temp = baseimg[j*ny + k];
			if (temp > *max) *max = temp;
			if (temp < *min) *min = temp;
			*mean += temp;
		}
	*mean = *mean / (nx*ny);
	cout << "The minimum is " << *min << endl;
	cout << "The maximum is " << *max << endl;
	cout << "The mean is " << *mean << endl;
	return 0;
}

int findDots(uint16 *baseimg, int nx, int ny, int hsp, int *loc_x, int *loc_y, int N, int *Nspots)
{
	double max, min, mean;
	double temp;
	uint16 px, spx, spy;
	getStats(baseimg, nx, ny, &max, &min, &mean);
	// find first spot
	for (int jj = 0; jj < nx; jj++)
	{
		for (int ii = 0; ii < ny; ii++)
		{
			px = baseimg[jj*ny + ii];
			if (px > 4 * mean)
			{
				// found a spot
				cout << "x coord " << jj << " and y coord " << ii << endl;
				spx = jj;
				spy = ii;
				break;
			}
		}
		if (px > 4 * mean) break;
	}
	// find all spots
	ofstream spotfile("spotfile.dat");
	int lx, ly;
	int spots = 0;
	int jj = spx-hsp;
	int ii = spy-hsp - 12*2*hsp;
	while (jj < (nx - hsp))
	{
		while (ii < (ny - hsp))
		{
			// search for spot
			temp = 0;
			for (int xx = (jj-hsp); xx < (jj + hsp); ++xx)
			{
				for (int yy = (ii-hsp); yy < (ii + hsp); ++yy)
				{
					px = baseimg[xx*ny + yy];
					if (px > temp)
					{
						// found a spot
						//cout << "x coord " << jj << " and y coord " << ii << endl;
						temp = px;
						lx = xx;
						ly = yy;
					}
				}
			}
			if (temp > 4 * mean)
			{
				//cout << temp << " at x coord " << lx << " and y coord " << ly << endl;
				spotfile << lx << ", " << ly << "\n";
				if (spots < N)
				{
					loc_x[spots] = lx;
					loc_y[spots] = ly;
				}
				++spots;
				ii = ly + 2*hsp;
				jj = lx;
			}
			else
			{
				ii = ii + 2 * hsp;
			}
		}
		jj = lx + 2 * hsp;
		ii = spy - hsp - 12 * 2 * hsp;
	}
	spotfile.close();
	cout << endl << "There are " << spots << " spots\n";
	*Nspots = spots;
	return 0;
}

int getGrads(uint16 *img, int nx, int ny, int hsp, int *loc_x, int *loc_y, int N, double *gradx, double *grady)
{
	double xc = 0, yc = 0;
	double intensity = 0, total_intensity = 0;
	int px = 0, py = 0;
	for (int mm = 0; mm < N; ++mm)
	{
		px = loc_x[mm];
		py = loc_y[mm];
		// now get center of mass
		xc = 0;
		yc = 0;
		total_intensity = 0;
		for (int xx = (px - hsp); xx < (px + hsp); ++xx)
		{
			for (int yy = (py - hsp); yy < (py + hsp); ++yy)
			{
				intensity = img[xx*ny + yy];
				xc += xx*intensity;
				yc += yy*intensity;
				total_intensity += intensity;
			}
		}
		xc = xc / total_intensity;
		yc = yc / total_intensity;
		//cout << px << " " << py << " " << xc << " " << yc << endl;  
		gradx[mm] = xc;
		grady[mm] = yc;
	}
	return 0;
}

int getMask(int *loc_x, int *loc_y, int N, int sz)
{
	int x_index = 0, y_index = 0;
	int min_x = INT_MAX, min_y = INT_MAX;
	uint16 *mask = NULL;
	mask = (uint16 *)malloc(sz * sz * sizeof(uint16));
	// initialize to 0
	for (int mm = 0; mm < (sz*sz); ++mm) mask[mm] = 0;
	// get min_x and min_y
	for (int mm = 0; mm < N; ++mm)
	{
		if (loc_x[mm] < min_x) min_x = loc_x[mm];
		if (loc_y[mm] < min_y) min_y = loc_y[mm];
	}
	//cout << min_x << " " << min_y << endl;
	for (int mm = 0; mm < N; ++mm)
	{
		x_index = (int)round((loc_x[mm] - min_x) / spacing);
		y_index = (int)round((loc_y[mm] - min_y) / spacing);
		//cout << x_index << " " << y_index << endl;
		if ((x_index < sz) && (y_index < sz)) mask[x_index*sz + y_index] = 1;
	}
	writetiff16("mask.tif", mask, sz, sz);
	free(mask);
	return 0;
}

int gradsFill(double *phix, double *phiy, int *loc_x, int *loc_y, int N, int sz, double *gradx0, double *grady0, double *gradx, double *grady)
{
	int x_index = 0, y_index = 0;
	// setup for indexing by getting minima
	int min_x = INT_MAX, min_y = INT_MAX;
	// initialize to 0
	for (int mm = 0; mm < (sz*sz); ++mm)
	{
		phix[mm] = 0;
		phiy[mm] = 0;
	}
	for (int mm = 0; mm < N; ++mm)
	{
		if (loc_x[mm] < min_x) min_x = loc_x[mm];
		if (loc_y[mm] < min_y) min_y = loc_y[mm];
	}
	for (int mm = 0; mm < N; ++mm)
	{
		x_index = (int)round((loc_x[mm] - min_x) / spacing);
		y_index = (int)round((loc_y[mm] - min_y) / spacing);
		phix[x_index*sz + y_index] = gradx0[mm] - gradx[mm];
		phiy[x_index*sz + y_index] = grady0[mm] - grady[mm];
	}
	return 0;
}

int hudgins_extend(double *phix, double *phiy, int sz)
{
	for (int jj = 0; jj < sz; ++jj)
	{
		for (int ii = sz / 2; ii < sz; ++ii)
		{
			if (phix[jj*sz + ii] == 0) phix[jj*sz + ii] = phix[jj*sz + ii - 1];
			if (phiy[ii*sz + jj] == 0) phiy[ii*sz + jj] = phiy[(ii - 1)*sz + jj];
		}
		for (int ii = sz / 2; ii >= 0; --ii)
		{
			if (phix[jj*sz + ii] == 0) phix[jj*sz + ii] = phix[jj*sz + ii + 1];
			if (phiy[ii*sz + jj] == 0) phiy[ii*sz + jj] = phiy[(ii + 1)*sz + jj];
		}
	}
	for (int jj = 0; jj < sz; ++jj)
	{
		phiy[jj*sz + (sz - 1)] = 0;
		for (int ii = 0; ii < (sz - 1); ++ii)
			phiy[jj*sz + (sz - 1)] -= phiy[jj*sz + ii];
		phix[(sz - 1)*sz + jj] = 0;
		for (int ii = 0; ii < (sz - 1); ++ii)
			phix[(sz - 1)*sz + jj] -= phix[ii*sz + jj];
	}
	return 0;
}

int recon_hudgins(double *phix, double *phiy, int sz)
{
	// first take fourier transforms of phix and phiy
	shiftdouble(phix, sz);
	shiftdouble(phiy, sz);
	int N = sz*sz;
	double *in;
	fftw_complex *out;
	fftw_plan p;
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	p = fftw_plan_dft_r2c_2d(sz, sz, in, out, FFTW_ESTIMATE);
	fftw_execute(p); /* repeat as needed */
	// cleanup
	fftw_destroy_plan(p);
	fftw_free(in); fftw_free(out);
	return 0;
}

int shiftdouble(double *data, int N)
{
	double *tempdata;
	tempdata = (double *)malloc(N * N * sizeof(double));
	int iis = 0, jjs = 0;
	for (int ii = 0; ii < N; ++ii)
	{
		iis = (ii + N / 2) % N;
		for (int jj = 0; jj < N; ++jj)
		{
			jjs = (jj + N / 2) % N;
			tempdata[iis*N + jjs] = data[ii*N + jj];
		}
	}
	// copy to data
	for (int ii = 0; ii < (N*N); ++ii)
		data[ii] = tempdata[ii];
	// free memory
	free(tempdata);
	return 0;
}