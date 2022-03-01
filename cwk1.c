//
// COMP/XJCO3221 Parallel Computation Coursework 1: OpenMP / Shared Memory Parallelism.
//


//
// Includes.
//

// Standard includes.
#include <stdio.h>
#include <stdlib.h>

// The OMP library.
#include <omp.h>

// The include file for this coursework. You may inspect this file, but should NOT alter it, as it will be replaced
// with a different version for assessment.
#include "cwk1_extra.h"
// This file includes the following definition for the Image struct that the rest of the code uses.
// struct Image {
// 	int width, height;			// Width (=no. columns) and height (=no. rows) of the image.
// 	int maxValue;				// The maximum greyscale value (the minimum is always 0).
// 	int** pixels;				// pixels[row][col] for the pixel with the given row and column.
// };


//
// The following functions are called by main() for each of the options.
//

// Constructs a 'negative' image in place, and saves as a new .pgm file.
void saveNegativeImage( struct Image *img )
{
	int row, col;
	#pragma omp parallel for collapse(2)
	for( row=0; row<img->height; row++ )
		for( col=0; col<img->width; col++ )
			img->pixels[row][col] = img->maxValue - img->pixels[row][col];

	// Save as "negative.pgm". You must call this function to save your final image.
	writeNegativeImage( img );
}

// Mirrors the image in-place, and outputs to a new .pgm file.
void saveMirrorImage( struct Image *img )
{
	// Your parallel implementation should go here.
	/*
	Create a mirror image in which the order of pixels within each row is reversed, so the image
	is flipped left–to–right. More precisely, for row row and column col, the corresponding
	pixel value pixels[row][col] should be replaced with pixels[row][width-col-1]. The
	resulting image is saved as mirror.pgm.
	*/

	int row, col, tmp;
	#pragma omp parallel for collapse(2) private(tmp)
	for( row=0; row<img->height; row++ )
		for( col=0; col<img->width/2; col++ )
		{
			tmp = img->pixels[row][col];
			img->pixels[row][col] = img->pixels[row][img->width-col-1];
			img->pixels[row][img->width-col-1] = tmp;
		}

	// Save as "mirror.pgm". You must call this function to save your final image.
	writeMirrorImage( img );
}

// Blurs the image and outputs to a new .pgm file.
void saveBlurredImage( struct Image *img )
{
	// Your parallel implementation should go here.
/*
Blurs the image and outputs as blurred.pgm. The blurring is to be performed by sweeping
through the image 10 times. For each sweep, the value of each pixel is replaced by the
average of the 4 pixels surrounding that pixel, i.e.
pixels[row][col]=(pixels[x-1][col]+pixels[x+1][col]+pixels[row][y-1]+pixels[row][y+1])/4;
Pixels at the edges of the image are not changed. Your parallel algorithm should follow
the red–black Gauss–Seidel pattern discussed at the end of Lecture 5.
*/
		
	int sweep, redBlack, row, col;

	for( sweep=0; sweep<10; sweep++ )
		for( redBlack=0; redBlack<2; redBlack++ )
			for( row=1; row<img->height-1; row++ )
				for( col=1; col<img->width-1; col++ )
				{
					if( col%2 == redBlack )
					{
						img->pixels[row][col] = (img->pixels[row-1][col] + img->pixels[row+1][col] + img->pixels[row][col-1] + img->pixels[row][col+1])/4;
					}
				}

	// Save as "blurred.pgm". You must call this function to save your final image.
	writeBlurredImage( img );
}

// Constructs and outputs a histogram containing the number of each greyscale value in the image.
void generateHistogram( struct Image *img )
{
	// Initialise the histogram to zero ("calloc" rather than "malloc"). You do not need to parallelise this initialisation.
	int *hist = (int*) calloc( img->maxValue, sizeof(int) );

	// Loop through all pixels and add to the relevant histogram bin.
	int row, col;
	for( row=0; row<img->height; row++ )
		for( col=0; col<img->width; col++ )
		{
			int val = img->pixels[row][col];			// Double-check that the value is in the valid range.
			if( val>=0 && val<img->maxValue )
				hist[val]++;
		}
	
	// Save the histogram to file. There is a Python script you can use to visualise the results from this file.
	saveHistogram( hist, img->maxValue );

	// Free up the memory allocated for the histogram.
	free( hist );

	/*
	Generates a histogram of greyscale values; that is, an array that counts how many pixels
have greyscale value 0, how many have value 1, etc. up until maxValue. This is then saved
to a file histogram.dat to be displayed using a Python script plotHistogram.py
*/
}


//
// You should not modify the code in main(), but should understand how it works.
//
int main( int argc, char **argv )
{
    //
    // Parse command line arguments. Requires a filename and an option number.
    //

    // Make sure we have exactly 2 command line arguments, which, plus the executable name, means 'argc' should be exactly 3.
    if( argc != 3 )
    {
        printf( "Call with the name of the image file to read, and a single digit for the operation to perform:\n" );
		printf( "(1) Save a negative version of the image;\n(2) Save a mirror image;\n(3) Save a blurred image; or\n" );
		printf( "(4) Save a histogram of grey scale values in the image.\n" );

        return EXIT_FAILURE;
    }

    // Convert to an option number, and ensure it is in the valid range. Note argv[0] is the executable name.
    int option = atoi( argv[2] );
    if( option<1 || option>4 )
    {
        printf( "Option number '%s' invalid.\n", argv[2] );
        return EXIT_FAILURE;
    }
 
	// Attempts to open the file and return a structure for the image.
	struct Image img;
	readImage( argv[1], &img );
	printf( "Loaded a %ix%i image with a maximum greyscale value of %i.\n", img.width, img.height, img.maxValue );

	//
	// Perform the action based on the option provided on the command line.
	//
	printf( "Performing option '%i' using %i OpenMP thread(s).\n", option, omp_get_max_threads() );

	switch( option )
	{
		case 1 : saveNegativeImage( &img ); break;
		case 2 : saveMirrorImage  ( &img ); break;
		case 3 : saveBlurredImage ( &img ); break;
		case 4 : generateHistogram( &img ); break;
		default:
			return EXIT_FAILURE;
	}

	//
	// Finalise.
	//

	// Release memory allocated for the image.
	freeImage( &img );

    return EXIT_SUCCESS;
}
