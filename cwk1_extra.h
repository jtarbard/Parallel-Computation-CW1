//
// Extras file for the OpenMP coursework "orderedList". Do not change anything in this file,
// as it will be replaced with a different version for assessment.
//


//
// Includes.
//
#include <stdio.h>
#include <stdlib.h>


//
// The structure used to store the image and related information.
//
struct Image {
	int width, height;			// Width (=no. columns) and height (=no. rows) of the image.
	int maxValue;				// The maximum greyscale value (the minimum is always 0).
	int** pixels;				// pixels[row][col] for the pixel with the given row and column.
};


//
// Routines for reading and writing .pgm files.
//

// Reads in a image file in PGM format (ASCII, greyscale) with the given filename, and fills in the 
// passed pointer to the Image struct. This will allocate memory for the pixels, which will need to
// be freed after usage.
void readImage( char *fname, struct Image *img )
{
	int row, col;			// Rows and columns of pixels in the image.

	// Try to open the image file.
	FILE *file = fopen( fname, "r" );
	if( !file )
	{
		printf( "Could not open image '%s' file for reading.\n", fname );
		return;
	}

	// The first line should be "P2"; provide some basic error checking.
	char header[10];
	fscanf( file, "%s", header );
	if( header[0]!='P' || header[1]!='2' )
	{
		printf( "File not greyscale ASCII pgm.\n" );
		return;
	}

	// The next two numbers are the image dimensions.
	fscanf( file, "%i %i", &img->width, &img->height );

	// Can now allocate the two-dimensional array.
	img->pixels = (int**) malloc( sizeof(int*) * img->height );
	for( col=0; col<img->height; col++ )
		img->pixels[col] = (int*) malloc( sizeof(int) * img->width );

	// The next number is the maximum allowed value in the grey scale, i.e. the value corresponding to 'white.'
	fscanf( file, "%i", &img->maxValue );

	// Read in all of the pixels, which are arranged in blocks of pixels for each row, hence 'row' is the inner loop.
	for( col=0; col<img->height; col++ )
		for( row=0; row<img->width; row++ )
			fscanf( file, "%i", &img->pixels[col][row] );

	// Close the file.
	fclose(file);	
}

// Frees up the resources associated with the passed image. Need to know the number of rows.
void freeImage( struct Image *img )
{
	int row;

	// Free up the memory for each row first of all.
	for( row=0; row<img->height; row++ ) free( img->pixels[row] );

	// Now free up the memory for all of the row pointers.
	free( img->pixels );
}

// Saves the given image to the given filename, in PGM format.
void writeImage( char *fname, struct Image *img )
{
	int row, col;

	// Try to open the file for writing.
	FILE *file = fopen( fname, "w" );
	if( !file )
	{
		printf( "Could not open the file '%s' for writing.\n", fname );
		return;
	}

	// The first lines specify the format (lien 1), width and height (line 2), and maximum greyscale value (line 3).
	fprintf( file, "P2\n" );
	fprintf( file, "%i %i\n", img->width, img->height );
	fprintf( file, "%i\n", img->maxValue );

	// Now output all of the geyscale values, pixel by pixel, with one line per row.
	for( col=0; col<img->height; col++ )
	{
		for( row=0; row<img->width; row++ ) fprintf( file, "%i ", img->pixels[col][row] );
		fprintf( file, "\n" );
	}

	// Message to stdout.
	printf( "Saving image to file '%s'.\n", fname );

	// Close the file.
	fclose(file);
}

//
// Output routines for this coursework.
//

// Outputs the negative image using a hard-coded filename.
void writeNegativeImage( struct Image *img ) { writeImage("negative.pgm",img); }

// Outputs the mirror image using a hard-coded filename.
void writeMirrorImage( struct Image *img ) { writeImage("mirror.pgm",img); }

// Outputs the blurred image using a hard-coded filename.
void writeBlurredImage( struct Image *img ) { writeImage("blurred.pgm",img); }

// Saves a histogram to file in a format that can be read by the associated Python script.
void saveHistogram( int *hist, int maxValue )
{
	int bin;

	// Try to open the file.
	FILE *file = fopen( "histogram.dat", "w" );
	if( !file )
	{
		printf( "Could not open the file 'histogram.dat' to output the histogram.\n" );
		return;
	}

	// Output one bin at a time.
	for( bin=0; bin<maxValue; bin++ )
		fprintf( file, "%i\t%i\n", bin, hist[bin] );
	
	// Message to stdout.
	printf( "Output greyscale histogram to 'histogram.dat'. Use the provided Python script to view.\n" );

	// Close the file.
	fclose( file );
}
