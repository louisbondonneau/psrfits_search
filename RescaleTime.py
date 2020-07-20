#				RESCALE TIME
#	Resize data array extracted from a FITS file to a new number of rows.
#	Used to decrease the time lapse of a block for a involve the rfifind processing.


# MODULES

import numpy as np
import pyfits as fi
import sys
import argparse as arg
import os


# ARGUMENTS LIST

parser = arg.ArgumentParser( description = 'Resize the FITS file to decrease the time lapse of a row.' )

parser.add_argument( '-f' , dest='fileName' , type=str , help='Name of the FITS file to change.' )
parser.add_argument( '-n' , dest='n' , type=int , help='Dividing factor on samples per block. Equivalent as multiplying factor for the number of row. Must be a power of 2.' )
parser.add_argument( '-o' , dest='newFileName' , type=str , help='Name of the new FITS file to write.' )

args = parser.parse_args()


# CHECKING INPUT PARAMETERS

if os.path.isfile( args.fileName ) :		# Checking file existence
	print '\nExtraction of data from {:s}.\n'.format( args.fileName )
else :
	print '\n{:s} is not a file.\n'.format( args.fileName )
	sys.exit()

if args.n % 2 != 0 :				# Check if n is a power of 2
	print 'n is not a power of 2.\n'
	sys.exit()


if args.newFileName :				# Define the name of the new FITS file
	print 'Resized arrays writed in {:s}.\n'.format( args.newFileName )
else :
	print 'None new FITS file name defined. Default name used : new_{:s}.\n'.format( args.fileName )


# DATA EXTRACTION OF THE PREVIOUS FITS

headObs = fi.getheader( args.fileName , 0 , do_not_scale_image_data=True , scale_back=True )		# Extraction of the observation header
head = fi.getheader( args.fileName , 1 , do_not_scale_image_data=True , scale_back=True )		# Extraction of the data header
data = fi.getdata( args.fileName , do_not_scale_image_data=True , scale_back=True )			# Extraction of the data arrays

chan = head[ 'NCHAN' ]				# Number of frequency channel
samples = head[ 'NSBLK' ]			# Number of samples per block
blocks = head[ 'NAXIS2' ]			# Number of blocks
pol = head[ 'NPOL' ]				# Number of polarizations
bin = head[ 'NBIN' ]				# Number of bins ( normally 1 )
bits = head[ 'NBITS' ]				# Number of bits by datum
tsample = head['TBIN']				# Time lapse of a sample
lst = headObs[ 'STT_LST' ]


# COMPUTING OF THE NEW INFORMATION

newSamples = int( samples / args.n )			# Computing of the new number of samples per block
newBlocks = int( blocks * args.n )			# Computing of the new number of blocks
newSize = int( bin * chan * pol * newSamples )		# Computing of the new size of the data array
newTblock = float( tsample * newSamples )		# Computing of the new block time lapse
newLst = round( lst - tsample * samples / 2. + newTblock / 2. , 0 )


# WRITING MODIFIED LINES IN HEADERS

head[ 'NAXIS1' ] = newSize				# Writing of the new data array size
head[ 'NAXIS2' ] = newBlocks				# Writing of the new number of rows in the data fits
head[ 'TFORM17' ] = str( newSize ) + 'E'		# Writing of the new amplitude data array byte size
head[ 'NSBLK' ] = newSamples				# Writing of the number of samples per block in the new fits file
head[ 'TDIM17' ] = '(' + str( bin ) + ',' + str( chan ) + ',' + str( pol ) + ',' + str( newSamples ) +')'	# Writing of the new amplitude data array dimension
headObs[ 'STT_LST' ] = newLst				# Writing the new LST of the first block center

if newSize < bits :		# Checking if there are a number of samples greater than the number of bits
	print 'The new number of samples per block is less than {:d}.'.format( bits )
	print 'The minimal samples number is {:d}. Thus, n must be less or equal than {:d}.\n'.format( bits , samples / bits )
	sys.exit()


# RESIZING ARRAYS

colList = []				# Field list for the new fits file

	# Column of the time lapse of each block
oldArray = data.field( 0 )						# Copy of the old data array
oldCol = data.columns[ 0 ].copy()					# Copy of the old corresponding header
newArray = np.resize( oldArray / args.n , ( newBlocks , ) )		# Computing of the new values and resizing of the data array
newCol = fi.Column( name=oldCol.name , format=oldCol.format , unit=oldCol.unit , array=newArray )	# Creation of the new field
colList.append( newCol )						# Adding to the new field list

	# Column of the time offset of the center of each block
oldCol = data.columns[ 1 ].copy()					# Copy of the old corresponding header
newArray = np.arange( newTblock / 2. , tsample * samples * blocks , newTblock )			# Creation of the new block offset 1D array
newCol = fi.Column( name=oldCol.name , format=oldCol.format , unit=oldCol.unit , array=newArray )	# Creation of the new field
colList.append( newCol )						# Adding to the new field list

	# Column of the LST time of the center of each block
oldCol = data.columns[ 2 ].copy()					# Copy of the old corresponding header
newArray = np.around( np.linspace( newLst , newLst + newBlocks * newTblock , newBlocks ) , 0 )		# Creation of the new LST time 1D array
newCol = fi.Column( name=oldCol.name , format=oldCol.format , unit=oldCol.unit , array=newArray )	# Creation of the new field
colList.append( newCol )						# Adding to the new field list

for f in range( 3 , 12 ) :			# Loop on other 1D subint arrays

	oldArray = data.field( f )				# Copy of the old data array
	oldCol = data.columns[ f ].copy()			# Copy of the old corresponding header
	newArray = np.resize( oldArray , ( newBlocks , ) )	# Resizing of the data array
	newCol = fi.Column( name=oldCol.name , format=oldCol.format , unit=oldCol.unit , dim=oldCol.dim , array=newArray )	# Creation of the new field
	colList.append( newCol )				# Adding to the new field list

for f in range( 12 , 14 ) :			# Loop on 2D weight arrays

	oldArray = data.field( f )				# Copy of the old data array
	oldCol = data.columns[ f ].copy()			# Copy of the old corresponding headoer
	newArray = np.resize( oldArray , ( newBlocks , chan ) )	# Resizing of the data array
	newCol = fi.Column( name=oldCol.name , format=oldCol.format , unit=oldCol.unit , dim=oldCol.dim , array=newArray )	# Creation of the new field
	colList.append( newCol )				# Adding to the new field list

for f in range( 14 , 16 ) :			# Loop on 2D weight arrays

	oldArray = data.field( f )				# Copy of the old data array
	oldCol = data.columns[ f ].copy()			# Copy of the old corresponding headoer
	newArray = np.tile( oldArray , args.n ).reshape( ( newBlocks , chan * pol ) )		# Resizing of the data array
	newCol = fi.Column( name=oldCol.name , format=oldCol.format , unit=oldCol.unit , dim=oldCol.dim , array=newArray )	# Creation of the new field
	colList.append( newCol )				# Adding to the new field list

oldArray = data.field( 16 )						# Copy of the old amplitude data array
oldCol = data.columns[ 16 ].copy()					# Copy of the old corresponding header
newFormat = fi.column._ColumnFormat( str( newSize ) + 'E' )		# Definition of the new data array format
newDim = '(' + str( bin ) + ',' + str( chan ) + ',' + str( pol ) + ',' + str( newSamples ) + ')'		# Definition of the new data array definition
newArray = np.reshape( oldArray , ( newBlocks , newSamples , pol , chan , bin ) )				# Resizing of the data array
newCol = fi.Column( name=oldCol.name , format=newFormat , unit=oldCol.unit , dim=newDim , array=newArray )	# Creation of the new field
colList.append( newCol )						# Adding to the new field list


# DEFINITION OF THE NEW FITS

colDefs = fi.ColDefs( colList )					# Creation of the new fields object
tbhdu = fi.BinTableHDU.from_columns( colDefs , header=head )	# Creation of the new data table object

prihdu = fi.PrimaryHDU( header=headObs )			# Creation of the new observation header (exactly the same that the old fits file)
hdulist = fi.HDUList( [ prihdu , tbhdu ] )			# Creation of the new HDU object

hdulist.writeto( args.newFileName , output_verify='exception' )				# Writing the new HDU object on the new fits file
hdulist.close()
