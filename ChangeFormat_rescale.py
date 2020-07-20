#                RESCALE TIME
#    Resize data array extracted from a FITS file to a new number of rows.
#    Used to decrease the time lapse of a block for a involve the rfifind processing.


# MODULES

import numpy as np
import pyfits as fi
import sys
import argparse as arg
import os


# ARGUMENTS LIST

parser = arg.ArgumentParser( description = 'transforme 32 bits data to 8 bits using scales and offsets. Do not forget to use the option -scloffs in dspsr!!' )

parser.add_argument( '-f' , dest='fileName' , type=str , help='Name of the FITS file to change.' )
parser.add_argument( '-o' , dest='newFileName' , type=str , help='Name of the new FITS file to write.' )
parser.add_argument( '-ds' , dest='ds' , type=int , default=1, help='downsample value.' )
parser.add_argument( '-pscrunch', dest='pscrunch', action='store_true', default = False, help="scrunch the polarisation")
parser.add_argument( '-noscale', dest='noscale', action='store_true', default = False, help="force all scales to 1")
parser.add_argument( '-longscales', dest='longscales', action='store_true', default = False, help="used 32 bits scales and offset (replace 8 bit uint by a 32 bits float)")
parser.add_argument( '-notimevar', dest='notimevar', action='store_true', default = False, help="do not take in count the time dependency of the offset and the scale")
parser.add_argument( '-threshold' , dest='threshold' , type=int , default=6, help='Change the threshold value.' )
parser.add_argument( '-plot', dest='plot', action='store_true', default = False, help="plot statistics")

### log
#030220 fix mean_median_array during notimevar
#030220 add the option longscales to use 32 bits matrix for scales and offsets
#040320 fix npol not definded when pscrunch

args = parser.parse_args()

ds = int(2**(round(np.log(args.ds)/np.log(2))))
SIGMA = args.threshold
SIGMA = SIGMA*(2./3)
# CHECKING INPUT PARAMETERS

if os.path.isfile( args.fileName ) :        # Checking file existence
    print '\nExtraction of data from {:s}.\n'.format( args.fileName )
else :
    print '\n{:s} is not a file.\n'.format( args.fileName )
    sys.exit()


if args.newFileName :                # Define the name of the new FITS file
    print 'Scaled Integer arrays writed in {:s}.\n'.format( args.newFileName )
else :
    print 'None new FITS file name defined. Default name used : new_{:s}.\n'.format( args.fileName )

# DATA EXTRACTION OF THE PREVIOUS FITS

headObs = fi.getheader( args.fileName , 0 , do_not_scale_image_data=True , scale_back=True )        # Extraction of the observation header
head = fi.getheader( args.fileName , 1 , do_not_scale_image_data=True , scale_back=True )        # Extraction of the data header
data = fi.getdata( args.fileName , do_not_scale_image_data=True , scale_back=True )            # Extraction of the data arrays

print(data.columns)


old_offset = data.field( 14 ).astype('float32') 
old_scale = data.field( 15 ).astype('float32') 
old_data = data.field( 16 ).astype('float32')                      # Copy of the old amplitude data array


print(np.shape(old_offset))
print(np.shape(old_scale))
print(np.shape(old_data))


#print(data.field( 1 ) )
#print(len(data.field( 1 ) ))

# RESIZING ARRAYS

colList = []                # Field list for the new fits file


for i in range( 14 ) :
    oldArray = data.field( i )                   # Copy of the old amplitude data array
    oldCol = data.columns[ i ].copy()            # Copy of the old corresponding header
    print(i, oldCol.name, oldCol.format, oldCol.unit, oldCol.dim)
    newCol = fi.Column(name=oldCol.name,         # Creation of the new field
                        format=oldCol.format,
                        unit=oldCol.unit,
                        dim=oldCol.dim,
                        array=oldArray)
    colList.append( newCol )                     # Adding to the new field list

oldCol_offset = data.columns[ 14 ].copy()               # Copy of the old corresponding header
oldCol_scale = data.columns[ 15 ].copy()               # Copy of the old corresponding header
oldCol_data = data.columns[ 16 ].copy()               # Copy of the old corresponding header

head[ 'NBITS' ] = 8
npol = int(head['NPOL'])
if(args.pscrunch and npol > 1):
    if(args.longscales):
        head[ 'TFORM15' ] = str(int(float(head[ 'TFORM15' ][0:-1])/npol))+'E'
        head[ 'TFORM16' ] = str(int(float(head[ 'TFORM16' ][0:-1])/npol))+'E'
    else:
        head[ 'TFORM15' ] = str(int(float(head[ 'TFORM15' ][0:-1])/npol))+'B'
        head[ 'TFORM16' ] = str(int(float(head[ 'TFORM16' ][0:-1])/npol))+'B'
    head[ 'TFORM17' ] = str(int(float(head[ 'TFORM17' ][0:-1])/npol/ds))+'B'

    head['NPOL'] = 1
    head['POL_TYPE'] = 'AA+BB'
else:
    if(args.longscales):
        head[ 'TFORM15' ] = str(int(float(head[ 'TFORM15' ][0:-1])))+'E'
        head[ 'TFORM16' ] = str(int(float(head[ 'TFORM16' ][0:-1])))+'E'
    else:
        head[ 'TFORM15' ] = str(int(float(head[ 'TFORM15' ][0:-1])))+'B'
        head[ 'TFORM16' ] = str(int(float(head[ 'TFORM16' ][0:-1])))+'B'
    head[ 'TFORM17' ] = str(int(float(head[ 'TFORM17' ][0:-1])/ds))+'B'

newFormat_offset = fi.column._ColumnFormat( head[ 'TFORM15' ] )      # Definition of the new data array format
newFormat_scale = fi.column._ColumnFormat( head[ 'TFORM16' ] )      # Definition of the new data array format
newFormat_data = fi.column._ColumnFormat( head[ 'TFORM17' ] )      # Definition of the new data array format




##### calculate constantes
nline, line_lenght, npol, nchan = np.shape(old_data[:, :, :, :, 0])

##### extract values
old_scale = np.resize(old_scale,(nline, npol, nchan))
old_offset = np.resize(old_offset,(nline, npol, nchan))
old_data = np.resize(old_data,(nline, line_lenght, npol, nchan, 1))

for bin in range(line_lenght) :
    old_data[:, bin, :, :, 0] = (old_data[:, bin, :, :, 0]*old_scale + old_offset)


if (ds>1):
    head[ 'NSBLK' ] = int(head[ 'NSBLK' ])/ds
    head['TBIN'] = float(head['TBIN'])*ds
    old_data = np.resize(old_data,(nline, line_lenght/ds, ds, npol, nchan, 1))
    old_data = np.sum(old_data, axis=2)
    old_data = np.resize(old_data,(nline, line_lenght/ds, npol, nchan, 1))
    line_lenght = line_lenght/ds

##### calcul des std et median


if(args.pscrunch and npol > 1):
    print('---------pscrunch---------')
    old_data = np.sum(old_data[:, :, 0:1, :, :], axis=2)
    old_data = np.resize(old_data,(nline, line_lenght, 1, nchan, 1))
    npol = 1

print('---------calculate median_array---------')
median_array = np.median(old_data, axis=1)  # OFFSET

if not (args.noscale):
    print('---------calculate std_array---------')
    std_array = np.std(old_data, axis=1)        # SCAL
else:
    std_array = 0*median_array


if (args.notimevar):
    print(np.shape(median_array))
    print(np.shape(std_array))
    mean_median_array = np.median(median_array, axis=0)
    mean_std_array = np.median(std_array, axis=0)
    for line in range(nline):
        median_array[line, :, :, :] = mean_median_array
        std_array[line, :, :, :] = mean_std_array

OFFSET = median_array - 0.5*SIGMA*std_array
#The signal is between median_array-0.5*SIGMA*std and median_array+1.5*SIGMA*std

SCAL = 2.*SIGMA*std_array/256.

if not (args.longscales):
    saturation = np.where(OFFSET>255)
    SCAL[saturation] = (OFFSET[saturation]-255 + 2.*SIGMA*std_array[saturation])/256.
    SCAL = np.ceil(SCAL)
    
    OFFSET[np.where(OFFSET>255)] = 255
    OFFSET[np.where(OFFSET<0)] = 0
    
    #SCAL = np.ceil(SCAL)
    SCAL[np.where(SCAL>255)] = 255
    SCAL[np.where(SCAL<1)] = 1
    
    OFFSET = OFFSET.astype( 'uint8' )  #cast OFFSET matrix in a uint8 matrix
    SCAL = SCAL.astype( 'uint8' )      #cast SCAL matrix in a uint8 matrix


#####some plots
if (args.plot):
    print('---------make plot median-std.pdf---------')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(top=0.98,
                        bottom=0.07,
                        left=0.1,
                        right=0.980,
                        hspace=0.215,
                        wspace=0.25)
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    for i in range(npol):
        mean_med = np.mean(median_array[:, i,:], axis=0)
        ax1.semilogy(mean_med)
        ax1.set_xlabel('channel number')
        ax1.set_ylabel('median value')
        bins = np.logspace(np.log10(1),np.log10(np.max(mean_med)), 32)
        if(np.max(mean_med) < 10):
            bins = np.logspace(np.log10(1),np.log10(10), 32)
        ax3.hist(mean_med, bins=bins, alpha=0.3, log=True)
        ax3.set_xscale("log")
        ax3.set_xlabel('median value')
        ax3.set_ylabel('number of value')
    for i in range(npol):
        mean_std = np.mean(std_array[:, i,:], axis=0)
        ax2.semilogy(mean_std)
        ax2.set_xlabel('channel number')
        ax2.set_ylabel('standard deviation value')
        bins = np.logspace(np.log10(1),np.log10(np.max(mean_std)), 32)
        if(np.max(mean_std) < 10):
            bins = np.logspace(np.log10(1),np.log10(10), 32)
        ax4.hist(mean_std, bins=bins, alpha=0.3, log=True)
        ax4.set_xscale("log")
        ax4.set_xlabel('std')
        ax4.set_ylabel('number of value')
    plt.savefig('median-std.pdf')


#####some plots
if (args.plot):
    print('---------make plot scal-offset.pdf---------')
    plt.clf()
    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(top=0.98,
                        bottom=0.07,
                        left=0.1,
                        right=0.980,
                        hspace=0.215,
                        wspace=0.25)
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    for i in range(npol):
        mean_scal = np.mean(SCAL[:, i, :, 0], axis=0)
        ax2.semilogy(mean_scal)
        if not (args.longscales):
            ax2.axhline(256, color="r")
        ax2.set_xlabel('channel number')
        ax2.set_ylabel('scal')
        bins = np.logspace(np.log10(1),np.log10(np.max(mean_scal)), 32)
        if(np.max(mean_scal) < 10):
            bins = np.logspace(np.log10(1),np.log10(10), 32)
        ax4.hist(mean_scal, bins=bins, alpha=0.3, log=True)
        ax4.set_xscale("log")
        ax4.set_xlabel('scal')
        ax4.set_ylabel('number of value')
    for i in range(npol):
        mean_offset = np.mean(OFFSET[:, i, :, 0], axis=0)
        ax1.semilogy(mean_offset)
        if not (args.longscales):
            ax1.axhline(256, color="r")
        ax1.set_xlabel('channel number')
        ax1.set_ylabel('offset')
        bins = np.logspace(np.log10(1),np.log10(np.max(mean_offset)), 32)
        if(np.max(mean_offset) < 10):
            bins = np.logspace(np.log10(1),np.log10(10), 32)
        ax3.hist(mean_offset, bins=bins, alpha=0.3, log=True)
        ax3.set_xscale("log")
        ax3.set_xlabel('offset')
        ax3.set_ylabel('number of value')
    plt.savefig('scal-offset.pdf')
#

#####some plots
if (args.plot):
    print('---------make plot data.pdf---PART1------')
    plt.clf()
    spectrum = np.mean(median_array, axis=0)
    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(top=0.98,
                        bottom=0.07,
                        left=0.1,
                        right=0.980,
                        hspace=0.215,
                        wspace=0.25)
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    pol = ['XX', 'YY', 'XY', 'YX']
    for ipol in range(npol):
        ax1.semilogy(spectrum[ipol, :, 0], label=pol[ipol])
    ax1.set_xlabel('channel')
    ax1.set_ylabel('OLD Amplitude (AU)')
    ax1.legend(loc='upper right')
    ax3.hist(np.resize(old_data, len(old_data)), alpha=1, log=True)
    ax3.set_xlabel('OLD values')
    ax3.set_ylabel('number of value')



print('---------apply offset and scaling---------')
#####apply offset and scalingine*ipol*ichan, nline*npol*nchan, prefix = 'Progress:', suffix = 'Complete', barLength = 50)

for bin in range(line_lenght) :
    old_data[:, bin, :, :, :] = (old_data[:, bin, :, :, :] - OFFSET)/SCAL


if (args.plot):
    print('---------make plot data.pdf---PART2------')
    spectrum = np.median(old_data, axis=1)
    spectrum = np.mean(spectrum, axis=0)
    for ipol in range(npol):
        ax2.semilogy(spectrum[ipol, :, 0], label=pol[ipol])
    if not (args.longscales):
        ax2.axhline(256, color="r")
    ax2.set_xlabel('channel')
    ax2.set_ylabel('NEW Amplitude (AU)')
    ax2.legend(loc='upper right')
    ax4.hist(np.resize(old_data, len(old_data)), alpha=1, log=True)
    ax4.set_xlabel('NEW values')
    ax4.set_ylabel('number of value')
    plt.savefig('oldDATA_newDATA.pdf')




OFFSET = np.resize(OFFSET,(nline, npol, nchan))
SCAL = np.resize(SCAL,(nline, npol, nchan))


print(np.shape(OFFSET))
print(np.shape(SCAL))
print(np.shape(old_data))

### replace OFFSET and SCAL   '(1,'+str(nchan)+','+str(npol)+')'
newCol = fi.Column( name=oldCol_offset.name  , format=newFormat_offset , unit=oldCol_offset.unit , dim='(1,'+str(nchan)+','+str(npol)+')' , array=OFFSET )    # Creation of the new field
colList.append( newCol )
newCol = fi.Column( name=oldCol_scale.name , format=newFormat_scale , unit=oldCol_scale.unit , dim='(1,'+str(nchan)+','+str(npol)+')'  , array=SCAL )    # Creation of the new field
colList.append( newCol )


old_data[np.where(old_data>255)] = 255
old_data[np.where(old_data<0)] = 0

newCol = fi.Column( name=oldCol_data.name , format=newFormat_data , unit=oldCol_data.unit , dim='(1,'+str(nchan)+','+str(npol)+','+str(line_lenght)+')' , array=old_data.astype('uint8') )    # Creation of the new field
colList.append( newCol )                        # Adding to the new field list


# DEFINITION OF THE NEW FITS

print('---------save data to '+args.newFileName+' ---------')
colDefs = fi.ColDefs( colList )                    # Creation of the new fields object
tbhdu = fi.BinTableHDU.from_columns( colDefs , header=head )    # Creation of the new data table object

prihdu = fi.PrimaryHDU( header=headObs )            # Creation of the new observation header (exactly the same that the old fits file)
hdulist = fi.HDUList( [ prihdu , tbhdu ] )            # Creation of the new HDU object

hdulist.writeto( args.newFileName ) #output_verify='exception' )                # Writing the new HDU object on the new fits file
hdulist.close()
