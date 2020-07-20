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

parser = arg.ArgumentParser( description = 'merge 2 fits files in the frequency direction. And transforme the 32 bits data to 8 bits using scales and offsets. Do not forget to use the option -scloffs in dspsr!!' )

parser.add_argument( '-o' , dest='newFileName' , type=str , help='Name of the new FITS file to write.', default='dont_forget_the_name.fits' )
parser.add_argument( '-ds' , dest='ds' , type=int , default=1, help='downsample value.' )
parser.add_argument( '-pscrunch', dest='pscrunch', action='store_true', default = False, help="scrunch the polarisation")
parser.add_argument( '-noscale', dest='noscale', action='store_true', default = False, help="force all scales to 1")
parser.add_argument( '-longscales', dest='longscales', action='store_true', default = True, help="used 32 bits scales and offset (replace 8 bit uint by a 32 bits float)")
parser.add_argument( '-notimevar', dest='notimevar', action='store_true', default = False, help="do not take in count the time dependency of the offset and the scale")
parser.add_argument( '-norescaloff', dest='norescaloff', action='store_true', default = False, help="do not calculat new scale and offset will use those from fits files, this option should be used only on 8 bits files")
parser.add_argument( '-threshold' , dest='threshold' , type=int , default=6, help='Change the threshold value (default threshold = 6).' )
parser.add_argument( '-plot', dest='plot', action='store_true', default = False, help="plot statistics")
parser.add_argument( 'INPUT_ARCHIVE', nargs='+', help="Name of the FITS files to merge.")

### log
#030220 fix mean_median_array during notimevar
#030220 add the option longscales to use 32 bits matrix for scales and offsets
#040320 add the option 'norescaloff' do not calculat new scale and offset will use those from fits files

args = parser.parse_args()


def extract_data_array(fileName):
    old_data, old_scale, old_offset = extract_data_scales_offsets(fileName)
    old_data = old_data.astype('float32')
    old_scale = old_scale.astype('float32')
    old_offset = old_offset.astype('float32')
    for bin in range(line_lenght) :
        old_data[:, bin, :, :, 0] = (old_data[:, bin, :, :, 0]*old_scale + old_offset)
    return old_data

def extract_data_scales_offsets(fileName):
    data = fi.getdata( fileName , do_not_scale_image_data=True , scale_back=True )            # Extraction of the data arrays
    old_offset = data.field( 14 ) 
    old_scale = data.field( 15 ) 
    old_data = data.field( 16 )                      # Copy of the old amplitude data array
    ##### calculate constantes
    nline, line_lenght, npol, nchan = np.shape(old_data[:, :, :, :, 0])
    ##### extract values
    old_scale = np.resize(old_scale,(nline, npol, nchan))
    old_offset = np.resize(old_offset,(nline, npol, nchan))
    old_data = np.resize(old_data,(nline, line_lenght, npol, nchan, 1))
    return old_data, old_scale, old_offset

def freqarray_from_name(fileName):
    data = fi.getdata( fileName , do_not_scale_image_data=True , scale_back=True )            # Extraction of the data arrays
    #for i in range(60):
    #    print(i,np.shape(data.field(i)))
    #exit(0)
    freq_array = data.field(12)
    return freq_array

def get_other_array_from_name(fileName):
    data = fi.getdata( fileName , do_not_scale_image_data=True , scale_back=True )            # Extraction of the data arrays
    TSUBINT_array = data.field(0)
    OFFS_SUB_array = data.field(1)
    LST_SUB_array = data.field(2)
    RA_SUB_array = data.field(3)
    DEC_SUB_array = data.field(4)
    GLON_SUB_array = data.field(5)
    GLAT_SUB_array = data.field(6)
    FD_ANG_array = data.field(7)
    POS_ANG_array = data.field(8)
    PAR_ANG_array = data.field(9)
    TEL_AZ_array = data.field(10)
    TEL_ZEN_array = data.field(11)
    return (TSUBINT_array, OFFS_SUB_array, LST_SUB_array, RA_SUB_array, DEC_SUB_array, GLON_SUB_array, GLAT_SUB_array, FD_ANG_array, POS_ANG_array, PAR_ANG_array, TEL_AZ_array, TEL_ZEN_array)

def weightarray_from_name(fileName):
    data = fi.getdata( fileName , do_not_scale_image_data=True , scale_back=True )            # Extraction of the data arrays
    weight_array = data.field(13)
    return weight_array

def minchan_from_name(fileName):
    data = fi.getdata( fileName , do_not_scale_image_data=True , scale_back=True )            # Extraction of the data arrays
    headObs = fi.getheader( fileName , 0 , do_not_scale_image_data=True , scale_back=True )        # Extraction of the observation header
    freq_array = data.field(12)[0,:]
    bw = headObs[30]
    nchan = headObs[31]
    minchan = np.min(freq_array)/(bw/nchan)
    return minchan

def nchan_from_name(fileName):
    headObs = fi.getheader( fileName , 0 , do_not_scale_image_data=True , scale_back=True )        # Extraction of the observation header
    nchan = headObs[31]
    return nchan

def bw_from_name(fileName):
    headObs = fi.getheader( fileName , 0 , do_not_scale_image_data=True , scale_back=True )        # Extraction of the observation header
    bw = headObs[30]
    return bw

def tsample_from_name(fileName):
    head = fi.getheader( fileName , 1 , do_not_scale_image_data=True , scale_back=True )        # Extraction of the data header
    tsample = head[47]
    return tsample

def nsubint_from_name(fileName):
    head = fi.getheader( fileName , 1 , do_not_scale_image_data=True , scale_back=True )        # Extraction of the data header
    nsubint = head[4]
    return nsubint

def imjd_from_name(fileName):
    headObs = fi.getheader( fileName , 0 , do_not_scale_image_data=True , scale_back=True )        # Extraction of the observation header
    STT_IMJD = float(headObs[53])
    #for i in range(60):
    #    print(i,headObs[i])
    #exit(0)
    return STT_IMJD

def smjd_from_name(fileName):
    headObs = fi.getheader( fileName , 0 , do_not_scale_image_data=True , scale_back=True )        # Extraction of the observation header
    STT_SMJD = float(headObs[54])+float(headObs[55])
    return STT_SMJD

def print_headObs(headObs):
    param_length = 80
    headObs_str = str(headObs)
    headObs_length = len(headObs_str) - len(headObs_str)%param_length
    beg_param = np.linspace(0, headObs_length-param_length, (headObs_length/param_length))
    end_param = np.linspace(param_length, headObs_length, (headObs_length/param_length))
    for i in range(len(beg_param)):
        print(headObs_str[int(beg_param[i]):int(end_param[i])])

def print_headObs_from_file(fileName):
    headObs = fi.getheader( fileName , 0 , do_not_scale_image_data=True , scale_back=True )        # Extraction of the observation header
    print_headObs(headObs)  

def data_to_offsets_and_scales(old_data):
    ds = int(2**(round(np.log(args.ds)/np.log(2))))
    SIGMA = args.threshold
    SIGMA = SIGMA*(2./3)
    ##### calculate constantes
    nline, line_lenght, npol, nchan = np.shape(old_data[:, :, :, :, 0])
    if (ds>1):
        old_data = np.resize(old_data,(nline, line_lenght/ds, ds, npol, nchan, 1))
        old_data = np.sum(old_data, axis=2)
        old_data = np.resize(old_data,(nline, line_lenght/ds, npol, nchan, 1))
        line_lenght = line_lenght/ds

    if(args.pscrunch and npol > 1):
        print('---------pscrunch---------')
        old_data = np.sum(old_data[:, :, 0:1, :, :], axis=2)
        old_data = np.resize(old_data,(nline, line_lenght, 1, nchan, 1))
        npol = 1
    
    ##### calcul des std et median

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

    return (old_data, SCAL, OFFSET)

def fits_maker(fits0, TSUBINT_array, OFFS_SUB_array,
               LST_SUB_array, RA_SUB_array, DEC_SUB_array,
               GLON_SUB_array, GLAT_SUB_array, FD_ANG_array,
               POS_ANG_array, PAR_ANG_array, TEL_AZ_array,
               TEL_ZEN_array, DAT_FREQ, DAT_WTS,
               DAT_OFFS, DAT_SCL, DATA):
    head = fi.getheader( fits0 , 1 , do_not_scale_image_data=True , scale_back=True )        # Extraction of the data header
    nline, line_lenght, npol, nchan = np.shape(DATA[:, :, :, :, 0])
    args.ds
    args.pscrunch
    args.longscales

    array_list = [TSUBINT_array, OFFS_SUB_array,
                  LST_SUB_array, RA_SUB_array, DEC_SUB_array,
                  GLON_SUB_array, GLAT_SUB_array, FD_ANG_array,
                  POS_ANG_array, PAR_ANG_array, TEL_AZ_array,
                  TEL_ZEN_array]

    colList = []
    for i in range( 12 ) :
        oldArray = data.field( i )                   # Copy of the old amplitude data array
        oldCol = data.columns[ i ].copy()            # Copy of the old corresponding header
        print(i, oldCol.name, oldCol.format, oldCol.unit, oldCol.dim)
        newCol = fi.Column(name=oldCol.name,         # Creation of the new field
                           format=oldCol.format,
                           unit=oldCol.unit,
                           dim=oldCol.dim,
                           array=array_list[i])
        colList.append( newCol )

    oldCol_freq = data.columns[ 12 ].copy()               # Copy of the old corresponding header
    oldCol_weight = data.columns[ 13 ].copy()               # Copy of the old corresponding header
    oldCol_offset = data.columns[ 14 ].copy()               # Copy of the old corresponding header
    oldCol_scale = data.columns[ 15 ].copy()               # Copy of the old corresponding header
    oldCol_data = data.columns[ 16 ].copy()               # Copy of the old corresponding header

    head[ 'NSBLK' ] = line_lenght

    head['TBIN'] = float(head['TBIN'])*args.ds


    head[ 'NBITS' ] = 8
    head[ 'NPOL' ] = npol
    head[ 'NCHAN' ] = nchan
    head[ 'TFORM13' ] = str(int(nchan))+'E' #freq
    head[ 'TFORM14' ] = str(int(nchan))+'E' #weight
    if(args.pscrunch):
        head['POL_TYPE'] = 'AA+BB'

    if(DAT_SCL.dtype == np.dtype('float32')):
        head[ 'TFORM15' ] = str(int(nchan*npol))+'E'
        head[ 'TFORM16' ] = str(int(nchan*npol))+'E'
    else:
        head[ 'TFORM15' ] = str(int(nchan*npol))+'B'
        head[ 'TFORM16' ] = str(int(nchan*npol))+'B'
    head[ 'TFORM17' ] = str(int(nchan*npol*line_lenght/float(args.ds)))+'B'
    head['TDIM17'] = str('(1,'+str(nchan)+','+str(npol)+','+str(line_lenght)+')')

    newFormat_freq = fi.column._ColumnFormat( head[ 'TFORM13' ] )      # Definition of the new data array format
    newFormat_weight = fi.column._ColumnFormat( head[ 'TFORM14' ] )      # Definition of the new data array format

    newFormat_offset = fi.column._ColumnFormat( head[ 'TFORM15' ] )      # Definition of the new data array format
    newFormat_scale = fi.column._ColumnFormat( head[ 'TFORM16' ] )      # Definition of the new data array format
    newFormat_data = fi.column._ColumnFormat( head[ 'TFORM17' ] )      # Definition of the new data array format

    newCol = fi.Column( name=oldCol_freq.name  , format=newFormat_freq , unit=oldCol_freq.unit , dim='(1,'+str(nchan)+')' , array=DAT_FREQ )    # Creation of the new field
    colList.append( newCol )
    newCol = fi.Column( name=oldCol_weight.name , format=newFormat_weight , unit=oldCol_weight.unit , dim='(1,'+str(nchan)+')'  , array=DAT_WTS )    # Creation of the new field
    colList.append( newCol )
    newCol = fi.Column( name=oldCol_offset.name  , format=newFormat_offset , unit=oldCol_offset.unit , dim='(1,'+str(nchan)+','+str(npol)+')' , array=DAT_OFFS )    # Creation of the new field
    colList.append( newCol )
    newCol = fi.Column( name=oldCol_scale.name , format=newFormat_scale , unit=oldCol_scale.unit , dim='(1,'+str(nchan)+','+str(npol)+')'  , array=DAT_SCL )    # Creation of the new field
    colList.append( newCol )
    
    
    DATA[np.where(DATA>255)] = 255
    DATA[np.where(DATA<0)] = 0
    
    newCol = fi.Column( name=oldCol_data.name , format=newFormat_data , unit=oldCol_data.unit , dim='(1,'+str(nchan)+','+str(npol)+','+str(line_lenght)+')' , array=DATA.astype('uint8') )    # Creation of the new field
    colList.append( newCol )                        # Adding to the new field list


    headObs = fi.getheader( fits0 , 0 , do_not_scale_image_data=True , scale_back=True )        # Extraction of the observation header
    headObs[ 'OBSBW' ] = int(nchan)*(headObs[ 'OBSBW' ]/headObs[ 'OBSNCHAN' ])
    headObs[ 'OBSFREQ' ] = np.mean(DAT_FREQ[0,:])
    headObs[ 'OBSNCHAN' ] = int(nchan)
    #STT_IMJD
    #STT_SMJD
    #STT_OFFS
    #STT_LST
    print_headObs(head)
    print_headObs(headObs)

    return (colList, head, headObs)

# CHECKING INPUT PARAMETERS

if(len(args.INPUT_ARCHIVE)<2):
    print('\nNeed at least tow FITS files to work.\n')
    sys.exit()


# DATA EXTRACTION OF THE PREVIOUS FITS
imjd = []
smjd = []
tsample = []
nsubint = []
minchan = []
nchan = []
bw = []

for file in args.INPUT_ARCHIVE:
    minchan.append(minchan_from_name(file))
    if os.path.isfile( file ) :        # Checking file existence
        print('\nOppen the observation {:s}.\n'.format( file ))
        print_headObs_from_file(file)
    else :
        print('\n{:s} is not a file.\n'.format( file ))
        sys.exit()

file_liste = []
for i in range(len(args.INPUT_ARCHIVE)):
    file_liste.append(args.INPUT_ARCHIVE[np.argsort(minchan)[i]])
minchan = []


for file in file_liste:
    imjd.append(imjd_from_name(file))
    smjd.append(smjd_from_name(file))
    minchan.append(minchan_from_name(file))
    tsample.append(tsample_from_name(file))
    nsubint.append(nsubint_from_name(file))
    nchan.append(nchan_from_name(file))
    bw.append(bw_from_name(file))
ERROR = 0
for i in range(len(imjd)):
    #extract_data_array(args.INPUT_ARCHIVE[i])
    if(i==0):
        file0 = file_liste[i]
        imjd0 = imjd[i]
        smjd0 = smjd[i]
        nsubint0 = nsubint[i]
        tsample0 = tsample[i]
        minchan0 = minchan[i]
        nchan0 = nchan[i]
    else:
        if(imjd[i] != imjd0):
            print('\nThe imjd is different between %s and %s.\n' % ( file0, file_liste[i] ))
            ERROR += 1
        if(round(smjd[i]) != round(smjd0)):
            print('\nThe smjd is different between %s and %s.\n' % ( file0, file_liste[i] ))
            ERROR += 1
        if(tsample[i] != tsample0):
            print('\nThe tsample are different between %s and %s.\n' % ( file0, file_liste[i] ))
            ERROR += 1
        if(nsubint[i] != nsubint0):
            print('\nThe nsubint are different between %s and %s.\n' % ( file0, file_liste[i] ))
            ERROR += 1
        if(minchan[i]%nchan[i] != minchan0%nchan0 ):
            print('\nFiles %s and %s are not aligned in the frequency domain.\n' % ( file0, file_liste[i] ))
            ERROR += 1
        for y in range(len(imjd)):
            if(y != i):
                if(minchan[i] == minchan[y] ):
                    print('\nFiles %s and %s used the same botom channels.\n' % ( file0, file_liste[i] ))
                    ERROR += 1

if(ERROR > 1): exit(0)

for i in range(len(imjd)):
    if(i==0):
        head = fi.getheader( file_liste[i] , 1 , do_not_scale_image_data=True , scale_back=True )        # Extraction of the data header
        #print_headObs(head)
        data = fi.getdata( file_liste[i] , do_not_scale_image_data=True , scale_back=True ) 
        #print(data.columns)
        if (args.norescaloff):
            main_data, main_scales, main_offsets = extract_data_scales_offsets(file_liste[i])
        else:
            main_data = extract_data_array(file_liste[i])
        
        main_freqarray = freqarray_from_name(file_liste[i])
        main_weightarray = weightarray_from_name(file_liste[i])
        TSUBINT_array, OFFS_SUB_array, LST_SUB_array, RA_SUB_array, DEC_SUB_array, GLON_SUB_array, GLAT_SUB_array, FD_ANG_array, POS_ANG_array, PAR_ANG_array, TEL_AZ_array, TEL_ZEN_array = get_other_array_from_name(file_liste[i])

    else:
        ### get the arrays of the next file
        data = fi.getdata( file_liste[i] , do_not_scale_image_data=True , scale_back=True )
        if (args.norescaloff):
            new_data, new_scales, new_offsets = extract_data_scales_offsets(file_liste[i])
        else:
            new_data = extract_data_array(file_liste[i])
        new_freqarray = freqarray_from_name(file_liste[i])
        new_weightarray = weightarray_from_name(file_liste[i])

        ### append with the main matrix
        main_freqarray = np.append(main_freqarray,new_freqarray,axis=1)
        main_weightarray = np.append(main_weightarray,new_weightarray,axis=1)
        if (args.norescaloff):
            main_data = np.append(main_data, new_data,axis=3)
            main_scales = np.append(main_scales, new_scales,axis=2)
            main_offsets = np.append(main_offsets, new_offsets,axis=2)
        else:
            main_data = np.append(main_data, new_data,axis=3)

if not (args.norescaloff):
    main_data, main_scales, main_offsets = data_to_offsets_and_scales(main_data)   ## 8 bits conversion

colList, head, headObs = fits_maker(file_liste[0], TSUBINT_array, OFFS_SUB_array,
           LST_SUB_array, RA_SUB_array, DEC_SUB_array,
           GLON_SUB_array, GLAT_SUB_array, FD_ANG_array,
           POS_ANG_array, PAR_ANG_array, TEL_AZ_array,
           TEL_ZEN_array, main_freqarray, main_weightarray,
           main_offsets, main_scales, main_data)


# DEFINITION OF THE NEW FITS

print('---------save data to '+args.newFileName+' ---------')
colDefs = fi.ColDefs( colList )                    # Creation of the new fields object
tbhdu = fi.BinTableHDU.from_columns( colDefs , header=head )    # Creation of the new data table object

prihdu = fi.PrimaryHDU( header=headObs )            # Creation of the new observation header (exactly the same that the old fits file)
hdulist = fi.HDUList( [ prihdu , tbhdu ] )            # Creation of the new HDU object

hdulist.writeto( args.newFileName ) #output_verify='exception' )                # Writing the new HDU object on the new fits file
hdulist.close()
