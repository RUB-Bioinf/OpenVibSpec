from __future__ import absolute_import

import math
import os  # from pathlib import Path  # might want to look into this at some point
import struct
import sys

###########################################
###########################################
import h5py
import numpy as np
import scipy.io as sio


###########################################

def read_mat2py(x):
    with h5py.File(x, 'r') as f:
        fk = [str(r) for r in f.keys()]
        print("VARIABELS FROM MATLAB ARE GIVEN AS STRINGS")
        print(fk)
        return (fk)


def import_dat(x, var):
    """
    very important to ensure that the data is:
    x,y,z = data.shape
    inside the python environment i make a transpose on the incoming Matlab File
    TODO:
    PRINT ONLY SPECIFIC PARTS OF THE LIST WITH VARIABLE-NAMES FROM MATLAB
    """
    with h5py.File(x, 'r') as f:
        # f.keys()
        # f.items()
        dat = np.asarray(f[var])
        return (dat.T)


def import_mat2py(x):
    arrays = {}
    with h5py.File(x, 'r') as f:
        fk = [str(r) for r in f.keys()]
        for k, v in fk:
            arrays[k] = np.array(v)
        return (arrays[k])


def read_old_matlab(x):
    """
    Matlab-Files before Version 7.3
    """
    # with sio.loadmat(x) as f:
    # try:
    # f = sio.loadmat(x)
    f = sio.loadmat(x)
    # f = sio.loadmat(x)
    fk = [str(r) for r in f.keys()]
    print("VARIABELS FROM MATLAB ARE GIVEN AS STRINGS")
    print("IN THE FIRST AND THRID POSITION")
    print(fk)

    # except:
    # print("Wrong Function For This Matlab-Format:", sys.exc_info()[0])
    print("Wrong Function For This Matlab-Format:", sys.exc_info()[0])
    return (fk)


def import_old_matlab(x, var):
    """
    Matlab-Files before Version 7.3
    """
    # try:
    f = sio.loadmat(x)
    dat = np.asarray(f[var])
    # except:
    print("Wrong Function For This Matlab-Format:", sys.exc_info()[0])
    return (dat)


# Python 2.7
# def import_agielnt(x):
#	with open("a01.dmt", "rb") as binary_file:
#    	binary_file.seek(2236)
#    	couple_bytes = binary_file.read()
#    	print(couple_bytes)
#    return 

# Python 3.5
def import_agilent(x):
    with open("a01.dmt", "rb") as binary_file:
        binary_file.seek(2236)
        couple_bytes = binary_file.read()
        sys.stdout.buffer.write(couple_bytes)
    return


def bands(x):
    import sys
    with open("A01.hdr", "rb") as binary:
        binary.seek(76, 0)
        bytes_ = binary.read(6)
        return (sys.stdout.buffer.write(bytes_))


def load_import(name):
    with h5py.File(str(name), 'r') as f:

        fk = [str(r) for r in f.keys()]

        if fk[0] != 'HyperSpecCube':
            print(
                'Raw import not from OpenVibSpec. Please use read_mat2py() instead to get variable names, and load manually')

        else:
            c = np.asarray(f['HyperSpecCube'])
            wn = np.asarray(f['WVN'])

    return c, wn


class RawIO():
    """

    #-----------------------------------------------------------------------#
    # IMPORT RAW DATA FROM WITHIN A DATA DIRECTORY
    #
    #
    #
    #
    #-----------------------------------------------------------------------#


    These Methods load all individual tiles from the spectroscopes and save them in the specified directory

    under the specified name as a new *.h5 file. The file created in this way is packed at the current time using gzip,

    which affects the runtime. At a later time, this parameter will be freely selectable in OpenVibSpec.

    Automatically, the spectral mean image is saved in the same folder as a form of verification for the spectroscopist.


    #-----------------------------------------------------------------------#
    #-----------------------------------------------------------------------#

    """

    def import_raw_qcl(self, file_dir, new_file_name):
        """

        PARAMS:
            file_dir(path): directory with the specific files (ONLY THE FILES!)
            new_file_name(string): name for the new *.h5 file

        """

        # self.file_dir = file_dir
        # self.new_file_name = new_file_name

        # def read_raw_qcl(self, file_dir):
        def read_raw_qcl():
            from os import listdir
            import numpy as np
            from os.path import isfile, join

            onlyfiles = [f for f in listdir(file_dir) if isfile(join(file_dir, f))]

            its = []

            for ii, jj in enumerate(onlyfiles):
                its.append(ii)

            itsn = np.asarray(its)

            return (onlyfiles, itsn, file_dir)

        def cut_info(onlyfiles, itsn, file_dir):

            ll_sub = []
            for it in onlyfiles:
                ll_sub.append(it.split('[')[1].split(']')[0].split('-')[1].split('_'))

            table = np.asarray(ll_sub, dtype='i4')

            all_n = np.column_stack((table, itsn))

            x_sort = all_n[all_n[:, 0].argsort()]

            y_sort = x_sort[x_sort[:, 1].argsort()]

            sorted_ids = y_sort[y_sort[:, 0].argsort()]

            return sorted_ids

        def info():
            import glob
            import scipy.io as sio

            files = glob.glob(file_dir + '*')

            i = sio.loadmat(files[0])

            print(i.keys())

            wn = i['wn']

            dy = i['dY']

            dx = i['dX']

            head = i['__header__']

            return (wn, wn.shape[0], int(dy), int(dx), head)

        def load_tiles_by_id(dats, id_, fpasize, wn):
            import scipy.io as sio

            i = sio.loadmat(dats[id_])

            d = i['r']

            return d.reshape(fpasize, fpasize, wn)

        def load_tc():
            import numpy as np
            import os

            ll, lon, pf = read_raw_qcl()

            sorted_ids = cut_info(ll, lon, file_dir)

            wvn, wnd, dx, dy, head = info()

            os.chdir(file_dir)

            l = []

            for i in sorted_ids[:, 2]:
                id_ = load_tiles_by_id(ll, i, dx, wnd)
                l.append(id_)

            matrices = np.asarray(l)

            del l

            return matrices, wvn, wnd, dx, dy, head

        m1, wvn, wnd, dx, dy, head = load_tc()

        ms1 = m1.swapaxes(1, 2)[::-1]

        def create_mosaic_arrofarrs(images):
            # very strongly inspired by:
            # https://pypi.org/project/ImageMosaic/1.0/
            #

            # Trivial cases
            if len(images) == 0:
                return None

            if len(images) == 1:
                return images[0]

            imshapes = [image.shape for image in images]
            imdims = [len(imshape) for imshape in imshapes]
            dtypes = [image.dtype for image in images]
            n_cols = int(np.ceil(np.sqrt(len(images))))
            n_rows = int(np.ceil(len(images) / n_cols))

            if n_rows <= 0 or n_cols <= 0:
                raise ValueError("'nrows' and 'ncols' must both be > 0")

            if n_cols * n_rows < len(images):
                raise ValueError(
                    "Grid is too small: n_rows * n_cols < len(images)")

            fpa_size_x = images[0].shape[0]
            fpa_size_y = images[0].shape[1]

            # Allocation
            out_block = np.ones((fpa_size_x * n_rows, fpa_size_y * n_cols, images[0].shape[2]), dtype=dtypes[0])

            for fld in range(len(images)):
                index_1 = int(np.floor(fld / n_rows))
                index_0 = fld - (n_rows * index_1)

                start0 = index_0 * (fpa_size_x)
                end0 = start0 + fpa_size_x
                start1 = index_1 * (fpa_size_y)
                end1 = start1 + fpa_size_y
                out_block[start0:end0, start1:end1] = images[fld]

            return out_block

        out = create_mosaic_arrofarrs(ms1)
        del ms1

        def save_mosaic(out, wvn):

            import h5py

            # Standard should be gzip based on the distribution
            # internally choose between gzip,bzip or others

            # ToDo:
            # from mpi4py import MPI
            # import h5py
            #
            # rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
            #
            # f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
            #
            # dset = f.create_dataset('test', (4,), dtype='i')
            # dset[rank] = rank
            #
            # f.close()

            hf = h5py.File(str(new_file_name) + '.h5', 'w')
            hf.create_dataset('HyperSpecCube', data=out, compression="gzip", compression_opts=6)
            hf.create_dataset('WVN', data=wvn, compression="gzip", compression_opts=6)
            # hf.create_dataset('FPA_Size', data=dx, compression="gzip", compression_opts=4)
            # hf.create_dataset('header', data=str(head), compression="gzip", compression_opts=4)
            hf.close()
            return

        def save_overview_image(out):
            # out after update
            # import scipy.misc as sm
            import imageio
            # img_uint8 = out.astype(np.uint8)
            # imageio.imwrite(str(new_file_name)+'.png',img_uint8.mean(axis=2))
            imageio.imwrite(str(new_file_name) + '.png', out.mean(axis=2))
            # sm.imsave(str(new_file_name)+'.png',out.mean(axis=2))

            return

        save_mosaic(out, wvn)
        save_overview_image(out)

    # --------------------TODO------------------------------------------------------
    # ------------------------------------------------------------------------------
    # THIS HAPPENS WHEN THERE IS MORE IN THE DIR THAN IT SHOULD BE
    # 	WE NEED BETTER ERROR DESRCIPTION HERE
    # ------------------------------------------------------------------------------
    #	---------------------------------------------------------------------------
    #	IndexError                                Traceback (most recent call last)
    #	<timed eval> in <module>
    #
    #	~/cdwork/openvibspec/OpenVibSpec/scr/openvibspec/io_ftir.py in import_raw_qcl(self, file_dir, new_file_name)
    #	    264
    #	    265
    #	--> 266                 m1, wvn, wnd, dx, dy, head = load_tc()
    #	    267
    #	    268                 ms1 = m1.swapaxes(1,2)[::-1]
    #
    #	~/cdwork/openvibspec/OpenVibSpec/scr/openvibspec/io_ftir.py in load_tc()
    #	    240                         ll,lon,pf = read_raw_qcl()
    #	    241
    #	--> 242                         sorted_ids = cut_info(ll,lon,file_dir)
    #	    243
    #	    244                         wvn, wnd, dx, dy, head = info()
    #
    #	~/cdwork/openvibspec/OpenVibSpec/scr/openvibspec/io_ftir.py in cut_info(onlyfiles, itsn, file_dir)
    #	    173                         ll_sub = []
    #	    174                         for it in onlyfiles:
    #	--> 175                                 ll_sub.append(it.split('[')[1].split(']')[0].split('-')[1].split('_'))
    #	    176
    #	    177
    #
    #	IndexError: list index out of range

    # -----------------------------------------------------------------------#
    # IMPORT RAW AGILENT DATA FROM WITHIN A DATA DIRECTORY
    # -----------------------------------------------------------------------#

    def import_agilent_mosaic(self, file_dir, new_file_name):
        """

        PARAMS:
            file_dir(path): directory with the specific files (ONLY THE FILES!)
            new_file_name(string): name for the new *.h5 file

        """

        def get_dtype():
            return np.dtype('<f')

        def read_raw_agilent(file_dir):
            from os.path import isfile, join

            all_files = [f for f in ([os.path.join(file_dir, fi) for fi in os.listdir(file_dir)]) if
                         isfile(join(file_dir, f))]  # f for f in listdir(file_dir) if isfile(join(file_dir, f))]
            dms_files = list(filter(lambda x: x.endswith((".dms")), all_files))
            return dms_files

        def isreadable(filename=None):
            # Check filename is provided.
            if filename is not None:
                # Check file extension.
                fname = os.path.basename(filename)
                (fstub, fext) = os.path.splitext(fname)
                if fext.lower() not in (".dmd", ".drd", ".dms", ".dmt"):
                    return False
                # Additional tests here
                # Passed all available tests so we can read this file
                return True
            else:
                return False

        def _readwinint32(binary_file):
            return int.from_bytes(binary_file.read(4), byteorder='little')

        def _readwindouble(binary_file):
            return struct.unpack('<d', binary_file.read(8))[0]

        def _readtile(filename, numberofpoints, fpasize):
            tile = np.memmap(filename, dtype=get_dtype(), mode='r')
            # skip the (unknown) header
            tile = tile[255:]
            tile = np.reshape(tile, (numberofpoints, fpasize, fpasize))
            # tile = xr.DataArray(tile)
            return tile

        def _getwavenumbersanddate(fpath, fstub):
            dmtfilename = os.path.join(fpath, (fstub + ".dmt"))
            with open(dmtfilename, "rb") as binary_file:
                binary_file.seek(2228, os.SEEK_SET)
                startwavenumber = _readwinint32(binary_file)
                # print(self.startwavenumber)
                binary_file.seek(2236, os.SEEK_SET)
                numberofpoints = _readwinint32(binary_file)
                # print(self.numberofpoints)
                binary_file.seek(2216, os.SEEK_SET)
                wavenumberstep = _readwindouble(binary_file)
                # print(self.wavenumberstep)

                stopwavenumber = startwavenumber + (wavenumberstep * numberofpoints)

                wavenumbers = np.arange(1, numberofpoints + startwavenumber)
                wavenumbers = wavenumbers * wavenumberstep
                wavenumbers = np.delete(wavenumbers, range(0, startwavenumber - 1))

                # # read in the whole file (it's small) and regex it for the acquisition date/time
                # binary_file.seek(0, os.SEEK_SET)
                # contents = binary_file.read()
                # regex = re.compile(b"Time Stamp.{44}\w+, (\w+) (\d\d), (\d\d\d\d) (\d\d):(\d\d):(\d\d)")
                # matches = re.match(regex, contents)
                # matches2 = re.match(b'(T)', contents)
                return numberofpoints, wavenumbers

        def _getfpasize(fpath, fstub, numberofpoints):
            tilefilename = os.path.join(fpath, (fstub + "_0000_0000.dmd"))
            tilesize = os.path.getsize(tilefilename)
            data = tilesize - (255 * 4)  # remove header
            data = data / numberofpoints
            data = data / 4  # sizeof float
            fpasize = int(math.sqrt(data))  # fpa size likely to be 64 or 128 pixels square
            return fpasize

        def _xtiles(fpath, fstub):
            finished = False
            counter = 0
            while not finished:
                tilefilename = os.path.join(fpath, (fstub + "_{:04d}_0000.dmd".format(counter)))
                if not os.path.exists(tilefilename):
                    return counter
                else:
                    counter += 1
            return counter

        def _ytiles(fpath, fstub):
            finished = False
            counter = 0
            while not finished:
                tilefilename = os.path.join(fpath, (fstub + "_0000_{:04d}.dmd".format(counter)))
                if not os.path.exists(tilefilename):
                    return counter
                else:
                    counter += 1
            return counter

        def read(filename=None):
            """ToDo: If filename is None, open a dialog box"""
            if isreadable(filename):
                fpath = os.path.dirname(filename)
                fname = os.path.basename(filename)
                (fstub, fext) = os.path.splitext(fname)

                # Read the .dmt file to get the wavenumbers and date of acquisition
                # Generate the .dmt filename
                numberofpoints, wavenumbers = _getwavenumbersanddate(fpath, fstub)
                fpasize = _getfpasize(fpath, fstub, numberofpoints)

                xtiles = _xtiles(fpath, fstub)
                ytiles = _ytiles(fpath, fstub)

                numxpixels = fpasize * xtiles
                numypixels = fpasize * ytiles

                alldata = np.empty((numberofpoints, numypixels, numxpixels), dtype=get_dtype())

                ystart = 0
                for y in reversed(range(ytiles)):
                    ystop = ystart + fpasize

                    xstart = 0
                    for x in (range(xtiles)):
                        xstop = xstart + fpasize

                        tilefilename = os.path.join(fpath, (fstub + "_{:04d}_{:04d}.dmd".format(x, y)))
                        tile = _readtile(tilefilename, numberofpoints, fpasize)
                        # tile = da.from_delayed(tile, (self.numberofpoints, self.fpasize, self.fpasize), self.datatype)
                        # tile = xr.DataArray(tile)

                        alldata[:, ystart:ystop, xstart:xstop] = tile

                        xstart = xstop
                    ystart = ystop

                alldata = np.fliplr(alldata)
                alldata = np.transpose(alldata, (1, 2, 0))  # from (z,x,y) to (x,y,z)

                info = {'filename': filename,
                        'xpixels': numxpixels,
                        'ypixels': numypixels,
                        'xtiles': xtiles,
                        'ytiles': ytiles,
                        'numpts': numberofpoints,
                        'fpasize': fpasize
                        }
                # kwargs consists of: xlabel, ylabel, xdata, ydata, info.
                return dict(ydata=alldata, ylabel='absorbance',
                            xdata=wavenumbers, xlabel='wavenumbers (cm-1)',
                            info=info)

        def save_agent_mosaic(out, wvn):

            import h5py

            hf = h5py.File(str(new_file_name) + '.h5', 'w')
            hf.create_dataset('HyperSpecCube', data=out, compression="gzip", compression_opts=6)
            hf.create_dataset('WVN', data=wvn, compression="gzip", compression_opts=6)
            # hf.create_dataset('FPA_Size', data=dx, compression="gzip", compression_opts=4)
            # hf.create_dataset('header', data=str(head), compression="gzip", compression_opts=4)
            hf.close()
            return

        def save_overview_image(out):

            # out after update
            # import scipy.misc as sm
            # sm.imsave(str(new_file_name)+'.png',out.mean(axis=2))

            import imageio
            # img_uint8 = out.astype(np.uint8)
            # imageio.imwrite(str(new_file_name)+'.png',img_uint8.mean(axis=2))
            imageio.imwrite(str(new_file_name) + '.png', out.mean(axis=2))
            # sm.imsave(str(new_file_name)+'.png',out.mean(axis=2))
            return

        dms_files = read_raw_agilent(file_dir)
        dict_d = read(dms_files[0])
        save_agent_mosaic(dict_d.get('ydata'), dict_d.get('xdata'))
        save_overview_image(dict_d.get('ydata'))
