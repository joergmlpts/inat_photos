# inat_photos

I have uploaded some of my photos to [iNaturalist](https://www.inaturalist.org) where the community identifies plants, animals, mushrooms and other life forms. This script `inat_photos.py` reads all my photos and finds the ones that have been uploaded to iNaturalist. It then annotatates my local photos with their location on iNaturalist (the observation and photo ids). It also looks up the identifications on iNaturalist and stores them as captions of my local photos.

`inat_photos.py` is a command-line tool that has been tested on Linux and Windows 10. It can be called as this on Linux

```
./inat_photos.py --user joergmlpts --captions --logfile mylogfile.html ./example_pictures
```

and like this on Windows

```
python.exe .\inat_photos.py --user joergmlpts --captions --logfile mylogfile.html .\example_pictures
```

where `joergmlpts` is a user login on iNaturalist, `--captions` requests that captions will be updated with iNaturalist identifications, a logfile called `mylogfile.html` will be written, and `example_pictures` is a directory of pictures.

`inat_photos.py` will read all pictures in `example_pictures` and its subdirectories, will look up iNaturalist observations for these pictures and annotate the pictures with their location on iNaturalist (the observation and photo ids) and will store the identifications from iNaturalist as captions. The example run looks like this:

```
Loading photos 
Loaded 2 local photos in 0.1 secs.
2021-02-01: 2 iNat pictures and 2 local pictures.
'./example_pictures/dcsf0124.jpg' without caption found for photo id 111753559; 'Milkmaids (Cardamine californica)', time from observation 18 secs, distance from observation 0.00 meters, similarity 98.3%.
./example_pictures/dcsf0124.jpg: Updating caption from 'None' to 'Milkmaids (Cardamine californica)'.
'./example_pictures/dcsf0123.jpg' without caption found for photo id 111753544; 'Milkmaids (Cardamine californica)', time from observation 12 secs, distance from observation 0.00 meters, similarity 98.7%.
./example_pictures/dcsf0123.jpg: Updating caption from 'None' to 'Milkmaids (Cardamine californica)'.

Summary: 2 iNaturalist annotations added, 0 modified; 2 captions added, 0 updated in 5 seconds.
```

The html logfile allows to manually verify that the correct photos have been found on iNaturalist. The logfile shows thumbnails of all the pictures. A click on these thumbnails opens the full-size pictures. Additional information is the time difference between the iNaturalist observation and the time the photo was taken. iNaturalist stores times without seconds, only hour and minutes. Hence, this time difference is unlikely to be zero. Also, the distance in meters between the iNaturalist observation and the gps coordinates of the photo is shown. This distance is unavailable when the photo has no gps coordinates or if the iNaturalist observation has an obscured location. Obscured locations occur to protect rare plants and animals. Finally, a [structural similarity score](https://en.wikipedia.org/wiki/Structural_similarity) is shown, a number between 0 and 1 which indicates how similar the local picture and iNaturalist picture are. Only if this number reaches a certain threshold (95% by default), `inat_photos.py` associates a local photo with an iNaturalist photo. Should the structural similarity score be below this threshold it will be shown in red in the logfile and the photo will not be annotated with its location on iNaturalist and the caption will not be updated.


## Dependencies

This code has been written in Python3. It uses f-strings and hence needs Python 3.6 or later. It requires `exiftool` to read and write the metadata of pictures. All necessary packages can be installed on Ubuntu Linux (and likely all other Debian-style distributions) with these two commands:

```
sudo apt install python3 python3-pip python3-requests python3-pil python3-scipy python3-pyopencl libimage-exiftool-perl
sudo pip3 install pyexiftool imagehash SSIM-PIL
```

where `python3-pyopencl` is only needed to run `inat_photos.py` with GPU support.

## Windows or other Operating Systems

First of all, `exiftool` needs to be installed. It can be downloaded from [its website](https://exiftool.org/). The `exiftool` variant for command-line use is needed; for Windows, the installation involves downloading a zip and renaming `exiftool(-k).exe` to `exiftool.exe` and placing it in `C:\Windows\`. In addition to `exiftool`, these Python packages need to be installed:

```
pip install requests pillow pyexiftool imagehash pyopencl SSIM-PIL
```

For me, `pyexiftool` did not work on Windows after I installed it with `pip` as described above. `inat_photos.py` died with error `OSError: [WinError 10038] An operation was attempted on something that is not a socket`. This turned out to be a [known issue](https://github.com/smarnach/pyexiftool/issues/26) on Windows and I could work around it by uninstalling the version installed by `pip`:

```
pip uninstall pyexiftool
```

and installing first [`git` for Windows](https://gitforwindows.org/) and finally the most recent `pyexiftool` from `github`:

```
pip install git+https://github.com/smarnach/pyexiftool.git
```
