# inat_photos

I have uploaded some of my photos to [iNaturalist](https://www.inaturalist.org)
where a community identifies plants, animals, mushrooms and other life forms.
This script `inat_photos.py` reads my photos and finds the ones that have been
uploaded to iNaturalist. It then annotatates my local photos with their location
on iNaturalist (the observation and photo ids). It also pulls the
identifications from iNaturalist and stores them as captions of my local photos.

`inat_photos.py` is a command-line tool that has been tested on Linux and
Windows 10. It can be called as this on Linux

```
./inat_photos.py --user joergmlpts --captions --logfile mylogfile.html ./example_pictures
```

and like this on Windows

```
python.exe .\inat_photos.py --user joergmlpts --captions --logfile mylogfile.html .\example_pictures
```

where `joergmlpts` is a user login on iNaturalist, `--captions` requests
captions to be updated with iNaturalist identifications, a logfile called
`mylogfile.html` will be written, and `example_pictures` is a directory of
pictures.

`inat_photos.py` reads all pictures in `example_pictures` and its
subdirectories, looks up iNaturalist observations for these pictures and
annotates the pictures with their location on iNaturalist (the observation and
photo ids) and stores the identifications from iNaturalist as captions. The
example run looks like this:

```
Loading photos 
Loaded 5 local photos in 0.2 secs.
2021-02-01: 7 iNat pictures and 5 local pictures.
'example_pictures/dcsf0119_crop.jpg' without caption found for photo id 111827921; 'Coyote (Canis latrans)', time from observation 58 secs, distance from observation 0.00 meters, similarity 100.0%.
example_pictures/dcsf0119_crop.jpg: Updating caption from 'None' to 'Coyote (Canis latrans)'.
'example_pictures/dcsf0118.jpg' without caption found for photo id 111827898; 'Coyote (Canis latrans)', time from observation 7 secs, distance from observation 72.38 meters, similarity 98.4%.
example_pictures/dcsf0118.jpg: Updating caption from 'None' to 'Coyote (Canis latrans)'.
'example_pictures/dcsf0119.jpg' without caption found for photo id 111827912; 'Coyote (Canis latrans)', time from observation 58 secs, distance from observation 0.00 meters, similarity 98.7%.
example_pictures/dcsf0119.jpg: Updating caption from 'None' to 'Coyote (Canis latrans)'.
Not associated with a local picture: photo id 111827935, 2nd photo of observation https://www.inaturalist.org/observations/68980359, 'Coyote (Canis latrans)'.
Not associated with a local picture: photo id 111827888, 5th photo of observation https://www.inaturalist.org/observations/68980359, 'Coyote (Canis latrans)'.
'example_pictures/dcsf0124.jpg' without caption found for photo id 111753559; 'Milkmaids (Cardamine californica)', time from observation 18 secs, distance from observation 0.00 meters, similarity 98.3%.
example_pictures/dcsf0124.jpg: Updating caption from 'None' to 'Milkmaids (Cardamine californica)'.
'example_pictures/dcsf0123.jpg' without caption found for photo id 111753544; 'Milkmaids (Cardamine californica)', time from observation 12 secs, distance from observation 0.00 meters, similarity 98.7%.
example_pictures/dcsf0123.jpg: Updating caption from 'None' to 'Milkmaids (Cardamine californica)'.

Summary: 5 iNaturalist annotations added, 0 modified; 5 captions added, 0 updated in 11 seconds. 0 local photos and 2 iNaturalist photos without known association.
```

This is excerpt from the html logfile generated by the above run:
![logfile](/images/logfile_successful_associations.png)

The html logfile allows to visually verify that the correct photos have been
found on iNaturalist. The logfile shows thumbnails of all the pictures. A click
on these thumbnails opens the full-size pictures. Additional information is the
time difference between the iNaturalist observation and the time the photo was
taken. iNaturalist stores times without seconds, only hour and minutes. Hence,
this time difference is unlikely to be zero. Also, the distance in meters
between the iNaturalist observation and the gps coordinates of the photo is
shown. This distance is unavailable when the photo has no gps coordinates or
when the iNaturalist observation has an obscured location. Obscured locations
occur to protect rare organisms. Finally, a
[structural similarity score](https://en.wikipedia.org/wiki/Structural_similarity)
is shown, a number between 0 and 1 which indicates how similar the local
picture and the iNaturalist picture are. Only if this number reaches a certain
threshold (95% by default), `inat_photos.py` will associate a local photo with
an iNaturalist photo.

![logfile](/images/logfile_bad_association.png)

A structural similarity score below this threshold is shown
in red in the logfile as seen above and the photo is not annotated with a
location on iNaturalist and the caption is not updated.

Subsequent runs will be much faster since the associations between local photos
and iNaturalist observations and their photos are already known:

```
Loading photos 
Loaded 5 local photos in 0.2 secs.
2021-02-01: 7 iNat pictures and 5 local pictures.
2021-02-01: 2 iNat pictures and 0 local pictures left after excluding known associations.
Not associated with a local picture: photo id 111827935, 2nd photo of observation https://www.inaturalist.org/observations/68980359, 'Coyote (Canis latrans)'.
Not associated with a local picture: photo id 111827888, 5th photo of observation https://www.inaturalist.org/observations/68980359, 'Coyote (Canis latrans)'.

Summary: 0 iNaturalist annotations added, 0 modified; 0 captions added, 0 updated in 0 seconds. 0 local photos and 2 iNaturalist photos without known association.
```

## Dependencies

This code has been written in Python. It uses f-strings and hence needs Python
3.6 or later. It requires `exiftool` to read and write the metadata of pictures.
All necessary packages can be installed on Ubuntu Linux and other Debian-style
distributions with these two commands:

```
sudo apt install python3 python3-pip python3-requests python3-pil python3-scipy python3-pyopencl libimage-exiftool-perl
sudo pip3 install pyexiftool imagehash SSIM-PIL
```

where `python3-pyopencl` is only needed to run `inat_photos.py` with GPU
support.

## Windows and other Operating Systems

First of all, `exiftool` needs to be installed. It can be downloaded from
[its website](https://exiftool.org/). The `exiftool` variant for command-line
use is needed; on Windows, the installation involves downloading a zip and
renaming `exiftool(-k).exe` to `exiftool.exe` and placing it in `C:\Windows\`.
In addition to `exiftool`, these Python packages need to be installed:

```
pip install requests pillow pyexiftool imagehash pyopencl SSIM-PIL
```

`pyexiftool` did not work for me on Windows after I installed it with `pip` as
described above. `inat_photos.py` died with error
`OSError: [WinError 10038] An operation was attempted on something that is not a socket`.
This turned out to be a [known issue](https://github.com/smarnach/pyexiftool/issues/26)
on Windows and I could work around it by uninstalling the version installed
with `pip`:

```
pip uninstall pyexiftool
```

and then first installed [git for Windows](https://gitforwindows.org/) and
finally instelled the most recent `pyexiftool` from `github`:

```
pip install git+https://github.com/smarnach/pyexiftool.git
```

## Command-line Arguments

This script is a command-line tool. It is called with options, file names and
directory names as arguments. These options are supported:

```
Usage: inat_photos.py [-h] --user USER [--cluster_threshold CLUSTER_THRESHOLD] [--ssim_threshold SSIM_THRESHOLD] [--bypass_cache] [--captions] [--recompute] [--logfile LOGFILE]
                      file/directory [file/directory ...]

positional arguments:
  file/directory        picture files or directories

optional arguments:
  -h, --help            show this help message and exit
  --user USER, -u USER  iNaturalist user login
  --cluster_threshold CLUSTER_THRESHOLD
                        threshold used in clusters based on time of day
  --ssim_threshold SSIM_THRESHOLD
                        structural similarity score threshold to accept candidates
  --bypass_cache, -b    do not use cached api responses even if they have not yet expired
  --captions, -c        save identifications as captions
  --recompute, -r       recompute already known associations for photos
  --logfile LOGFILE     write html logfile
```

### Option -u USER, --user USER

Option `-u` or `--user` sets the iNaturalist user login. This option is needed
for `inat_photo.py` to download observations from iNaturalist.

### Option --logfile LOGFILE

Option `--logfile` is optional. The argument is a html logfile that allows to
review the results.

### Option --cluster_threshold CLUSTER_THRESHOLD

Option `--cluster_threshold` allows to modify the cluster threshold. The default
is 180 seconds. Photos are compared in clusters which limit the search space to
photos that have been taken at about the same time. A photo is only added to a
cluster if there is at least one member of the cluster within
`CLUSTER_THRESHOLD` seconds. Otherwise a new cluster will be started.

### Option --ssim_threshold SSIM_THRESHOLD

By default, `inat_photo.py` associates a local photos and an iNaturalist photos
when the structural siomilariry score of these photos is at least 0.95. Option
`--ssim_threshold` allows to modify this threshold. The threshold can be
specified as a number between 0 and 1.0 or as a percentage. The default is 0.95
or 95%.

### Option -b, --bypass_cache

By default, `inat_photo.py` caches iNaturalist observations for 14 days before
they will be loaded again. Options `-b` and `--bypass_cache` instruct
`inat_photo.py` to bypass the cache and issue api calls to load iNaturalist
observations right away.

### Option -c, --captions

Options `-c` and `--captions` instructs `inat_photo.py` to pull identifications
from iNaturalist observations and write them as captions to local photos.
Captions of local photos are stored as `IPTC` tag `Caption-Abstract`, a tag
supported by Google Photos and other services. By default, the captions are not
written.

### Option -r, --recompute

Options `-r` and `--recompute` instructs `inat_photo.py` to ignore known
associations between local and iNaturalist photos and recompute them.

## Constants in the Code

A few constants in the code can also be customized. The constants are found near the top of the file:

```
USE_GPU = False             # change to True if you have a GPU

CLUSTER_THRESHOLD = 180     # 3 minutes
SSIM_THRESHOLD    = 0.95    # 95%, ssim threshold to consider pictures identical

SSIM_INATSIZE     = 'large' # iNaturalist picture size to use with ssim

CACHE_EXPIRATION  = 14 * 24 * 3600  # cache expires after 2 weeks
```

The constants `CLUSTER_THRESHOLD` and `SSIM_THRESHOLD` are the defaults for
options `--cluster_threshold` and `--ssim_threshold`.

Constant `USE_GPU` can be set to `True` to compute structural similarity scores
with GPU support. That setting should provide a runtime boost but requires
hardware support, e.g. a system with a GPU.

Constant `SSIM_INATSIZE` is the iNaturalist photo size to be downloaded to
compute structural similarity scores. The default is `large` which results in
slow but high-quality picture comparisons. Possible values are `small`,
`medium`, `large` and `original`. This setting is a tradeoff between runtime and
quality of results. If this size is lowered, the structural similarity score
needs to be lowered as well, e.g. `SSIM_INATSIZE = 'small'` and
`SSIM_THRESHOLD = 0.8` is a setting that provides a very significant speedup
while still finding roughly the same number of associations.

Constant `CACHE_EXPIRATION` is the age in seconds when a cache entry of an
iNaturalist observation expires. The default is 14 days. Lowering this number
increases network traffic. Command-line option `--bypass_cache` is an
alternative to customizing this constant.

## Metadata

`inat_photo.py` saves iNaturalist associations and captions to the local photos'
metadata. iNaturalist associations are stored in `json` format as `EXIF` tag
`UserComment`. Captions are stored as `IPTC` tag `Caption-Abstract`.

This metadata can be displayed with command `exiftool -UserComment -Caption-Abstract *.jpg`:

```
======== dcsf0118.jpg
User Comment                    : {"iNaturalist": {"observation": 68980359, "photo": 111827898}}
Caption-Abstract                : Coyote (Canis latrans)
======== dcsf0119_crop.jpg
User Comment                    : {"iNaturalist": {"observation": 68980359, "photo": 111827921}}
Caption-Abstract                : Coyote (Canis latrans)
======== dcsf0119.jpg
User Comment                    : {"iNaturalist": {"observation": 68980359, "photo": 111827912}}
Caption-Abstract                : Coyote (Canis latrans)
======== dcsf0123.jpg
User Comment                    : {"iNaturalist": {"observation": 68939822, "photo": 111753544}}
Caption-Abstract                : Milkmaids (Cardamine californica)
======== dcsf0124.jpg
User Comment                    : {"iNaturalist": {"observation": 68939822, "photo": 111753559}}
Caption-Abstract                : Milkmaids (Cardamine californica)
    5 image files read
```

The `json` for iNaturalist can be one of several photo associations.
Associations for other websites are supported as well. For instance, another
association may be present for [Calflora](https://www.calflora.org). In this
case, `inat_photo.py` does not touch the association for Calflora; only an
association for iNaturalist is added or modified.
