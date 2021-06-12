# inat_photos

I have uploaded some of my photos to [iNaturalist](https://www.inaturalist.org)
where a community of naturalists identifies plants, animals, mushrooms and
other organisms.
This script `inat_photos.py` reads offline photos and finds the ones that have
been uploaded to iNaturalist. It then annotatates these local photos with their
location on iNaturalist (the observation and photo ids). It also pulls the
identifications from iNaturalist and stores them in the metadata of the local
photos.

`inat_photos.py` is a command-line tool. It can be called as this on Linux

```
./inat_photos.py --user joergmlpts --logfile mylogfile.html ./example_pictures
```

where `joergmlpts` is a user login for iNaturalist, a logfile called
`mylogfile.html` will be written, and `example_pictures` is a directory of
pictures.

`inat_photos.py` reads all pictures in `example_pictures` and its
subdirectories, downloads iNaturalist observations for these pictures,
annotates the pictures with their location on iNaturalist (the observation and
photo ids) and stores the identifications from iNaturalist as captions/subjects.
The example run looks like this:

```
Loading photos 
Loaded 5 local photos in 0.1 secs.
2021-02-01: 7 iNat pictures and 5 local pictures.
'./example_pictures/dcsf0119_crop.jpg' without caption found for photo id 111827921; 'Coyote (Canis latrans)', time from observation 58 secs, distance from observation 0.00 meters, similarity 100.0%.
./example_pictures/dcsf0119_crop.jpg: Updating subject from 'None' to 'Coyote (Canis latrans)'.
'./example_pictures/dcsf0118.jpg' without caption found for photo id 111827898; 'Coyote (Canis latrans)', time from observation 7 secs, distance from observation 72.38 meters, similarity 98.4%.
./example_pictures/dcsf0118.jpg: Updating subject from 'None' to 'Coyote (Canis latrans)'.
'./example_pictures/dcsf0119.jpg' without caption found for photo id 111827912; 'Coyote (Canis latrans)', time from observation 58 secs, distance from observation 0.00 meters, similarity 98.7%.
./example_pictures/dcsf0119.jpg: Updating subject from 'None' to 'Coyote (Canis latrans)'.
Not associated with a local picture: photo id 111827888, 5th photo of observation https://www.inaturalist.org/observations/68980359, 'Coyote (Canis latrans)'.
Not associated with a local picture: photo id 111827935, 2nd photo of observation https://www.inaturalist.org/observations/68980359, 'Coyote (Canis latrans)'.
'./example_pictures/dcsf0124.jpg' without caption found for photo id 111753559; 'Milkmaids (Cardamine californica)', time from observation 18 secs, distance from observation 0.00 meters, similarity 98.3%.
./example_pictures/dcsf0124.jpg: Updating subject from 'None' to 'Milkmaids (Cardamine californica)'.
'./example_pictures/dcsf0123.jpg' without caption found for photo id 111753544; 'Milkmaids (Cardamine californica)', time from observation 12 secs, distance from observation 0.00 meters, similarity 98.7%.
./example_pictures/dcsf0123.jpg: Updating subject from 'None' to 'Milkmaids (Cardamine californica)'.

Summary: 5 iNaturalist annotations added, 0 modified; 5 identifications added, 0 modified in 10 seconds. 0 local photos and 2 iNaturalist photos without known association.
```

This is an excerpt from the html logfile generated by the above run:
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

Summary: 0 iNaturalist annotations added, 0 modified in 0 seconds. 0 local photos and 2 iNaturalist photos without known association.
```

## Dependencies

This code has been written in Python. It uses f-strings and hence needs Python
3.6 or later. It requires `py3exiv2` to read and write the metadata of pictures.
All necessary packages can be installed on Ubuntu Linux and other Debian-style
distributions with these two commands:

```
sudo apt install python3 python3-pip python3-requests python3-pil python3-scipy libexiv2-dev libboost-python-dev python3-pyopencl
sudo pip3 install py3exiv2 imagehash SSIM-PIL pyinaturalist
```

where `python3-pyopencl` is only needed to run `inat_photos.py` with GPU
support.

## Windows and other Operating Systems

Most of the dependencies can be installed with this command:

```
pip install requests pillow imagehash pyopencl SSIM-PIL pyinaturalist
```

where `pyopencl` is only needed to run `inat_photos.py` with GPU support.

On Windows package `py3exiv2` cannot simply be installed with `pip`. [This fork](https://github.com/auphofBSF/py3exiv2) of `py3exiv2` has support and installation instructions for Windows 10.

## Command-line Arguments

This script is a command-line tool. It is called with options, file names and
directory names as arguments. These options are supported:

```
Usage: inat_photos.py [-h] --user USER [--cluster_threshold CLUSTER_THRESHOLD] [--ssim_threshold SSIM_THRESHOLD] [--bypass_cache] [--recompute] [--logfile LOGFILE]
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

### Option -r, --recompute

Options `-r` and `--recompute` instructs `inat_photo.py` to ignore known
associations between local and iNaturalist photos and recompute them.

## Constants in the Code

A few constants in the code can also be customized. The constants are found
near the top of the file:

```
USE_GPU = False               # change to True if you have a GPU

CLUSTER_THRESHOLD = 180       # 3 minutes
SSIM_THRESHOLD    = 0.95      # 95%, ssim threshold to consider pictures identical

SSIM_INATSIZE     = 'large'   # iNaturalist picture size to use with ssim
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

## Metadata

`inat_photo.py` saves iNaturalist associations and captions to the local photos'
metadata. iNaturalist associations are stored in tags named `observation`,
`observationPhoto`, and `photo` in an `XMP` namespace `iNaturalist`. These tags
can be set to numeric ids and string uuids. This script sets them to the numeric
ids. Captions is stored as `XMP` tag `Subject`.

This metadata can be displayed with command `exiftool -XMP-iNaturalist:observation -XMP-iNaturalist:observationPhoto -XMP-iNaturalist:photo -Subject *.jpg`:

```
======== dcsf0118.jpg
Observation                     : 68980359
Observation Photo               : 104339085
Photo                           : 111827898
Subject                         : Coyote (Canis latrans)
======== dcsf0119_crop.jpg
Observation                     : 68980359
Observation Photo               : 104339082
Photo                           : 111827921
Subject                         : Coyote (Canis latrans)
======== dcsf0119.jpg
Observation                     : 68980359
Observation Photo               : 104339084
Photo                           : 111827912
Subject                         : Coyote (Canis latrans)
======== dcsf0123.jpg
Observation                     : 68939822
Observation Photo               : 104269804
Photo                           : 111753544
Subject                         : Milkmaids (Cardamine californica)
======== dcsf0124.jpg
Observation                     : 68939822
Observation Photo               : 104269803
Photo                           : 111753559
Subject                         : Milkmaids (Cardamine californica)
    5 image files read
```
