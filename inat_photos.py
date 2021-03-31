#!/usr/bin/python3

USE_GPU = False # change to True if you have a GPU

import argparse, base64, io, json, math, os, pickle, re, shelve, sys, time
from datetime import datetime
import requests                   # on Ubuntu install with "sudo apt install python3-requests"
from PIL import Image             # on Ubuntu install with "sudo apt install python3-pil"
import exiftool                   # on Ubuntu install with "sudo apt install libimage-exiftool-perl" and "pip3 install pyexiftool"
import imagehash                  # on Ubuntu install with "sudo apt install python3-scipy" and "pip3 install imagehash"
from SSIM_PIL import compare_ssim # on Ubuntu install with "pip3 install SSIM-PIL"
if USE_GPU:
    import pyopencl # on Ubuntu install with "pip3 install python3-pyopencl"

CLUSTER_THRESHOLD = 180     # 3 minutes
SSIM_THRESHOLD    = 0.95    # 95%, ssim threshold to consider pictures identical
SSIM_INATSIZE     = 'large' # iNaturalist picture size to use with ssim

EXIF_Orientation = 274
THUMBNAIL_MAX = (256, 256)

#
# iNaturalist API
#

API_HOST   = "https://api.inaturalist.org/v1"

CACHE_EXPIRATION  = 8 * 3600  # cache expires after 8 hours

TOO_MANY_API_CALLS_DELAY = 60   # wait this many seconds after error 429

api_time  = 0      # time of last api call sequence
api_count = 0      # number of calls in current api call sequence
API_MAX_CALLS = 60 # max 60 calls per minute
API_INTERVAL  = 60 # 1 minute

# if necessary wait to avoid more than API_MAX_CALLS in API_INTERVAL
def api_call_delay():
    global api_time
    global api_count

    tim = time.time()
    if tim > api_time + API_INTERVAL:
        api_time = tim
        api_count = 1
    else:
        api_count += 1
        if api_count > API_MAX_CALLS:
            print ('Throttling API calls, sleeping for %.1f seconds.' %
                   (API_INTERVAL - (tim - api_time)))
            time.sleep(API_INTERVAL - (tim - api_time))
            api_time = time.time()
            api_count = 1

if sys.platform == 'win32':
    cache_dir  = os.path.join(os.path.expanduser('~'),
                              'AppData', 'Local', 'inat_api')
else:
    cache_dir  = os.path.join(os.path.expanduser('~'), '.cache', 'inat_api')

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

cache = shelve.open(os.path.join(cache_dir, 'api.cache'))

photo_cache_dir = os.path.join(cache_dir, 'photos')
if not os.path.exists(photo_cache_dir):
    os.makedirs(photo_cache_dir)

# lookup iNaturalist user_login; called to check command-line argument -u
def get_user(user_login, bypass_cache=False):
    url = API_HOST + f'/users/{user_login}'
    tim = time.time()
    cache_expiration = 0 if bypass_cache else CACHE_EXPIRATION
    if not url in cache or cache[url][0] < tim - cache_expiration:
        headers = { 'Content-type' : 'application/json',
                    'User-Agent'   : None }
        delay = TOO_MANY_API_CALLS_DELAY
        while True:
            api_call_delay()
            response = requests.get(url, headers=headers)
            if response.status_code == requests.codes.too_many:
                time.sleep(delay)
                delay *= 2
            else:
                break
        if response.status_code == requests.codes.ok:
            cache[url] = (tim, response.json())
        else:
            return None
    return cache[url][1]

# retrieve iNaturalist observations
def get_observations(bypass_cache=False, **kwargs):
    url = API_HOST + '/observations'
    key = pickle.dumps((url, kwargs)).hex()
    tim = time.time()
    cache_expiration = 0 if bypass_cache else CACHE_EXPIRATION
    if not key in cache or cache[key][0] < tim - cache_expiration:
        headers = { 'Content-type' : 'application/json' }
        delay = TOO_MANY_API_CALLS_DELAY
        while True:
            api_call_delay()
            response = requests.get(url, headers=headers, params=kwargs)
            if response.status_code == requests.codes.too_many:
                time.sleep(delay)
                delay *= 2
            else:
                break
        if response.status_code == requests.codes.ok:
            cache[key] = (tim, response.json())
        else:
            print(response.status_code)
            return None
    return cache[key][1]

# retrieve iNaturalist photo
def get_photo(url, size, bypass_cache=False):
    assert size in ['original', 'large', 'medium', 'small', 'square']
    assert url.find('/square.') != -1
    id = int(url[url.find('/photos/')+8:url.find('/square.')])
    url = url.replace('/square.', '/' + size + '.')
    photo_directory = os.path.join(photo_cache_dir, size)
    photo_filename = os.path.join(photo_directory, f'{id}.jpg')
    if not bypass_cache and os.path.exists(photo_filename):
        try:
            with open(photo_filename, 'rb') as file:
                return file.read()
        except:
            print(f"Error: Could not load photo from cache '{photo_filename}'.")

    delay = TOO_MANY_API_CALLS_DELAY
    while True:
        response = requests.get(url)
        if response.status_code == requests.codes.too_many:
            time.sleep(delay)
            delay *= 2
        else:
            break

    if response.status_code == requests.codes.ok:
        try:
            if not os.path.exists(photo_directory):
                os.makedirs(photo_directory)
            with open(photo_filename, 'wb') as file:
                file.write(response.content)
        except:
            print(f"Error: Could not write photo to cache '{photo_filename}'.")
            try:
                os.remove(photo_filename) # delete only partially written file
            except:
                pass
        return response.content
    else:
        print(f"Error {response.status_code} while downloading photo {id}.")
        print(response.text)

#
# Utility Functions
#

# returns difference in seconds between photo and observation dates
def timeDifference(iNatPic, localPic):
    return int((iNatPic.getDateTime() - localPic.getDateTime()).total_seconds())

# returns distance in meters between two lat/lon pairs, Haversine formula
def distanceBetweenPoints(latLon1, latLon2):
    R = 6371e3  # Earth radius in meters
    lat1Rad = latLon1[0]  * math.pi/180
    lat2Rad = latLon2[0]  * math.pi/180
    lon1Rad = latLon1[1]  * math.pi/180
    lon2Rad = latLon2[1]  * math.pi/180
    u = math.sin((lat2Rad - lat1Rad) / 2)
    v = math.sin((lon2Rad - lon1Rad) / 2)
    return 2.0 * R * math.asin(math.sqrt(u * u + math.cos(lat2Rad) *
                                                 math.cos(lat1Rad) * v * v))

# computes structural similarity score; returns float between 0.0 and 1.0
def ssim(localPic, iNatPic):
    img1 = localPic.readImage(thumbNail=False)
    img2 = iNatPic.readImage(thumbNail=False)
    if img1.width != img2.width or img1.height != img2.height:
        if img1.width < img2.width or img1.height < img2.height:
            img2 = img2.resize((img1.width, img1.height))
        else:
            img1 = img1.resize((img2.width, img2.height))
    return compare_ssim(img1, img2, GPU=USE_GPU)

# extract scientific name from caption, returns scientific name or None
def scientific_name(caption):
    scientificName = None
    if caption is not None:
        lparen = caption.find('(')
        if lparen != -1:
            if caption[-1] == ')':
                scientificName = caption[lparen+1:-1]
        else:
            scientificName = caption
        if scientificName is not None:
            names = scientificName.split()
            if len(names) != 3 and len(names) <= 4 and \
               names[0][0].isupper() and \
               (len(names) <= 1 or names[1].islower()) and \
               (len(names) <= 2 or names[2] in ['var.', 'ssp.']) and \
               (len(names) <= 3 or names[3].islower()):
                return scientificName


# Stores info from an iNaturalist observation: the observation id, part of the
# community identification - taxon id, scientific and if available common names
# - and the ids of all photos that belong to this observation as well as geo
# coordinates. Geo coordinates may be obscured or unavailable.
class Observation:

    def __init__(self, id, taxon_id, scientific_name, common_name, photo_urls,
                 dateTime, obscured, latitude, longitude):
        self.id = id
        self.scientific_name = scientific_name
        self.common_name = common_name
        self.taxon_id = taxon_id
        self.photo_urls = photo_urls
        self.dateTime = datetime.fromisoformat(dateTime).replace(tzinfo=None)
        self.obscured = obscured
        self.latLon = (latitude, longitude)

    def getId(self):
        return self.id

    def latLonObscured(self):
        return self.obscured

    def isLatLonValid(self):
        return not self.obscured and self.latLon[0] is not None and \
               self.latLon[1] is not None

    def getLatLon(self):
        return self.latLon

    # combines scientific name with, if present, common name
    def getName(self, html=False):
        scientific_name = self.scientific_name
        if html:
            scientific_name = '<i>' + \
                        ' '.join(['</i>' + s + '<i>' if s[-1] == '.' else s
                                 for s in scientific_name.split()]) + '</i>'

        if self.common_name:
            return "%s (%s)" % (re.sub("(^|-|\s)(\S)",
                                       lambda m: m.group(1) +
                                                 m.group(2).upper(),
                                       self.common_name),
                                scientific_name)
        return scientific_name

    def getTaxonId(self):
        return self.taxon_id

    def numberOfPhotos(self):
        return len(self.photo_urls)

    # Returns list of iNatPhoto instances for this observation.
    def getPhotos(self):
        photos = []
        for photo_url, position in self.photo_urls.items():
            photo = iNatPhoto(photo_url, position, self)
            photos.append(photo)
        return photos

    # date and time of observation
    def getDateTime(self):
        return self.dateTime

    # the observation date
    def getDate(self):
        return self.dateTime.date()


# Base class for iNaturalist and local photos, maintains image hashes.
class Photo:

    def __init__(self):
        self.imageHashes = None

     # compute list of 12 image hashes: 4 hash functions * 3 colors (rgb)
    def computeImageHashes(self, image):
        assert self.imageHashes is None
        rgbImages = image.split()
        imgHashes = [imagehash.average_hash, imagehash.dhash,
                     imagehash.phash, imagehash.whash]
        self.imageHashes = [imageHash(img) for imageHash in imgHashes
                                           for img in rgbImages]

    def getImageHashes(self):
        return self.imageHashes

    def freeImageHashes(self):
        self.imageHashes = None

# Represents iNaturalist photo, stores photo id, links observation, downloads
# photo and computes image hashes.
class iNatPhoto(Photo):

    def __init__(self, url, position, observation):
        super().__init__()
        self.photo_url = url
        self.position = position
        self.observation = observation

    def readImage(self, thumbNail=True):
        return Image.open(io.BytesIO(get_photo(self.photo_url,
                                'small' if thumbNail else SSIM_INATSIZE)))

    def load(self):
        self.computeImageHashes(self.readImage())

    # return observation's date & time
    def getDateTime(self):
        return self.observation.getDateTime()

    # location obscured?
    def isObscured(self):
        return self.observation.obscured

    def isLatLonValid(self):
        return self.observation.isLatLonValid()

    def getLatLon(self):
        return self.observation.getLatLon()

    def getObservationId(self):
        return self.observation.getId()

    def getPhotoId(self):
        assert self.photo_url.find('/square.') != -1
        return int(self.photo_url[self.photo_url.find('/photos/')+8:
                                  self.photo_url.find('/square.')])

    def getPhotoUrl(self):
        return self.photo_url

    def getPosition(self):
        if self.observation.numberOfPhotos() == 1:
            return 'only photo'
        return '1st photo' if self.position == 0 else \
               '2nd photo' if self.position == 1 else \
               '3rd photo' if self.position == 2 else \
               f'{self.position+1}th photo'

    def name(self, html=False):
        return self.observation.getName(html)

    def __str__(self):
        return str(self.getPhotoId())


# Represent local photo, stores file name, date and optional caption, user
# user comment, latitude and longitude and loads and computes image hashes.
# Thumbnails, if present, are loaded instead of large pictures.
class LocalPhoto(Photo):

    def __init__(self, filename, dateTime, orientation, subject, caption,
                 userComment, thumbnailOffset, thumbnailLength, latitude,
                 longitude):
        super().__init__()
        self.filename = filename
        self.dateTime = datetime.fromisoformat(dateTime)
        self.orientation = orientation
        self.subject = subject
        self.caption = caption
        self.userComment = userComment
        self.thumbnailOffset = thumbnailOffset
        self.thumbnailLength = thumbnailLength
        self.latLon = (latitude, longitude)
        self.json = None
        if userComment is not None and len(userComment) and \
           ((userComment[0] == '{' and userComment[-1] == '}') or \
           (userComment[0] == '[' and userComment[-1] == ']')):
            self.json = json.loads(userComment)

    def readImage(self, thumbNail=True):
        if thumbNail and self.thumbnailOffset and self.thumbnailLength:
            with open(self.filename, 'rb') as file:
                file.seek(self.thumbnailOffset)
                image = Image.open(io.BytesIO(file.read(self.thumbnailLength)))
        else:
            image = Image.open(self.filename)
            if thumbNail:
                image.thumbnail(THUMBNAIL_MAX)
        if self.orientation == 1 or self.orientation is None:
            pass
        elif self.orientation == 3:
            image = image.rotate(180, expand=True)
            image.getexif()[EXIF_Orientation] = 1
        elif self.orientation == 6:
            image = image.rotate(270, expand=True)
            image.getexif()[EXIF_Orientation] = 1
        elif self.orientation == 8:
            image = image.rotate(90, expand=True)
            image.getexif()[EXIF_Orientation] = 1
        else:
            print(f'Unsupported EXIF orientation {self.orientation}.')
        return image

    def load(self):
        self.computeImageHashes(self.readImage())

    def isLatLonValid(self):
        return self.latLon[0] is not None and self.latLon[1] is not None

    def getLatLon(self):
        return self.latLon

    def getComment(self):
        return self.userComment

    def getJson(self):
        return self.json

    def getFilename(self):
        return self.filename

    def getDateTime(self):
        return self.dateTime

    def getDate(self):
        return self.dateTime.date()

    def getSubject(self):
        return self.subject

    def getCaption(self):
        return self.caption

    def __str__(self):
        return self.filename

#
# This class compares local photos with iNat photos and on succees
# updates the local photos' metadata.
#
class iNat2LocalImages:

    IMG_EXTS = ['jpg', 'jpeg', 'png', 'tif', 'tiff']

    def __init__(self, user_login, captions, recompute,
                 cluster_threshold, ssim_threshold, bypass_cache):
        self.user_login = user_login
        self.bypass_cache = bypass_cache
        self.captions = captions
        self.recompute = recompute
        self.localPicts = {}
        self.exiftool = exiftool.ExifTool()
        self.exiftool.start()
        self.allFiles = set()
        self.logfile = None
        self.cluster_threshold = cluster_threshold
        self.ssim_threshold = ssim_threshold
        self.no_inat_new = self.no_caption_new = self.no_subject_new = 0
        self.no_inat_updates = self.no_caption_updates = \
            self.no_subject_updates = 0
        self.no_unmatched_localPhotos = self.no_unmatched_iNatPhotos = 0
        self.exiftool_success = b'1 image files updated\r\n' \
                                if sys.platform == 'win32' \
                                else b'1 image files updated\n'

    def isDuplicate(self, fname):
        stat = os.stat(fname)
        key = f"{stat.st_ino},{stat.st_dev}"
        if key in self.allFiles:
            print(f"'{fullPath}' already included.")
            return True
        self.allFiles.add(stat)
        return False

    # Reads exif data of local image on disk.
    def getLocalImage(self, fullPath):

        if self.isDuplicate(fullPath):
            return

        subject = caption = orientation = exifDateTime = userComment = None
        latitude = longitude = None
        metadata = self.exiftool.get_metadata(fullPath)

        if 'EXIF:GPSLatitude' in metadata and \
           'EXIF:GPSLatitudeRef' in metadata and \
           'EXIF:GPSLongitudeRef' in metadata and \
           'EXIF:GPSLongitude' in metadata:
            latitude = float(metadata['EXIF:GPSLatitude'])
            if metadata['EXIF:GPSLatitudeRef'] == 'S':
                latitude = -latitude
            longitude = float(metadata['EXIF:GPSLongitude'])
            if metadata['EXIF:GPSLongitudeRef'] == 'W':
                longitude = -longitude

        if 'EXIF:Orientation' in metadata:
            orientation = metadata['EXIF:Orientation']

        if 'XMP:Subject' in metadata:
            subject = metadata['XMP:Subject']

        if 'EXIF:UserComment' in metadata:
            userComment = metadata['EXIF:UserComment']
        if (userComment is None or userComment == "") and \
             'XMP:INaturalistObservationId' in metadata and \
             'XMP:INaturalistPhotoId' in metadata:
            userComment = '{"iNaturalist": {"observation": %d, "photo": %d}}' %\
                          (metadata['XMP:INaturalistObservationId'],
                           metadata['XMP:INaturalistPhotoId'])

        if 'EXIF:DateTimeOriginal' in metadata:
            exifDateTime = metadata['EXIF:DateTimeOriginal']
        else:
            exifDateTime = metadata['File:FileModifyDate'][:19]
            if self.dots:
                print()
                self.dots = False
            print(f"'{fullPath}': EXIF tag 'Date/Time Original' is missing; "
                  f"using file modification time '{exifDateTime}'.")
        exifDateTime = exifDateTime[:10].replace(':', '-') + exifDateTime[10:]

        if 'IPTC:Caption-Abstract' in metadata:
            caption = metadata['IPTC:Caption-Abstract']

        thumbnailOffset = thumbnailLength = None
        if 'EXIF:ThumbnailLength' in metadata and \
           'EXIF:ThumbnailOffset' in metadata:
            thumbnailOffset = metadata['EXIF:ThumbnailOffset']
            thumbnailLength = metadata['EXIF:ThumbnailLength']

        picture = LocalPhoto(fullPath, exifDateTime, orientation, subject,
                             caption, userComment, thumbnailOffset,
                             thumbnailLength, latitude, longitude)

        exifDate = exifDateTime[:10]
        if exifDate in self.localPicts:
            self.localPicts[exifDate].append(picture)
        else:
            self.localPicts[exifDate] = [picture]

        if len(self.localPicts) % 25 == 0:
            self.dots = True
            print('.', end='')
            sys.stdout.flush()

    # Recursive directory traverse function, finds local images on disk and
    # reads their exif data.
    def getLocalImages(self, directory):

        if self.isDuplicate(directory):
            return

        for file in os.listdir(directory):
            if file[0] == '.':
                continue # skip hidden files and dirs as well as '.' and '..'
            fullPath = os.path.join(directory, file)

            if os.path.islink(fullPath):
                print(f"Ignoring symbolic link '{fullPath}'.")
                continue

            if os.path.isdir(fullPath):
                self.getLocalImages(fullPath)
                continue

            if file.split('.')[-1].lower() in self.IMG_EXTS:
                self.getLocalImage(fullPath)
            elif not file.endswith('.gpx') and not file.endswith('.db') \
                 and not file.endswith('_original'):
                if self.dots:
                    print()
                    self.dots = False
                print(f"File '{fullPath}' not recognized as image.")

    # obtains observations of given date for given user; loads and returns
    # photos and computes image hashes of photos
    def photosFromObservations(self, date, user_login):
        page = 1
        photos = []
        while True:
            results = get_observations(bypass_cache=self.bypass_cache,
                                       page=page, per_page=100,
                                       user_login=user_login,
                                       photos='true', day=date[8:],
                                       month=date[5:7], year=date[:4])
            for result in results["results"]:
                #print (json.dumps(result, indent=4))
                taxon = result["taxon"]
                scientificName = taxon["name"]
                if taxon["rank"] == "subspecies":
                    # insert ssp.
                    scientificName = ' ssp. '.join(scientificName.rsplit(' ',
                                                                         1))
                elif taxon["rank"] == "variety":
                    # insert var.
                    scientificName = ' var. '.join(scientificName.rsplit(' ',
                                                                         1))
                elif taxon["rank"] == "genus":
                    # append sp.
                    scientificName += ' sp.'

                dateTime = result["time_observed_at"]

                commonName = taxon["preferred_common_name"] \
                   if "preferred_common_name" in taxon else None

                photoUrls = {photo['photo']['url'] : photo['position']
                             for photo in result["observation_photos"]} \
                                 if "observation_photos" in result \
                                 else {}

                obscured = latitude = longitude = None
                if 'location' in result:
                    obscured = result["obscured"]
                    location = result["location"].split(',')
                    assert len(location) == 2
                    latitude = float(location[0])
                    longitude = float(location[1])

                observation = Observation(result["id"], taxon["id"],
                                          scientificName, commonName,
                                          photoUrls, dateTime,
                                          obscured, latitude, longitude)

                # load the iNat photos and compute their image hashes
                photos += observation.getPhotos()

            if page * results['per_page'] >= results['total_results']:
                break
            page += 1
        return photos

    def process(self, picture_args, logfile):
        self.logfile = logfile
        startTime = time.time()
        print(f"Loading photos ", end='')
        sys.stdout.flush()
        self.dots = True
        for arg in picture_args:
            if os.path.isdir(arg):
                self.getLocalImages(arg)
            else:
                self.getLocalImage(arg)
        if self.dots:
            print()
            self.dots = False
        no_photos = sum([len(pics) for pics in self.localPicts.values()])
        print(f"Loaded {no_photos} local photo{'' if no_photos == 1 else 's'} "
              f"in {time.time()-startTime:.1f} secs.")

        if no_photos == 0:
            return

        self.writeLogHeader(picture_args)

        for date, localPics in sorted(self.localPicts.items(),
                                      key=lambda x:x[0]):
            iNatPics = self.photosFromObservations(date, self.user_login)
            print(f"{date}: {len(iNatPics)} iNat picture"
                  f"{'' if len(iNatPics) == 1 else 's'} and {len(localPics)} "
                  f"local picture{'' if len(localPics) == 1 else 's'}.")
            if not iNatPics:
                for localPic in localPics:
                    self.reportCandicate(None, localPic)
                continue

            skipPhotos = {} # association with observation/photo already known
            if not self.recompute:
                newLocalPics = []
                for localPic in localPics:
                    if localPic.getJson() is not None and \
                       'iNaturalist' in localPic.getJson() and \
                       'photo' in localPic.getJson()['iNaturalist']:
                        skipPhotos[localPic.getJson()['iNaturalist']
                                   ['photo']] = localPic
                    else:
                        newLocalPics.append(localPic)
                localPics = newLocalPics

            localPics = sorted(localPics, key=lambda x:x.getDateTime())
            sortediNatPics = sorted(iNatPics, key=lambda x:x.getDateTime())

            skipped = False
            iNatPics = []
            for iNatPic in sortediNatPics:
                if iNatPic.getPhotoId() in skipPhotos:
                    self.updateCaption(skipPhotos[iNatPic.getPhotoId()],
                                       iNatPic)
                    skipped = True
                else:
                    iNatPics.append(iNatPic)

            if skipped:
                print(f"{date}: {len(iNatPics)} iNat picture"
                      f"{'' if len(iNatPics) == 1 else 's'} and "
                      f"{len(localPics)} "
                      f"local picture{'' if len(localPics) == 1 else 's'} "
                      "left after excluding known associations.")

            #
            # Partition into clusters of photos according to times the photos
            # were taken.
            #

            i = 0
            j = 0
            while i < len(iNatPics) and j < len(localPics):
                Δt = timeDifference(iNatPics[i], localPics[j])
                if Δt > self.cluster_threshold:
                    self.reportCandicate(None, localPics[j])
                    j += 1
                    continue
                elif Δt < -self.cluster_threshold:
                    self.reportCandicate(iNatPics[i], None)
                    i += 1
                    continue

                ii = i
                jj = j
                change = True
                while change:
                    change = False
                    while i+1 < len(iNatPics) and \
                          timeDifference(localPics[j], iNatPics[i+1]) > \
                             -self.cluster_threshold and \
                          timeDifference(localPics[j], iNatPics[i+1]) < \
                             self.cluster_threshold:
                        i += 1
                        change = True
                    while j+1 < len(localPics) and \
                          timeDifference(localPics[j+1], iNatPics[i]) > \
                             -self.cluster_threshold and \
                          timeDifference(localPics[j+1], iNatPics[i]) < \
                             self.cluster_threshold:
                        j += 1
                        change = True
                self.compareCluster(localPics[jj:j+1], iNatPics[ii:i+1])
                i += 1
                j += 1

            while i < len(iNatPics):
                self.reportCandicate(iNatPics[i], None)
                i += 1

            while j < len(localPics):
                self.reportCandicate(None, localPics[j])
                j += 1

        self.exiftool.terminate()
        self.writeLogTrailer(time.time()-startTime)

    def __del__(self):
        self.exiftool.terminate()

    # Compare lists of local pics with list of iNat observation pics.
    def compareCluster(self, localPics, iNatPics):

        # Load local and iNat pictures and compute their image hashes.
        for pic in localPics:
            pic.load()

        for pic in iNatPics:
            pic.load()

        #
        # Build image hashes for local pictures.
        #
        dicts = [{}] * 12  # 4 hash functions * 3 colors (RGB)
        filename2LocalImage = {}
        pass2_filenames = set()
        for localPic in localPics:
            imgHashes = localPic.getImageHashes()
            assert len(dicts) == len(imgHashes)
            for i in range(len(dicts)):
                key = imgHashes[i]
                dct = dicts[i]
                if key in dct:
                    dct[key].append(localPic)
                else:
                    dct[key] = [localPic]

        # Pass 1: Image Hash with Hamming distances = 0
        # =============================================
        # Iterate over iNaturalist observation photos and hash them to
        # corresponding local photos. Accept only unique matches with
        # Hamming distance 0.

        iNatPic2localPic = {}
        localPic2iNatPic = {}
        for iNatPic in iNatPics:
            imageHashes = iNatPic.getImageHashes()
            results = {} # count of matches for localPics
            for i in range(len(dicts)):
                key = imageHashes[i]
                dct = dicts[i]
                if key in dct:
                    for localPic in dct[key]:
                        if localPic in results:
                            results[localPic] += 1
                        else:
                            results[localPic] = 1
            if results:
                bestLocalPic = []
                bestCount = 0
                for localPic, count in results.items():
                    if count > bestCount:
                        bestCount = count
                        bestLocalPic = [localPic]
                    elif count == bestCount:
                        bestLocalPic.append(localPic)
                for localPic in bestLocalPic:
                    if iNatPic in iNatPic2localPic:
                        iNatPic2localPic[iNatPic].append(localPic)
                    else:
                        iNatPic2localPic[iNatPic] = [localPic]
                    if localPic in localPic2iNatPic:
                        localPic2iNatPic[localPic].append(iNatPic)
                    else:
                        localPic2iNatPic[localPic] = [iNatPic]

        # collect results from pass 1

        iNatPicsSet = set(iNatPics)
        localPicsSet = set(localPics)
        for iNatPic, localPicList in iNatPic2localPic.items():
            if len(localPicList) == 1:
                localPic = localPicList[0]
                if len(localPic2iNatPic[localPic]) == 1:
                    assert localPic2iNatPic[localPic][0] == iNatPic
                    self.reportCandicate(iNatPic, localPic)
                    iNatPicsSet.remove(iNatPic)
                    localPicsSet.remove(localPic)
                    iNatPic.freeImageHashes()
                    localPic.freeImageHashes()

        # Pass 2 uses a greedy algorithm that picks the first result found
        # at the lowest Hamming distance.

        if len(iNatPicsSet) <= len(localPicsSet):
            for iNatPic in iNatPicsSet:
                iNatPicImgHashes = iNatPic.getImageHashes()
                bestDistance = 65 * len(iNatPicImgHashes)
                bestLocalPic = None
                for localPic in localPicsSet:
                    localPicImgHashes = localPic.getImageHashes()
                    dist = sum([iNatPicImgHashes[i] - localPicImgHashes[i]
                                for i in range(len(iNatPicImgHashes))])
                    if dist < bestDistance:
                        bestDistance = dist
                        bestLocalPic = localPic
                self.reportCandicate(iNatPic, bestLocalPic)
                localPicsSet.remove(bestLocalPic)
                iNatPic.freeImageHashes()
                bestLocalPic.freeImageHashes()
            for localPic in localPicsSet:
                self.reportCandicate(None, localPic)
                localPic.freeImageHashes()
        else:
            for localPic in localPicsSet:
                localPicImgHashes = localPic.getImageHashes()
                bestDistance = 65 * len(localPicImgHashes)
                bestiNatPic = None
                for iNatPic in iNatPicsSet:
                    iNatPicImgHashes = iNatPic.getImageHashes()
                    dist = sum([iNatPicImgHashes[i] - localPicImgHashes[i]
                                for i in range(len(iNatPicImgHashes))])
                    if dist < bestDistance:
                        bestDistance = dist
                        bestiNatPic = iNatPic
                self.reportCandicate(bestiNatPic, localPic)
                iNatPicsSet.remove(bestiNatPic)
                bestiNatPic.freeImageHashes()
                localPic.freeImageHashes()
            for iNatPic in iNatPicsSet:
                self.reportCandicate(iNatPic, None)
                iNatPic.freeImageHashes()

    # A candicate pair has been found. This member function reports it,
    # computes the structural similarity and if the iNat and local photos
    # are indeed identical, updates the local photos metadata (exif).
    def reportCandicate(self, iNatPic, localPic):
        if iNatPic is None:
            caption = localPic.getCaption()
            caption = ", caption '" + caption + "'" if caption else ''
            print("Not associated with an iNaturalist photo: "
                  f"'{localPic}'{caption}.")
            self.no_unmatched_localPhotos += 1
            return
        elif localPic is None:
            print(f"Not associated with a local picture: photo id {iNatPic}, "
                  f"{iNatPic.getPosition()} of observation "
                  "https://www.inaturalist.org/observations/"
                  f"{iNatPic.getObservationId()}, '{iNatPic.name()}'.")
            self.no_unmatched_iNatPhotos += 1
            return

        identification = iNatPic.name()
        subject = localPic.getSubject()
        caption = localPic.getCaption()

        timeDiff = abs(timeDifference(iNatPic, localPic))
        diff = f', time from observation {timeDiff} secs'

        distance = distanceBetweenPoints(localPic.getLatLon(),
                                         iNatPic.getLatLon()) \
                    if iNatPic.isLatLonValid() and localPic.isLatLonValid()\
                    else 'obscured' if iNatPic.isObscured() else 'n/a'
        if isinstance(distance, float):
            dist = f', distance from observation {distance:.2f} meters'
        else:
            dist = ', ' + distance

        # Compare pictures by computing their structural similarity score
        # to check the quality of the result. This takes the most runtime.
        ssimScore = ssim(localPic, iNatPic)
        ssimStr = f", similarity {ssimScore*100:.1f}%"

        if caption and scientific_name(caption) == \
                       scientific_name(identification):
            print(f"'{localPic}' caption contains identification for photo id "
                  f"{iNatPic}{diff}{dist}{ssimStr}.")
        elif caption:
            print(f"'{localPic}' found for photo id {iNatPic}; "
                  f"identification '{identification}', "
                  f"current caption '{caption}'{diff}{dist}{ssimStr}.")
        else:
            print(f"'{localPic}' without caption found for photo id {iNatPic}; "
                  f"'{identification}'{diff}{dist}{ssimStr}.")

        self.writeLogLine(localPic, iNatPic, timeDiff, distance, ssimScore)

        if ssimScore >= self.ssim_threshold:
            args = []
            if (subject is None or subject != identification):
                if subject is None:
                    self.no_subject_new += 1
                else:
                    self.no_subject_updates += 1
                print(f"{localPic.getFilename()}: Updating subject from "
                      f"'{subject}' to '{identification}'.")
                args.append(('-Subject='+identification).encode())
            if self.captions and (caption is None or caption != identification):
                if caption is None:
                    self.no_caption_new += 1
                else:
                    self.no_caption_updates += 1
                print(f"{localPic.getFilename()}: Updating caption from "
                      f"'{caption}' to '{identification}'.")
                args.append(('-Caption-Abstract='+identification).encode())
            iNatJson = { 'observation' : iNatPic.getObservationId(),
                         'photo'       : iNatPic.getPhotoId() }
            if not localPic.getJson() or \
               not 'iNaturalist' in localPic.getJson() or \
               localPic.getJson()['iNaturalist'] != iNatJson:
                localJson = localPic.getJson() \
                               if localPic.getJson() is not None else {}
                if 'iNaturalist' in localJson:
                    self.no_inat_updates += 1
                else:
                    self.no_inat_new += 1
                localJson['iNaturalist'] = iNatJson
                args.append(('-UserComment='+json.dumps(localJson)).encode())
            if len(args):
                args.append(b'-m')
                args.append(localPic.getFilename().encode())
                rsl = self.exiftool.execute(*args)
                if rsl != self.exiftool_success:
                    raise Exception(f'exiftool error: {rsl.decode()}')

    # updates the caption of a local photo with the observation's identification
    def updateCaption(self, localPic, iNatPic):
        identification = iNatPic.name()
        subject = localPic.getSubject()
        caption = localPic.getCaption()
        args = []
        if (subject is None or subject != identification):
            if subject is None:
                self.no_subject_new += 1
            else:
                self.no_subject_updates += 1
            print(f"{localPic.getFilename()}: Updating subject from "
                  f"'{subject}' to '{identification}'.")
            args.append(('-Subject='+identification).encode())
        if self.captions and (caption is None or caption != identification):
            if caption is None:
                self.no_caption_new += 1
            else:
                self.no_caption_updates += 1
            print(f"{localPic.getFilename()}: Updating caption from "
                  f"'{caption}' to '{identification}'.")
            args.append(('-Caption-Abstract='+identification).encode())
        if len(args):
            args.append(b'-m')
            args.append(localPic.getFilename().encode())
            rsl = self.exiftool.execute(*args)
            if rsl != self.exiftool_success:
                raise Exception(f'exiftool error: {rsl.decode()}')

    def writeLogHeader(self, picture_args):
        if self.logfile is None:
            return
        title = ' and '.join(picture_args) if len(picture_args) <= 2 else \
                ', '.join(picture_args[:-2] +
                          [', and '.join(picture_args[-2:])])
        print('<html>', file=self.logfile)
        print('<title>' + title + '</title>', file=self.logfile)
        print('<h1>' +  title + '</h1>', file=self.logfile)
        print('<table>', file=self.logfile)
        print('<tr><td></td><td></td><td></td>'
              '<td style="text-align:center"><b>Time from</b></td>'
              '<td style="text-align:center"><b>Distance to</b></td>'
              '<td style="text-align:center"><b>Structural</b></td></tr>',
              file=self.logfile)
        print('<tr><td style="text-align:center"><b>Local File</b></td>'
              '<td style="text-align:center"><b>iNaturalist Photo</b></td>'
              '<td style="text-align:center"><b>Observation</b></td>'
              '<td style="text-align:center"><b>Observation</b></td>'
              '<td style="text-align:center"><b>Observation</b></td>'
              '<td style="text-align:center"><b>Similarity</b></td></tr>',
              file=self.logfile)
        print('<tr><td></td><td></td><td></td>'
              '<td style="text-align:center"><b>[seconds]</b></td>'
              '<td style="text-align:center"><b>[meters]</b></td>'
              '<td style="text-align:center"><b>Score</b></td></tr>',
              file=self.logfile)

    def writeLogLine(self, localPic, iNatPic, timeDiff, distance, ssimScore):
        if self.logfile is None:
            return
        if isinstance(distance, float): # either 'obscured', 'n/a' or float
            distance = f'{distance:.2f}'
        with io.BytesIO() as localIO:
            lImg = localPic.readImage()
            lImg.save(localIO, 'JPEG')
            localImg = localIO.getvalue() # local image as bytestring
        with io.BytesIO() as iNatIO:
            img = iNatPic.readImage()
            img.thumbnail((lImg.width, lImg.height))
            img.save(iNatIO, 'JPEG') # iNat image as bytestring
            iNatImg = iNatIO.getvalue()
        red = ';color:Red' if ssimScore < self.ssim_threshold else ''
        caption = localPic.getCaption()
        if caption is None:
            caption = '<i>no caption</i>'
        print('<tr><td style="text-align:center">'
              f'<a href="file:{os.path.abspath(localPic.getFilename())}" '
              'target="_blank"><img src="data:image/jpeg;base64, '
              f'{base64.b64encode(localImg).decode()}" '
              f'alt="{localPic}"></a><br />{caption}</td>'
              f'<td style="text-align:center"><a href="'
              f'{iNatPic.getPhotoUrl().replace("/square.", "/original.")}"'
              f'target="_blank"><img src="data:image/jpeg;base64, '
              f'{base64.b64encode(iNatImg).decode()}" '
              f'alt="{iNatPic}"></a><br />{iNatPic.getPosition()} of <a '
              'href="https:/www.inaturalist.org/observations/'
              f'{iNatPic.getObservationId()}" target="_blank">observation</a>'
              '</td><td style="word-wrap:break-word;text-align:center">'
              f'<a href="https://www.inaturalist.org/observations/'
              f'{iNatPic.getObservationId()}" target="_blank">'
              f'{iNatPic.name(True)}</a></td>'
              f'<td style="text-align:center">{timeDiff}</td>'
              f'<td style="text-align:center">{distance}</td>'
              f'<td style="text-align:center{red}">{100*ssimScore:.1f}%'
              '</td></tr>', file=self.logfile)

    def writeLogTrailer(self, runTime):
        summary = f"Summary: {self.no_inat_new} iNaturalist annotation" \
                  f"{'' if self.no_inat_new == 1 else 's'} added, " \
                  f"{self.no_inat_updates} modified"
        if self.captions:
            summary += f"; {self.no_caption_new} caption" \
                      f"{'' if self.no_caption_new == 1 else 's'} added, " \
                      f"{self.no_caption_updates} updated"
        summary += f' in {runTime:.0f} seconds.'
        if self.no_unmatched_localPhotos or self.no_unmatched_iNatPhotos:
            summary += f" {self.no_unmatched_localPhotos} local photo" \
                   f"{'' if self.no_unmatched_localPhotos == 1 else 's'} and " \
                   f"{self.no_unmatched_iNatPhotos} iNaturalist photo" \
                   f"{'' if self.no_unmatched_iNatPhotos == 1 else 's'} with"\
                   "out known association."
        print()
        print(summary)
        if self.logfile is not None:
            print(f'</table><p>{summary}</p></html>', file=self.logfile)

#
# Checks for command-line arguments.
#

def userCheck(arg):
    user = get_user(arg)
    if user is None:
        raise argparse.ArgumentTypeError(f"'{arg}' is not an iNaturalist "
                                         "user login.")
    if user['results'][0]['observations_count'] == 0:
        raise argparse.ArgumentTypeError("No observations for iNaturalist user"
                                         f" '{arg}'.")
    return arg

def clusterCheck(arg):
    try:
        iarg = int(arg)
        if iarg >= 60:
            return iarg
    except:
        pass
    raise argparse.ArgumentTypeError(f"'{arg}' is not an acceptable cluster "
                                     "threshold (>= 60).")

def ssimCheck(arg):
    try:
        farg = float(arg[:-1]) / 100 if arg[-1] == '%' else float(arg)
        if farg >= 0 and farg < 1.0:
            return farg
    except:
        pass
    raise argparse.ArgumentTypeError(f"'{arg}' is not an acceptable ssim "
                                     "threshold (0.0 <= ssim < 1.0 or 0% "
                                     "<= ssim < 100%).")

def argCheck(arg):
    if os.path.isdir(arg):
        return arg
    if os.path.isfile(arg) and \
       arg.split('.')[-1].lower() in iNat2LocalImages.IMG_EXTS:
        return arg
    raise argparse.ArgumentTypeError(f"'{arg}' is not a picture file "
                                     "or directory.")

#
# Parse command-line and invoke above functionality.
#

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--user', '-u', type=userCheck, required=True,
                        help='iNaturalist user login')
    parser.add_argument('--cluster_threshold', type=clusterCheck,
                        required=False, default=CLUSTER_THRESHOLD,
                        help='threshold used in clusters based on time of day')
    parser.add_argument('--ssim_threshold', type=ssimCheck,
                        required=False, default=SSIM_THRESHOLD,
                        help='structural similarity score threshold to accept '
                        'candidates')
    parser.add_argument('--bypass_cache', '-b', action="store_true",
                        help='do not use cached api responses even if '
                        'they have not yet expired')
    parser.add_argument('--captions', '-c', action="store_true",
                        help='save identifications as captions')
    parser.add_argument('--recompute', '-r', action="store_true",
                        help='recompute already known associations for photos')
    parser.add_argument('--logfile', type=argparse.FileType('w'),
                        help='write html logfile')
    parser.add_argument('pictures', type=argCheck, nargs='+',
                        metavar='file/directory',
                        help='picture files or directories')


    args = parser.parse_args()

    inat2imgs = iNat2LocalImages(args.user, args.captions, args.recompute,
                                 args.cluster_threshold, args.ssim_threshold,
                                 args.bypass_cache)

    inat2imgs.process(args.pictures, args.logfile)
