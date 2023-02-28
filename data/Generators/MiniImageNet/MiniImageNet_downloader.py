from learn2learn.data.utils import download_file_from_google_drive, download_file
import os
import requests
import tqdm
import pickle


CHUNK_SIZE = 1 * 1024 * 1024

def download_pkl(google_drive_id, data_root, mode):
    filename = 'mini-imagenet-cache-' + mode
    file_path = os.path.join(data_root, filename)

    if not os.path.exists(file_path + '.pkl'):
        print('Downloading:', file_path + '.pkl')
        download_file_from_google_drive(google_drive_id, file_path + '.pkl')
    else:
        print("Data was already downloaded")

def download_file(source, destination, size=None):
    if size is None:
        size = 0
    req = requests.get(source, stream=True)
    with open(destination, 'wb') as archive:
        for chunk in tqdm.tqdm(
            req.iter_content(chunk_size=CHUNK_SIZE),
            total=size // CHUNK_SIZE,
            leave=False,
        ):
            if chunk:
                archive.write(chunk)


download_links_google = {"test": '1wpmY-hmiJUUlRBkO9ZDCXAcIpHEFdOhD',
                               "train": '1I3itTXpXxGV68olxM5roceUMG8itH9Xj',
                               "validation": '1KY5e491bkLFqJDp0-UWou3463Mo8AOco'}


download_links_dropbox = {"test":'https://www.dropbox.com/s/ye9jeb5tyz0x01b/mini-imagenet-cache-test.pkl?dl=1',
                          "train":'https://www.dropbox.com/s/9g8c6w345s2ek03/mini-imagenet-cache-train.pkl?dl=1',
                          "validation":'https://www.dropbox.com/s/ip1b7se3gij3r1b/mini-imagenet-cache-validation.pkl?dl=1'}

for mode in ['train', 'test', 'validation']:
    pickle_file = os.path.join('../../', 'mini-imagenet-cache-' + mode + '.pkl')
    try:
        print('Downloading Mini-ImageNet --', mode)
        download_pkl(download_links_google[mode], './', mode)
        with open(pickle_file, 'rb') as f:
            pickle.load(f)
    except pickle.UnpicklingError:
        print('Download failed. Re-trying Mini-ImageNet --', mode)
        download_file(download_links_dropbox[mode], pickle_file)
        with open(pickle_file, 'rb') as f:
            pickle.load(f)
