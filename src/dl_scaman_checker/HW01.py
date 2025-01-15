import sys

import torchvision
import itertools
import requests
import re
import os
import warnings
import contextlib
import copy
import urllib.request

from dl_scaman_checker.common import check_install

def _extract_gdrive_form(response, chunk_size: int = 32 * 1024):
    content = response.iter_content(chunk_size)
    first_chunk = None
    # filter out keep-alive new chunks
    while not first_chunk:
        first_chunk = next(content)
    content = itertools.chain([first_chunk], content)

    try:
        match = re.search("<form [^>]*>(?P<api_response>.+?)</form>", first_chunk.decode())
        api_response = match["api_response"] if match is not None else None
        if api_response is not None:
            match = re.findall("<input [^>]* name=\"(?P<name>.+?)\" [^>]*value=\"(?P<value>.+?)\"[^>]*>", api_response)
            dic = { k: v for k,v in match }
        else:
            dic = {}
    except UnicodeDecodeError:
        api_response = None
    return api_response, dic, content

def inject_and_resubmit(params, response):
    url = "https://drive.usercontent.google.com/download"
    _, dic, _ = _extract_gdrive_form(response)

    with requests.Session() as session:
        response = session.get(url, params=dict(params, **dic), stream=True)

    return torchvision.datasets.utils._extract_gdrive_api_response(response)


def patched__download_file_from_google_drive(file_id, root, filename, md5):
    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.fspath(os.path.join(root, filename))

    os.makedirs(root, exist_ok=True)

    if torchvision.datasets.utils.check_integrity(fpath, md5):
        print(f"Using downloaded {'and verified ' if md5 else ''}file: {fpath}")
        return

    _extract_gdrive_api_response = torchvision.datasets.utils._extract_gdrive_api_response

    url = "https://drive.usercontent.google.com/download"
    params = dict(id=file_id, export="download")
    with requests.Session() as session:
        response = session.get(url, params=params, stream=True)
        backup_response = copy.deepcopy(response)

        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break
        else:
            api_response, content = _extract_gdrive_api_response(response)
            token = None

        if token is not None:
            response = session.get(url, params=dict(params, confirm=token), stream=True)
            api_response, content = _extract_gdrive_api_response(response)
        elif api_response == "Virus scan warning":
            api_response, content = inject_and_resubmit(params, backup_response)

        if api_response == "Quota exceeded":
            raise RuntimeError(
                f"The daily quota of the file {filename} is exceeded and it "
                f"can't be downloaded. This is a limitation of Google Drive "
                f"and can only be overcome by trying again later."
            )

        torchvision.datasets.utils._save_response_content(content, fpath)

    # In case we deal with an unhandled GDrive API response, the file should be smaller than 10kB and contain only text
    if os.stat(fpath).st_size < 10 * 1024:
        with contextlib.suppress(UnicodeDecodeError), open(fpath) as fh:
            text = fh.read()
            # Regular expression to detect HTML. Copied from https://stackoverflow.com/a/70585604
            if re.search(r"</?\s*[a-z-][^>]*\s*>|(&(?:[\w\d]+|#\d+|#x[a-f\d]+);)", text):
                warnings.warn(
                    f"We detected some HTML elements in the downloaded file. "
                    f"This most likely means that the download triggered an unhandled API response by GDrive. "
                    f"Please report this to torchvision at https://github.com/pytorch/vision/issues including "
                    f"the response:\n\n{text}"
                )

    if md5 and not torchvision.datasets.utils.check_md5(fpath, md5):
        raise RuntimeError(
            f"The MD5 checksum of the download file {fpath} does not match the one on record."
            f"Please delete the file and try again. "
            f"If the issue persists, please report this to torchvision at https://github.com/pytorch/vision/issues."
        )


class PCAM(torchvision.datasets.pcam.PCAM):
    def __init__(self, *args, use_robin_mirror=False, **kwargs):
        self.use_robin_mirror = use_robin_mirror
        super().__init__(*args, **kwargs)

    def _mirror_download(self):
        folder = self._base_folder

        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)

        for file_name, file_id, md5 in self._FILES[self._split].values():
            url = f"https://www.robindar.com/pcam-hw01/{file_name}.gz"
            fpath = str(self._base_folder / f"{file_name}.gz")
            print("Using mirror:", url, flush=True)
            urllib.request.urlretrieve(url, fpath)

            if md5 and not torchvision.datasets.utils.check_md5(fpath, md5):
                raise RuntimeError(
                        f"The MD5 checksum of the download file {fpath} does not match the one on record."
                        f"Please delete the file and try again. This was downloaded from the robindar.com mirror."
                        f"If the issue persists, please report this to torchvision at https://github.com/pytorch/vision/issues."
                        )
            torchvision.datasets.utils._decompress(fpath)

    def _download(self):
        if self._check_exists():
            return

        if self.use_robin_mirror:
            return self._mirror_download()

        for file_name, file_id, md5 in self._FILES[self._split].values():
            archive_name = file_name + ".gz"
            patched__download_file_from_google_drive(file_id, str(self._base_folder), filename=archive_name, md5=md5)
            torchvision.datasets.utils._decompress(str(self._base_folder / archive_name))
