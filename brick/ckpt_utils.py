import os

import pathlib
import shutil
import uuid
from composer.loggers import RemoteUploaderDownloader as ComposerRemoteUploaderDownloader


class RemoteUploaderDownloader(ComposerRemoteUploaderDownloader):

    def upload_file(
        self,
        remote_file_name: str,
        file_path: pathlib.Path,
        *,
        overwrite: bool,
    ):
        copied_path = os.path.join(self._upload_staging_folder,
                                   str(uuid.uuid4()))
        os.makedirs(self._upload_staging_folder, exist_ok=True)
        shutil.copy2(file_path, copied_path)
        formatted_remote_file_name = self._remote_file_name(remote_file_name)
        with self._object_lock:
            if formatted_remote_file_name in self._logged_objects and not overwrite:
                raise FileExistsError(
                    f'Object {formatted_remote_file_name} was already enqueued to be uploaded, but overwrite=False.'
                )
            self._logged_objects[formatted_remote_file_name] = (copied_path,
                                                                overwrite)