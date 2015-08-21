# -*- encoding: utf-8 -*-
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')


DOWNLOAD_DIRECTORY = os.path.join(BASE_DIR, '.downloads')
BINARIES_DIRECTORY = os.path.join(BASE_DIR, 'autosklearn/binaries')
METADATA_DIRECTORY = os.path.join(BASE_DIR, 'autosklearn/metalearning/files')

SCRIPT_FOLDER = os.path.join(BASE_DIR, 'scripts')