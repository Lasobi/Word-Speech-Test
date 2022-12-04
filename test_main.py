import unittest
import main
import pathlib


class TestMain(unittest.TestCase):
    def test_if_file_from_recording_is_made(self):
        path = pathlib.Path("out.wav")
        main.record_audio()
        self.assertEqual((str(path), path.is_file()), (str(path), True))

    def test_if_files_are_removed(self):
        path = pathlib.Path("test_file.txt")
        with open(str(path), 'w') as fp:
            fp.write(str(path))
            pass
        main.remove_file(str(path))
        self.assertEqual((str(path), path.is_file()), (str(path), False))
