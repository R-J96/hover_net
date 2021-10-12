from collections import OrderedDict
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage import color
import re
import subprocess

import openslide

from tiatoolbox.wsicore import wsireader


class FileHandler(object):
    def __init__(self):
        """The handler is responsible for storing the processed data, parsing
        the metadata from original file, and reading it from storage.
        """
        self.metadata = {
            ("available_mag", None),
            ("base_mag", None),
            ("vendor", None),
            ("mpp  ", None),
            ("base_shape", None),
        }
        pass

    def __load_metadata(self):
        raise NotImplementedError

    def get_full_img(self, read_mag=None, read_mpp=None):
        """Only use `read_mag` or `read_mpp`, not both, prioritize `read_mpp`.

        `read_mpp` is in X, Y format
        """
        raise NotImplementedError

    def read_region(self, coords, size):
        """Must call `prepare_reading` before hand.

        Args:
            coords (tuple): (dims_x, dims_y),
                          top left coordinates of image region at selected
                          `read_mag` or `read_mpp` from `prepare_reading`
            size (tuple): (dims_x, dims_y)
                          width and height of image region at selected
                          `read_mag` or `read_mpp` from `prepare_reading`

        """
        raise NotImplementedError

    def get_dimensions(self, read_mag=None, read_mpp=None):
        """Will be in X, Y."""
        if read_mpp is not None:
            read_scale = (self.metadata["base_mpp"] / read_mpp)[0]
            read_mag = read_scale * self.metadata["base_mag"]
        scale = read_mag / self.metadata["base_mag"]
        # may off some pixels wrt existing mag
        return (self.metadata["base_shape"] * scale).astype(np.int32)

    def prepare_reading(self, read_mag=None, read_mpp=None, cache_path=None):
        """Only use `read_mag` or `read_mpp`, not both, prioritize `read_mpp`.

        `read_mpp` is in X, Y format.
        """
        read_lv, scale_factor = self._get_read_info(
            read_mag=read_mag, read_mpp=read_mpp
        )

        if scale_factor is None:
            self.image_ptr = None
            self.read_lv = read_lv
        else:
            np.save(cache_path, self.get_full_img(read_mag=read_mag))
            self.image_ptr = np.load(cache_path, mmap_mode="r")
        return

    def _get_read_info(self, read_mag=None, read_mpp=None):
        if read_mpp is not None:
            assert read_mpp[0] == read_mpp[1], "Not supported uneven `read_mpp`"
            read_scale = (self.metadata["base_mpp"] / read_mpp)[0]
            read_mag = read_scale * self.metadata["base_mag"]

        hires_mag = read_mag
        scale_factor = None
        if read_mag not in self.metadata["available_mag"]:
            if read_mag > self.metadata["base_mag"]:
                scale_factor = read_mag / self.metadata["base_mag"]
                hires_mag = self.metadata["base_mag"]
            else:
                mag_list = np.array(self.metadata["available_mag"])
                mag_list = np.sort(mag_list)[::-1]
                hires_mag = mag_list - read_mag
                # only use higher mag as base for loading
                hires_mag = hires_mag[hires_mag > 0]
                # use the immediate higher to save compuration
                hires_mag = mag_list[np.argmin(hires_mag)]
                scale_factor = read_mag / hires_mag

        hires_lv = self.metadata["available_mag"].index(hires_mag)
        return hires_lv, scale_factor


class OpenSlideHandler(FileHandler):
    """Class for handling OpenSlide supported whole-slide images."""

    def __init__(self, file_path):
        """file_path (string): path to single whole-slide image."""
        super().__init__()
        self.file_ptr = openslide.OpenSlide(file_path)  # load OpenSlide object
        self.metadata = self.__load_metadata()

        # only used for cases where the read magnification is different from
        self.image_ptr = None  # the existing modes of the read file
        self.read_level = None

    def __load_metadata(self):
        metadata = {}

        wsi_properties = self.file_ptr.properties

        # TODO bug here, this line produces a key error with tiff files
        # hacked in a version of toolbox WSIreader for this case without
        # the warnings and exceptions
        if openslide.PROPERTY_NAME_OBJECTIVE_POWER in wsi_properties:
            level_0_magnification = wsi_properties[
                openslide.PROPERTY_NAME_OBJECTIVE_POWER
            ]
            mpp = [
                wsi_properties[openslide.PROPERTY_NAME_MPP_X],
                wsi_properties[openslide.PROPERTY_NAME_MPP_Y],
            ]
            mpp = np.array(mpp)
        else:
            level_0_magnification = None
            mpp = None
            tiff_res_units = wsi_properties.get("tiff.ResolutionUnit")
            microns_per_unit = {
                "centimeter": 1e4,  # 10k
                "inch": 25400,
            }
            x_res = float(wsi_properties["tiff.XResolution"])
            y_res = float(wsi_properties["tiff.YResolution"])
            mpp_x = 1 / x_res * microns_per_unit[tiff_res_units]
            mpp_y = 1 / y_res * microns_per_unit[tiff_res_units]
            mpp = [mpp_x, mpp_y]
        if level_0_magnification is None:
            if mpp is not None:
                mpp = float(np.mean(mpp))
                common_powers = (1, 1.25, 2, 2.5, 4, 5, 10, 20, 40, 60, 90, 100)
                op = 10 / float(mpp)
                distances = [np.abs(op - power) for power in common_powers]
                closest_match = common_powers[np.argmin(distances)]
                level_0_magnification = closest_match

        level_0_magnification = float(level_0_magnification)

        downsample_level = self.file_ptr.level_downsamples
        magnification_level = [level_0_magnification / lv for lv in downsample_level]

        metadata = [
            ("available_mag", magnification_level),  # highest to lowest mag
            ("base_mag", magnification_level[0]),
            ("vendor", wsi_properties[openslide.PROPERTY_NAME_VENDOR]),
            ("mpp  ", mpp),
            ("base_shape", np.array(self.file_ptr.dimensions)),
        ]
        return OrderedDict(metadata)

    def read_region(self, coords, size):
        """Must call `prepare_reading` before hand.

        Args:
            coords (tuple): (dims_x, dims_y),
                          top left coordinates of image region at selected
                          `read_mag` or `read_mpp` from `prepare_reading`
            size (tuple): (dims_x, dims_y)
                          width and height of image region at selected
                          `read_mag` or `read_mpp` from `prepare_reading`

        """
        if self.image_ptr is None:
            # convert coord from read lv to lv zero
            lv_0_shape = np.array(self.file_ptr.level_dimensions[0])
            lv_r_shape = np.array(self.file_ptr.level_dimensions[self.read_lv])
            up_sample = (lv_0_shape / lv_r_shape)[0]
            new_coord = [0, 0]
            new_coord[0] = int(coords[0] * up_sample)
            new_coord[1] = int(coords[1] * up_sample)
            region = self.file_ptr.read_region(new_coord, self.read_lv, size)
        else:
            region = self.image_ptr[
                coords[1] : coords[1] + size[1], coords[0] : coords[0] + size[0]
            ]
        return np.array(region)[..., :3]

    def get_full_img(self, read_mag=None, read_mpp=None):
        """Only use `read_mag` or `read_mpp`, not both, prioritize `read_mpp`.

        `read_mpp` is in X, Y format.
        """

        read_lv, scale_factor = self._get_read_info(
            read_mag=read_mag, read_mpp=read_mpp
        )

        read_size = self.file_ptr.level_dimensions[read_lv]

        wsi_img = self.file_ptr.read_region((0, 0), read_lv, read_size)
        wsi_img = np.array(wsi_img)[..., :3]  # remove alpha channel
        if scale_factor is not None:
            # now rescale then return
            if scale_factor > 1.0:
                interp = cv2.INTER_CUBIC
            else:
                interp = cv2.INTER_LINEAR
            wsi_img = cv2.resize(
                wsi_img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=interp
            )
        return wsi_img


class JP2000Handler(FileHandler):
    """Class for handling jp2 wsis"""

    def __init__(self, file_path):
        super().__init__()
        self.wsi_reader = wsireader.get_wsireader(file_path)
        self.wsi_metadata = self.wsi_reader.info.as_dict()
        self.metadata = self.__load_metadata(self.wsi_metadata)

        self.image_ptr = None
        self.read_level = None

    def __load_metadata(self, metadata):
        wsi_metadata = metadata

        level_0_mag = wsi_metadata["objective_power"]
        downsample_level = wsi_metadata["level_downsamples"]
        mag_level = [level_0_mag / lv for lv in downsample_level]

        metadata = [
            ("available_mag", mag_level),  # highest to lowest mag
            ("base_mag", mag_level[0]),
            ("vendor", wsi_metadata["vendor"]),
            ("mpp  ", wsi_metadata["mpp"][0]),
            ("base_shape", np.array(wsi_metadata["slide_dimensions"])),
        ]
        return OrderedDict(metadata)

    def read_region(self, coords, size):
        if self.image_ptr is None:
            # convert coord from read lv to lv zero
            lv_0_shape = np.array(self.wsi_metadata.level_dimensions[0])
            lv_r_shape = np.array(self.wsi_metadata.level_dimensions[self.read_lv])
            up_sample = (lv_0_shape / lv_r_shape)[0]
            new_coord = [0, 0]
            new_coord[0] = int(coords[0] * up_sample)
            new_coord[1] = int(coords[1] * up_sample)
            region = self.wsi_reader.read_region(new_coord, self.read_lv, size)
        else:
            region = self.image_ptr[
                coords[1] : coords[1] + size[1], coords[0] : coords[0] + size[0]
            ]
        return np.array(region)[..., :3]

    def get_full_img(self, read_mag=None, read_mpp=None):
        read_lv, scale_factor = self._get_read_info(
            read_mag=read_mag, read_mpp=read_mpp
        )

        read_size = self.wsi_metadata['level_dimensions'][read_lv]

        wsi_img = self.wsi_reader.read_region((0, 0), read_lv, read_size)
        wsi_img = np.array(wsi_img)[..., :3]  # remove alpha channel
        if scale_factor is not None:
            # now rescale then return
            if scale_factor > 1.0:
                interp = cv2.INTER_CUBIC
            else:
                interp = cv2.INTER_LINEAR
            wsi_img = cv2.resize(
                wsi_img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=interp
            )
        return wsi_img


def get_file_handler(path, backend):
    if backend in [
        ".svs",
        ".tif",
        ".vms",
        ".vmu",
        ".ndpi",
        ".scn",
        ".mrxs",
        ".tiff",
        ".svslide",
        ".bif",
    ]:
        return OpenSlideHandler(path)
    elif backend == ".jp2":
        return JP2000Handler(path)
    else:
        assert False, "Unknown WSI format `%s`" % backend


if __name__ == "__main__":
    # svs_test = get_file_handler("/data/TCGA/Bladder/WSIs/TCGA-4Z-AA7O-01Z-00-DX1.A45C32E8-AA40-4182-8A6F-986DFE56A748.svs", ".svs")
    test = get_file_handler("/data/UHCW/WSIs/Test/H10-3166_B2H_and_E_1.tiff", ".tiff")
    print(test)
    jp2_test = get_file_handler("/data/UHCW/WSIs_jp2/H09-4350_A1H_and_E_1.jp2", ".jp2")
    print(jp2_test)
