import numpy as np
from skimage import draw
import SimpleITK as sitk

from dcmrtstruct2nii.exceptions import ContourOutOfBoundsException

import logging


class DcmPatientCoords2Mask():
    def _poly2mask(self, coords_x, coords_y, shape):
        # Compute a mask from polygon.
        mask = draw.polygon2mask(tuple(reversed(shape)), np.column_stack((coords_y, coords_x)))
        # NOTE: arg 2: The polygon coordinates of shape (N, 2) where N is the number of points.
        return mask

    def convert(self, rtstruct_contours, dicom_image, mask_background, mask_foreground):
        shape = dicom_image.GetSize() # (512, 512, 246)

        mask = sitk.Image(shape, sitk.sitkUInt8)
        mask.CopyInformation(dicom_image)

        np_mask = sitk.GetArrayFromImage(mask)
        np_mask.fill(mask_background)

        for contour in rtstruct_contours:
            if contour['type'].upper() not in ['CLOSED_PLANAR', 'INTERPOLATED_PLANAR', 'OPEN_NONPLANAR']:
                if 'name' in contour:
                    logging.info(f'Skipping contour {contour["name"]}, unsupported type: {contour["type"]}')
                else:
                    logging.info(f'Skipping unnamed contour, unsupported type: {contour["type"]}')
                continue

            coordinates = contour['points']
            # {'x': ['30.11', '26.91', '23.37', '12.27'], 'y': ['112.67', '115.01', '117.57', '129.2'], 'z': ['274.71', '250.6', '222.87', '118.19']}

            pts = np.zeros([len(coordinates['x']), 3])

            for index in range(0, len(coordinates['x'])):
                # lets convert world coordinates to voxel coordinates
                world_coords = dicom_image.TransformPhysicalPointToContinuousIndex((coordinates['x'][index], coordinates['y'][index], coordinates['z'][index]))
                pts[index, 0] = world_coords[0]
                pts[index, 1] = world_coords[1]
                pts[index, 2] = world_coords[2]

            #   pts =   
            #    array([[332.31841584, 226.26851485, 167.67      ],
            #    [324.20752475, 232.19960396, 143.56      ],
            #    [315.23485149, 238.68831683, 115.83      ],
            #    [287.10019802, 268.16633663,  11.15      ]])

            z = int(round(pts[0, 2])) # NOTE: decide z level to use for the entire structure by picking first z-value

            try:
                if contour['type'].upper() != 'OPEN_NONPLANAR':
                    # NOTE: x, y, shape discarding z axis...
                    filled_poly = self._poly2mask(pts[:, 0], pts[:, 1], [shape[0], shape[1]]) 
                    # array([[False, False, False, ..., False, False, False],
                    # [False, False, False, ..., False, False, False],
                    # [False, False, False, ..., False, False, False],
                    # ...,
                    # [False, False, False, ..., False, False, False],
                    # [False, False, False, ..., False, False, False],
                    # [False, False, False, ..., False, False, False]]) # shape=(512,512)
                    np_mask[z, filled_poly] = mask_foreground  # sitk is xyz, numpy is zyx
                else:
                    for row in range(0, len(pts)):
                        this_z = int(round(pts[row, 2]))
                        this_y = int(round(pts[row, 1]))
                        this_x = int(round(pts[row, 0]))
                        # import ipdb; ipdb.set_trace()
                        np_mask[this_z, this_y, this_x] = mask_foreground # NOTE: (246, 512, 512) zyx shape

            except IndexError:
                # if this is triggered the contour is out of bounds
                raise ContourOutOfBoundsException()
            except RuntimeError as e:
                # this error is sometimes thrown by SimpleITK if the index goes out of bounds
                if 'index out of bounds' in str(e):
                    raise ContourOutOfBoundsException()
                raise e  # something serious is going on
                
        mask = sitk.GetImageFromArray(np_mask)

        mask = sitk.GetImageFromArray(np_mask)  # Avoid redundant calls by moving this here
        return mask
