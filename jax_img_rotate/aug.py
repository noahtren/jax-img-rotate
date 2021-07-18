import math

import jax
import jax.numpy as np
from jax import random


def get_mat(rotation, shear, height_zoom, width_zoom, height_shift,
            width_shift):
  """Return a 3x3 transformmatrix which transforms indicies of original images
  """

  # CONVERT DEGREES TO RADIANS
  rotation = math.pi * rotation / 180.
  shear = math.pi * shear / 180.

  one = np.array([1])
  zero = np.array([0])
  # ROTATION MATRIX
  c1 = np.cos(rotation)
  s1 = np.sin(rotation)
  rotation_matrix = np.reshape(
      np.concatenate([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0),
      [3, 3])

  # SHEAR MATRIX
  c2 = np.cos(shear)
  s2 = np.sin(shear)
  shear_matrix = np.reshape(
      np.concatenate([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0),
      [3, 3])

  # ZOOM MATRIX
  zoom_matrix = np.reshape(
      np.concatenate([
          one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero,
          zero, one
      ],
                     axis=0), [3, 3])

  # SHIFT MATRIX
  shift_matrix = np.reshape(
      np.concatenate(
          [one, zero, height_shift, zero, one, width_shift, zero, zero, one],
          axis=0), [3, 3])

  return np.matmul(np.matmul(rotation_matrix, shear_matrix),
                   np.matmul(zoom_matrix, shift_matrix))


def transform_batch(rng, images, max_rot_deg, max_shear_deg, zoom_min,
                    zoom_max, max_shift_pct):
  """Transform a batch of square images with the same randomized affine
  transformation.
  """
  def clipped_random(rng):
    use_rng, rng = random.split(rng)
    rand = random.normal(use_rng, [1], dtype=np.float32)
    rand = np.clip(rand, -2., 2.) / 2.
    return rand, rng

  batch_size = images.shape[0]
  DIM = images.shape[1]
  channels = images.shape[3]
  XDIM = DIM % 2

  rot, rng = clipped_random(rng)
  rot = rot * max_rot_deg
  shr, rng = clipped_random(rng)
  shr = shr * max_shear_deg
  h_zoom, rng = clipped_random(rng)
  h_zoom = 1.0 + np.abs(h_zoom) * ((zoom_max - zoom_min) + zoom_min)
  w_zoom, rng = clipped_random(rng)
  w_zoom = 1.0 + np.abs(w_zoom) * ((zoom_max - zoom_min) + zoom_min)
  h_shift, rng = clipped_random(rng)
  h_shift = h_shift * (DIM * max_shift_pct)
  w_shift, rng = clipped_random(rng)
  w_shift = w_shift * (DIM * max_shift_pct)

  # GET TRANSFORMATION MATRIX
  m = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

  # LIST DESTINATION PIXEL INDICES
  x = np.repeat(np.arange(DIM // 2, -DIM // 2, -1), DIM)  # 10000,
  y = np.tile(np.arange(-DIM // 2, DIM // 2), [DIM])
  z = np.ones([DIM * DIM], np.int32)
  idx = np.stack([x, y, z])  # [3, 10000]

  # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
  idx2 = np.matmul(m, idx.astype(np.float32))
  idx2 = idx2.astype(np.int32)
  idx2 = np.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

  # FIND ORIGIN PIXEL VALUES
  idx3 = np.stack([DIM // 2 - idx2[0, ], DIM // 2 - 1 + idx2[1, ]])
  idx3 = idx3.T

  idx4 = idx3[:, 0] * DIM + idx3[:, 1]
  images = np.reshape(images, [batch_size, DIM * DIM, channels])
  # d = jax.lax.gather(images, idx4, axis=1)
  d = images[:, idx4]
  return np.reshape(d, [batch_size, DIM, DIM, channels])
