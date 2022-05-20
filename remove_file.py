import os
import glob

#blade
blade_bgr_left = glob.glob("/home/user/shape_detection/blade/bgr/bgr_left/*.jpg")
blade_bgr_path_left = "/home/user/shape_detection/blade/bgr/bgr_left"
print('Sample Folder: ', os.listdir(blade_bgr_path_left))
for t in blade_bgr_left:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

blade_bgr_right = glob.glob("/home/user/shape_detection/blade/bgr/bgr_right/*.jpg")
blade_bgr_path_right = "/home/user/shape_detection/blade/bgr/bgr_right"
print('Sample Folder: ', os.listdir(blade_bgr_path_right))
for t in blade_bgr_right:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

blade_center = glob.glob("/home/user/shape_detection/blade/center/*.jpg")
blade_center_path = "/home/user/shape_detection/blade/center"
print('Sample Folder: ', os.listdir(blade_center_path))
for t in blade_center:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

blade_orig_left = glob.glob("/home/user/shape_detection/blade/orig/orig_left/*.jpg")
blade_orig_path_left = "/home/user/shape_detection/blade/orig/orig_left"
print('Sample Folder: ', os.listdir(blade_orig_path_left))
for t in blade_orig_left:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

blade_orig_right = glob.glob("/home/user/shape_detection/blade/orig/orig_right/*.jpg")
blade_orig_path_right= "/home/user/shape_detection/blade/orig/orig_right"
print('Sample Folder: ', os.listdir(blade_orig_path_right))
for t in blade_orig_right:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

blade_process = glob.glob("/home/user/shape_detection/blade/process/*.jpg")
blade_process_path = "/home/user/shape_detection/blade/process"
print('Sample Folder: ', os.listdir(blade_process_path))
for t in blade_process:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

blade_mat_left = glob.glob("/home/user/shape_detection/blade/mat/mat_left/*.jpg")
blade_mat_path_left = "/home/user/shape_detection/blade/mat/mat_left"
print('Sample Folder: ', os.listdir(blade_mat_path_left))
for t in blade_mat_left:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

blade_mat_right = glob.glob("/home/user/shape_detection/blade/mat/mat_right/*.jpg")
blade_mat_path_right = "/home/user/shape_detection/blade/mat/mat_right"
print('Sample Folder: ', os.listdir(blade_mat_path_right))
for t in blade_mat_right:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

#circle
circle_bgr_left = glob.glob("/home/user/shape_detection/circle/bgr/bgr_left/*.jpg")
circle_bgr_path_left = "/home/user/shape_detection/circle/bgr/bgr_left"
print('Sample Folder: ', os.listdir(circle_bgr_path_left))
for t in circle_bgr_left:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

circle_bgr_right = glob.glob("/home/user/shape_detection/circle/bgr/bgr_right/*.jpg")
circle_bgr_path_right = "/home/user/shape_detection/circle/bgr/bgr_right"
print('Sample Folder: ', os.listdir(circle_bgr_path_right))
for t in circle_bgr_right:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

circle_center = glob.glob("/home/user/shape_detection/circle/center/*.jpg")
circle_center_path = "/home/user/shape_detection/circle/center"
print('Sample Folder: ', os.listdir(circle_center_path))
for t in circle_center:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

circle_orig_left = glob.glob("/home/user/shape_detection/circle/orig/orig_left/*.jpg")
circle_orig_path_left = "/home/user/shape_detection/circle/orig/orig_left"
print('Sample Folder: ', os.listdir(circle_orig_path_left))
for t in circle_orig_left:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

circle_orig_right = glob.glob("/home/user/shape_detection/circle/orig/orig_right/*.jpg")
circle_orig_path_right = "/home/user/shape_detection/circle/orig/orig_right"
print('Sample Folder: ', os.listdir(circle_orig_path_right))
for t in circle_orig_right:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

circle_process = glob.glob("/home/user/shape_detection/circle/process/*.jpg")
circle_process_path = "/home/user/shape_detection/circle/process"
print('Sample Folder: ', os.listdir(circle_process_path))
for t in circle_process:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

circle_mat_left = glob.glob("/home/user/shape_detection/circle/mat/mat_left/*.jpg")
circle_mat_path_left = "/home/user/shape_detection/circle/mat/mat_left"
print('Sample Folder: ', os.listdir(circle_mat_path_left))
for t in circle_mat_left:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

circle_mat_right = glob.glob("/home/user/shape_detection/circle/mat/mat_right/*.jpg")
circle_mat_path_right = "/home/user/shape_detection/circle/mat/mat_right"
print('Sample Folder: ', os.listdir(circle_mat_path_right))
for t in circle_mat_right:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

#columnar
columnar_bgr_left = glob.glob("/home/user/shape_detection/columnar/bgr/bgr_left/*.jpg")
columnar_bgr_path_left = "/home/user/shape_detection/columnar/bgr/bgr_left"
print('Sample Folder: ', os.listdir(columnar_bgr_path_left))
for t in columnar_bgr_left:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

columnar_bgr_right = glob.glob("/home/user/shape_detection/columnar/bgr/bgr_right/*.jpg")
columnar_bgr_path_right = "/home/user/shape_detection/columnar/bgr/bgr_right"
print('Sample Folder: ', os.listdir(columnar_bgr_path_right))
for t in columnar_bgr_right:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

columnar_center = glob.glob("/home/user/shape_detection/columnar/center/*.jpg")
columnar_center_path = "/home/user/shape_detection/columnar/center"
print('Sample Folder: ', os.listdir(columnar_center_path))
for t in columnar_center:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

columnar_orig_left = glob.glob("/home/user/shape_detection/columnar/orig/orig_left/*.jpg")
columnar_orig_path_left = "/home/user/shape_detection/columnar/orig/orig_left"
print('Sample Folder: ', os.listdir(columnar_orig_path_left))
for t in columnar_orig_left:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

columnar_orig_right = glob.glob("/home/user/shape_detection/columnar/orig/orig_right/*.jpg")
columnar_orig_path_right = "/home/user/shape_detection/columnar/orig/orig_right"
print('Sample Folder: ', os.listdir(columnar_orig_path_right))
for t in columnar_orig_right:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

columnar_process = glob.glob("/home/user/shape_detection/columnar/process/*.jpg")
columnar_process_path = "/home/user/shape_detection/columnar/process"
print('Sample Folder: ', os.listdir(columnar_process_path))
for t in columnar_process:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

columnar_mat_right = glob.glob("/home/user/shape_detection/columnar/mat//mat_right/*.jpg")
columnar_mat_path_right = "/home/user/shape_detection/columnar/mat/mat_right"
print('Sample Folder: ', os.listdir(columnar_mat_path_right))
for t in columnar_mat_right:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

columnar_mat_left = glob.glob("/home/user/shape_detection/columnar/mat/mat_left/*.jpg")
columnar_mat_path_left = "/home/user/shape_detection/columnar/mat/mat_left"
print('Sample Folder: ', os.listdir(columnar_mat_path_left))
for t in columnar_mat_left:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

#long
long_bgr_left = glob.glob("/home/user/shape_detection/long/bgr/bgr_left/*.jpg")
long_bgr_path_left = "/home/user/shape_detection/long/bgr/bgr_left"
print('Sample Folder: ', os.listdir(long_bgr_path_left))
for t in long_bgr_left:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

long_bgr_right = glob.glob("/home/user/shape_detection/long/bgr/bgr_right/*.jpg")
long_bgr_path_right = "/home/user/shape_detection/long/bgr/bgr_right"
print('Sample Folder: ', os.listdir(long_bgr_path_right))
for t in long_bgr_right:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

long_center = glob.glob("/home/user/shape_detection/long/center/*.jpg")
long_center_path = "/home/user/shape_detection/long/center"
print('Sample Folder: ', os.listdir(long_center_path))
for t in long_center:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

long_orig_left = glob.glob("/home/user/shape_detection/long/orig/orig_left/*.jpg")
long_orig_path_left = "/home/user/shape_detection/long/orig/orig_left"
print('Sample Folder: ', os.listdir(long_orig_path_left))
for t in long_orig_left:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

long_orig_right = glob.glob("/home/user/shape_detection/long/orig/orig_right/*.jpg")
long_orig_path_right = "/home/user/shape_detection/long/orig/orig_right"
print('Sample Folder: ', os.listdir(long_orig_path_right))
for t in long_orig_right:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

long_process = glob.glob("/home/user/shape_detection/long/process/*.jpg")
long_process_path = "/home/user/shape_detection/long/process"
print('Sample Folder: ', os.listdir(long_process_path))
for t in long_process:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

long_mat_left = glob.glob("/home/user/shape_detection/long/mat/mat_left/*.jpg")
long_mat_path_left = "/home/user/shape_detection/long/mat/mat_left"
print('Sample Folder: ', os.listdir(long_mat_path_left))
for t in long_mat_left:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

long_mat_right = glob.glob("/home/user/shape_detection/long/mat/mat_right/*.jpg")
long_mat_path_right = "/home/user/shape_detection/long/mat/mat_right"
print('Sample Folder: ', os.listdir(long_mat_path_right))
for t in long_mat_right:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)



