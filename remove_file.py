import os
import glob

#blade
blade_bgr = glob.glob("/home/user/shape_detection/blade/bgr/*.jpg")
blade_bgr_path = "/home/user/shape_detection/blade/bgr"
print('Sample Folder: ', os.listdir(blade_bgr_path))
for t in blade_bgr:
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

blade_orig = glob.glob("/home/user/shape_detection/blade/orig/*.jpg")
blade_orig_path = "/home/user/shape_detection/blade/orig"
print('Sample Folder: ', os.listdir(blade_orig_path))
for t in blade_orig:
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

blade_mat = glob.glob("/home/user/shape_detection/blade/mat/*.jpg")
blade_mat_path = "/home/user/shape_detection/blade/mat"
print('Sample Folder: ', os.listdir(blade_mat_path))
for t in blade_mat:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

#circle
circle_bgr = glob.glob("/home/user/shape_detection/circle/bgr/*.jpg")
circle_bgr_path = "/home/user/shape_detection/circle/bgr"
print('Sample Folder: ', os.listdir(circle_bgr_path))
for t in circle_bgr:
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

circle_orig = glob.glob("/home/user/shape_detection/circle/orig/*.jpg")
circle_orig_path = "/home/user/shape_detection/circle/orig"
print('Sample Folder: ', os.listdir(circle_orig_path))
for t in circle_orig:
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

circle_mat = glob.glob("/home/user/shape_detection/circle/mat/*.jpg")
circle_mat_path = "/home/user/shape_detection/circle/mat"
print('Sample Folder: ', os.listdir(circle_mat_path))
for t in circle_mat:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

#columnar
columnar_bgr = glob.glob("/home/user/shape_detection/columnar/bgr/*.jpg")
columnar_bgr_path = "/home/user/shape_detection/columnar/bgr"
print('Sample Folder: ', os.listdir(columnar_bgr_path))
for t in columnar_bgr:
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

columnar_orig = glob.glob("/home/user/shape_detection/columnar/orig/*.jpg")
columnar_orig_path = "/home/user/shape_detection/columnar/orig"
print('Sample Folder: ', os.listdir(columnar_orig_path))
for t in columnar_orig:
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

columnar_mat = glob.glob("/home/user/shape_detection/columnar/mat/*.jpg")
columnar_mat_path = "/home/user/shape_detection/columnar/mat"
print('Sample Folder: ', os.listdir(columnar_mat_path))
for t in columnar_mat:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)

#long
long_bgr = glob.glob("/home/user/shape_detection/long/bgr/*.jpg")
long_bgr_path = "/home/user/shape_detection/long/bgr"
print('Sample Folder: ', os.listdir(long_bgr_path))
for t in long_bgr:
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

long_orig = glob.glob("/home/user/shape_detection/long/orig/*.jpg")
long_orig_path = "/home/user/shape_detection/long/orig"
print('Sample Folder: ', os.listdir(long_orig_path))
for t in long_orig:
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

long_mat = glob.glob("/home/user/shape_detection/long/mat/*.jpg")
long_mat_path = "/home/user/shape_detection/long/mat"
print('Sample Folder: ', os.listdir(long_mat_path))
for t in long_mat:
  try:
    os.remove(t)
  except OSError as e:
    print('Delete Problem: ', e)



