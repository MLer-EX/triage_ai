import os

# نمایش دایرکتوری فعلی
current_directory = os.getcwd()

# تغییر دایرکتوری به والد
os.chdir('..')

new_directory = os.getcwd()

