import zipfile
zip_src=r'C:\nuts_sync\code_Space\learn_kaikeba\CCV4-adv\week1HomeWork.zip'
dst_dir=r'C:\nuts_sync\code_Space\learn_kaikeba\CCV4-adv\week1HomeWork'
r = zipfile.is_zipfile(zip_src)
if r:     
    fz = zipfile.ZipFile(zip_src, 'r')
    for file in fz.namelist():
        fz.extract(file, dst_dir)       
else:
    print('This is not zip')