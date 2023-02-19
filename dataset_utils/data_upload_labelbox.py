from labelbox import Client

from uuid import uuid4 ## to generate unique IDs

import datetime
import os
from PIL import Image
import argparse
import fnmatch
import shutil

image_format = ('.jpg','.png','.bmp','dcm')


def main(dataDIR):
    ## check input data
    local_file_paths = [] # limit: 15k files
    for file in os.listdir(dataDIR):
        if file.lower().endswith(image_format):
            local_file_paths.append(os.path.join(dataDIR,file))
        else:
            if not os.path.isdir(os.path.join(dataDIR,'tmp')):
                os.mkdir(os.path.join(dataDIR,'tmp'))

            file_tmp, ext = os.path.splitext(file)
            with Image.open(os.path.join(dataDIR,file)) as im:
                im.save(os.path.join(dataDIR,'tmp',file_tmp+'.png') , "png")
            local_file_paths.append(os.path.join(dataDIR,'tmp',file_tmp+'.png'))

    print(local_file_paths)
    if local_file_paths:
        with open("dataset_utils/api_key.txt") as LB_API_KEY:
            client = Client(api_key=LB_API_KEY.readlines()[0])
        print('API-key valid')
        
        dataset = client.create_dataset(name="neapel_001")
        # Create data payload
        # Use global key, a unique ID to identify an asset throughout Labelbox workflow. Learn more: https://docs.labelbox.com/docs/global-keys
        # You can add metadata fields to your data rows. Learn more: https://docs.labelbox.com/docs/import-metadata
  

        try:
            task = dataset.create_data_rows(local_file_paths)
            task.wait_till_done()
        except Exception as err:
            print(f'Error while creating labelbox dataset -  Error: {err}')
    ## delte tmp files
    if os.path.isdir(os.path.join(dataDIR,'tmp')):
        shutil.rmtree(os.path.join(dataDIR,'tmp'))


if __name__ == "__main__":

    dataDIR = os.path.join('.','data','Neapel','neapel_001','DICOM')

    main(dataDIR)
