from labelbox import Client

from uuid import uuid4 ## to generate unique IDs

import datetime
import os
from PIL import Image
import argparse
import fnmatch
import shutil

image_format = ('.jpg','.png','.bmp')


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
        client = Client(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja2x1bWtuc2hkYXB6MDc5MDY4bGw2bGd0Iiwib3JnYW5pemF0aW9uSWQiOiJja2x1bWtuczFhbXF3MDc3NHNyNGplMzRiIiwiYXBpS2V5SWQiOiJjbGRxYXNsbzMzMDEzMDd6NmdmajJncjlkIiwic2VjcmV0IjoiOTE4OTgyM2RiMjAwOGRhYjNlMjk0MDRjOWQ1NjZiYjkiLCJpYXQiOjE2NzU1MzU4NjcsImV4cCI6MjMwNjY4Nzg2N30.K7B4nfwPn7xQFbvCfoKJzVPHfzqfDU6UgOmtvM3HlEA")
        dataset = client.create_dataset(name="ENF_200_04_cam00")
        # Create data payload
        # Use global key, a unique ID to identify an asset throughout Labelbox workflow. Learn more: https://docs.labelbox.com/docs/global-keys
        # You can add metadata fields to your data rows. Learn more: https://docs.labelbox.com/docs/import-metadata
  

        try:
            task = dataset.create_data_rows(local_file_paths)
            task.wait_till_done()
        except Exception as err:
            print(f'Error while creating labelbox dataset -  Error: {err}')
    ## delte tmp files
    shutil.rmtree(os.path.join(dataDIR,'tmp'))


if __name__ == "__main__":

    dataDIR = os.path.join('.','datasets','00_Example-data','223_18_Duereth_ENF','AD0_10V_10kN','ENF_200_4','cam00')

    main(dataDIR)
