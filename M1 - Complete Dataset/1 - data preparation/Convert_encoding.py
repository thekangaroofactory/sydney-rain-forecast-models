
# -- Convert file ending from ANSI to UTF-8
# see readme in the datasets folder

import codecs
import os.path
from os import listdir

BLOCKSIZE = 1048576 # or some other, desired size in bytes

input_path = '../../datasets/raw_archive/'
file_list = listdir(input_path)

output_path = '../../datasets/raw_utf8/'

for file in file_list:
    print('   Processing file: ', file)

    sourceFileName = os.path.join(input_path, file)
    targetFileName = os.path.join(output_path, file)

    with codecs.open(sourceFileName, "r", "ANSI") as sourceFile:
        with codecs.open(targetFileName, "w", "utf-8") as targetFile:
            while True:
                contents = sourceFile.read(BLOCKSIZE)
                if not contents:
                    break
                targetFile.write(contents)
