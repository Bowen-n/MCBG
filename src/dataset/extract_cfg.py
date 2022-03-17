# created by Wu Bolun
# 2020.9.29
# bowenwu@sjtu.edu.cn

import os

from asm import *


if __name__ == '__main__':
    big2015_dir = '/home/wubolun/data/malware/big2015/train'
    store_dir = '/home/wubolun/data/malware/big2015/further/cfg'
    file_format = 'json'

    with open('../../log/empty_code.err', 'r') as f:
        empty_code_ids = f.read().split('\n')

    count = 0
    file_list = os.listdir(big2015_dir)
    file_list = list(filter(lambda x: '.asm' in x, file_list))
    for filepath in file_list:
        count += 1
        # if '.asm' not in filepath:
        #     continue

        print('{}/{} File: {}. Info: '.format(count, len(file_list), filepath), end='')

        binary_id = filepath.split('.')[0]
        if binary_id in empty_code_ids:
            print('Empty code.')
            continue

        store_path = os.path.join(store_dir, '{}.{}'.format(binary_id, file_format))
        if os.path.exists(store_path):
            print('Already parsed.')
            continue

        parser = AsmParser(directory=big2015_dir, binary_id=binary_id)
        success = parser.parse()
        if success:
            parser.store_blocks(store_path, fformat=file_format)
            print('Success.')
        else:
            print('Empty code or block after parse.')

