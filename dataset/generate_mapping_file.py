"""Generate Mapping File (json)
"""
import argparse
import json
import os
import sys
import logging
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Mapping File')
    parser.add_argument('--raw_dir', type=str, help='Raw data directory')
    parser.add_argument('--outfile', type=str, help='Output file')
    args = parser.parse_args()

    if not os.path.exists(args.raw_dir):
        print('Input file does not exist: {}'.format(args.raw_dir))
        sys.exit(1)
        
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s : %(message)s',
    )
    # walk through the raw directory
    mapping = {}
    total_sample_cnt = 0
    for dir in sorted(os.listdir(args.raw_dir)):
        if dir.startswith('.'):
            continue
        if not os.path.isdir(os.path.join(args.raw_dir, dir)):
            continue
        logger.info('Processing directory: {}'.format(dir))
        sample_cnt = 0
        for file in sorted(os.listdir(os.path.join(args.raw_dir, dir))):
            if not file.endswith('.jpg'):
                continue
            index = file.split('.')[0]
            total_index = dir + '_' + index
            mapping[total_index] = {}
            img_path = os.path.join(args.raw_dir, dir, file)
            txt_path = os.path.join(args.raw_dir, dir, index + '.txt')
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                caption = lines[0].strip('\n')
            mapping[total_index]['img_path'] = img_path
            mapping[total_index]['caption'] = caption
            sample_cnt += 1
        total_sample_cnt += sample_cnt
        logger.info('Extracting sample: {} | Total sample : {}'.format(sample_cnt, total_sample_cnt))
    
    with open(args.outfile, 'a') as f:
        json.dump(mapping, f)