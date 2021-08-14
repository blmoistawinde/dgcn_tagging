import numpy as np
import argparse
import csv
import os
import glob
import datetime
import time
import logging
import h5py
import librosa

from utilities import (create_folder, get_filename, create_logging, 
    float32_to_int16, pad_or_truncate, read_metadata)
import config


def download_wavs(args):
    """Download videos and extract audio in wav format.
    """

    # Paths
    csv_path = args.csv_path
    audios_dir = args.audios_dir
    mini_data = args.mini_data
    
    if mini_data:
        logs_dir = '_logs/download_dataset/{}'.format(get_filename(csv_path))
    else:
        logs_dir = '_logs/download_dataset_minidata/{}'.format(get_filename(csv_path))
    
    create_folder(audios_dir)
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Download log is saved to {}'.format(logs_dir))

    # Read csv
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    lines = lines[3:]   # Remove csv head info

    if mini_data:
        lines = lines[0 : 10]   # Download partial data for debug
    
    # download_time = time.time()

    # Download
    for (n, line) in enumerate(lines):
        
        items = line.split(', ')
        audio_id = items[0]
        start_time = items[1]
        end_time = items[2]
        # duration = end_time - start_time
        
        prev_path = os.path.join('/backup/data/Audioset/audiodata', '{}_{}_{}.wav'.format(audio_id, start_time, end_time))
        curr_path = os.path.join(audios_dir, 'Y' + audio_id + '.wav')

        os.system('mv {} {}'.format(prev_path, curr_path))

        # logging.info('{} {} start_time: {:.1f}, end_time: {:.1f}'.format(
        #     n, audio_id, start_time, end_time))
        
        # Download full video of whatever format
        # video_name = os.path.join(audios_dir, '_Y{}.%(ext)s'.format(audio_id))
        # os.system("youtube-dl --quiet -o '{}' -x https://www.youtube.com/watch?v={}"\
        #     .format(video_name, audio_id))

        # video_paths = glob.glob(os.path.join(audios_dir, '_Y' + audio_id + '.*'))

        # If download successful
        # if len(video_paths) > 0:
        #     video_path = video_paths[0]     # Choose one video

        #     # Add 'Y' to the head because some video ids are started with '-'
        #     # which will cause problem
        #     audio_path = os.path.join(audios_dir, 'Y' + audio_id + '.wav')

        #     # Extract audio in wav format
        #     os.system("ffmpeg -loglevel panic -i {} -ac 1 -ar 16000 -ss {} -t 00:00:{} {} "\
        #         .format(video_path, 
        #         str(datetime.timedelta(seconds=start_time)), duration, 
        #         audio_path))
            
        #     # Remove downloaded video
        #     os.system("rm {}".format(video_path))
            
            # logging.info("Download and convert to {}".format(audio_path))
                
    # logging.info('Download finished! Time spent: {:.3f} s'.format(
    #     time.time() - download_time))

    # logging.info('Logs can be viewed in {}'.format(logs_dir))
          

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_split = subparsers.add_parser('split_unbalanced_csv_to_partial_csvs')
    parser_split.add_argument('--unbalanced_csv', type=str, required=True, help='Path of unbalanced_csv file to read.')
    parser_split.add_argument('--unbalanced_partial_csvs_dir', type=str, required=True, help='Directory to save out split unbalanced partial csv.')

    parser_download_wavs = subparsers.add_parser('download_wavs')
    parser_download_wavs.add_argument('--csv_path', type=str, required=True, help='Path of csv file containing audio info to be downloaded.')
    parser_download_wavs.add_argument('--audios_dir', type=str, required=True, help='Directory to save out downloaded audio.')
    parser_download_wavs.add_argument('--mini_data', action='store_true', default=False, help='Set true to only download 10 audios for debugging.')

    parser_pack_wavs = subparsers.add_parser('pack_waveforms_to_hdf5')
    parser_pack_wavs.add_argument('--csv_path', type=str, required=True, help='Path of csv file containing audio info to be downloaded.')
    parser_pack_wavs.add_argument('--audios_dir', type=str, required=True, help='Directory to save out downloaded audio.')
    parser_pack_wavs.add_argument('--waveforms_hdf5_path', type=str, required=True, help='Path to save out packed .')
    parser_pack_wavs.add_argument('--mini_data', action='store_true', default=False, help='Set true to only download 10 audios for debugging.')

    args = parser.parse_args()
    
    if args.mode == 'split_unbalanced_csv_to_partial_csvs':
        split_unbalanced_csv_to_partial_csvs(args)
    
    elif args.mode == 'download_wavs':
        download_wavs(args)

    elif args.mode == 'pack_waveforms_to_hdf5':
        pack_waveforms_to_hdf5(args)

    else:
        raise Exception('Incorrect arguments!')
