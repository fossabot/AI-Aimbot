# main code for the agent
import sys
from utilz import utils_capture
import threading
from time import time, sleep
from agents import sis
from app import client
import os
import config
config = config.Config()

cwd = os.getcwd()
recording_folder = cwd+'/datasets/recording/'
test_folder = recording_folder+'/test/'
train_folder = recording_folder+'/train/'
model_folder = cwd+'/models/'

def record_session(duration, interval, mode, format):
    app = client.Client()
    t1 = threading.Thread(app.start(recording=True, mode=mode))
    t1.start()
    sc = utils_capture.InputRecord(recording_folder, config.capture_mode.get(mode)[0],config.capture_mode.get(mode)[1], interval=interval)
    t2 = threading.Thread(target=sc.begin_recording())
    t2.start()
    sleep(duration)
    sc.stop_recording(format)
    app.stop()
    return

def train_sis(mode):
    #training
    agent = sis.SIS(mode)
    agent.build_sis()
    return

def run():
    app = client.Client()
    app.start()
    return

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Welcome to AIAIM, please enter a command.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'help' 'record' 'train' or 'run'")
    #Recording                  
    parser.add_argument('--duration', required=False,
                        metavar="100",
                        help='Duration of the recording session in seconds')
    parser.add_argument('--interval', required=False,
                        metavar=".200",
                        help="Interval between screen captures in seconds")
    parser.add_argument('--mode', required=False,
                        metavar="lr",
                        help="The mode the images are captured as. 'wr' 'sr' or 'lr'")                      
    parser.add_argument('--format', required=False,
                        metavar="png",
                        help="The format the images are saved as. 'h5' or 'png'")


    #Training
    parser.add_argument('--network', required=False,
                        metavar="vae",
                        help='The network you want to train')
    parser.add_argument('--load_existing', required=False,
                        metavar="/path/to/existing/model.h5",
                        help='The path to the model you want to load.')

    #Eval
    parser.add_argument('--lol', required=False,
                        metavar="vae",
                        help='The network you want to evaluate')    

    args = parser.parse_args()

     # Validate arguments
    if args.command == "help":
        print('Hi from AIAIM!\n T Please refer to the readme instructions for setup.\nGo to the Quickstart section to see how to train your first agent!')
    elif args.command == "record":
        assert args.duration, "Argument --duration is required for recording"
        assert args.interval, "Argument --interval is required for recording"
        assert args.mode, "Argument --mode is required for recording"
        assert args.format, "Argument --format is required for recording"

        record_session(int(args.duration), float(args.interval), args.mode, args.format)

    elif args.command == "train":
        assert args.network, "Argument --network is required for training"

    elif args.command == "run":
        run()
