import os
import sys
import platform
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging
import traceback
import util
# Intil fixed in Keras, this is needed to remove a wrong warning
import warnings
warnings.filterwarnings("ignore")

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)

# Configure absl logging to suppress logs
import absl.logging
# Suppress Abseil logs
absl.logging.get_absl_handler().python_handler.stream = open(os.devnull, 'w')
absl.logging.set_verbosity(absl.logging.FATAL)
absl.logging.set_stderrthreshold(absl.logging.FATAL)

import tensorflow as tf

import time
import datetime
import asyncio
import websockets
import argparse
import game
import human
import conf
import functools
import numpy as np
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from sample import Sample
from urllib.parse import parse_qs, urlparse
from pbn2ben import load

from colorama import Fore, Back, Style, init
import gc
import psutil

init()

# Custom function to convert string to boolean
def str_to_bool(value):
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    raise ValueError("Invalid boolean value")

def handle_exception(e):
    sys.stderr.write(f"{str(e)}\n")
    traceback_str = traceback.format_exception(type(e), e, e.__traceback__)
    traceback_lines = "".join(traceback_str).splitlines()
    file_traceback = []
    for line in reversed(traceback_lines):
        if line.startswith("  File"):
            file_traceback.append(line.strip()) 
    if file_traceback:
        sys.stderr.write(f"{Fore.RED}")
        sys.stderr.write('\n'.join(file_traceback)+'\n')
        sys.stderr.write(f"{Fore.RESET}")

def log_memory_usage():
    # Get system memory info
    virtual_memory = psutil.virtual_memory()
    available_memory = virtual_memory.available / (1024 ** 2)  # Convert bytes to MB
    print(f"Available memory before request: {available_memory:.2f} MB")

def get_execution_path():
    # Get the directory where the program is started from either PyInstaller executable or the script
    return os.getcwd()

random = True
#For some strange reason parameters parsed to the handler must be an array
board_no = []
board_no.append(0) 

# Get the path to the config file
config_path = get_execution_path()

parser = argparse.ArgumentParser(description="Game server")
parser.add_argument("--boards", default="", help="Filename for boards")
parser.add_argument("--boardno", default=0, type=int, help="Board number to start from")
parser.add_argument("--config", default=f"{config_path}/config/default.conf", help="Filename for configuration")
parser.add_argument("--opponent", default="", help="Filename for configuration pf opponents")
parser.add_argument("--verbose", type=str_to_bool, default=False, help="Output samples and other information during play")
parser.add_argument("--port", type=int, default=4443, help="Port for appserver")
parser.add_argument("--auto", type=bool, default=False, help="BEN bids and plays all 4 hands")
parser.add_argument("--playonly", type=str_to_bool, default=False, help="Only play, no bidding")
parser.add_argument("--matchpoint", type=str_to_bool, default=None, help="Playing match point")
parser.add_argument("--seed", type=int, default=42, help="Seed for random")

args = parser.parse_args()

configfile = args.config
opponentfile = args.opponent
verbose = args.verbose
port = args.port
auto = args.auto
play_only = args.playonly
matchpoint = args.matchpoint
seed = args.seed
boards = []

np.set_printoptions(precision=2, suppress=True, linewidth=200)

print(f"{Fore.CYAN}{datetime.datetime.now():%Y-%m-%d %H:%M:%S} gameserver.py - Version 0.8.6.8")
if util.is_pyinstaller_executable():
    print(f"Running inside a PyInstaller-built executable. {platform.python_version()}")
else:
    print(f"Running in a standard Python environment: {platform.python_version()}")

print(f"Python version: {sys.version}{Fore.RESET}")

if sys.platform == 'win32':
    # Print the PythonNet version
    sys.stderr.write(f"PythonNet: {util.get_pythonnet_version()}\n") 
    sys.stderr.write(f"{util.check_dotnet_version()}\n") 

# Try to fetch Keras version or handle older TensorFlow versions
try:
    keras_version = tf.keras.__version__
except AttributeError:
    keras_version = "Not integrated with TensorFlow"
    configfile = configfile.replace("default.conf", "TF1.x/default_tf1x.conf")

# Write to stderr
sys.stderr.write(f"Loading TensorFlow {tf.__version__} - Keras version: {keras_version}\n")

configuration = conf.load(configfile)


try:
    if (configuration["models"]['tf_version'] == "2"):
        from nn.models_tf2 import Models
    else: 
        # Default to version 1. of Tensorflow
        from nn.models import Models
except KeyError:
        # Default to version 1. of Tensorflow
        from nn.models import Models

models = Models.from_conf(configuration, config_path.replace(os.path.sep + "src",""))

if sys.platform != 'win32':
    print("Disabling PIMC/BBA/SuitC as platform is not win32")
    models.pimc_use_declaring = False
    models.pimc_use_defending = False
    models.use_bba = False
    models.consult_bba = False
    models.use_bba_rollouts = False
    models.use_bba_to_count_aces = False
    models.use_suitc = False
    
print("Config:", configfile)
print("System:", models.name)

if opponentfile != "":
    # Override with information from opponent file
    print("Opponent:", opponentfile)
    opp_configuration = conf.load(opponentfile)
    opponents = Opponents.from_conf(opp_configuration, config_path.replace(os.path.sep + "src",""))
    models.opponent_model = opponents.opponent_model
    models.bba_their_cc = opponents.bba_their_cc
    sys.stderr.write(f"Expecting opponent: {opponents.name}\n")

if models.use_bba:
    print("Using BBA for bidding")
else:
    print("Model:   ", models.bidder_model.model_path)
    print("Opponent:", models.opponent_model.model_path)

if matchpoint is not None:
    models.matchpoint = matchpoint

if models.matchpoint:
    print("Matchpoint mode on")
else:
    print("Playing IMPS mode")

if models.use_bba or models.use_bba_to_count_aces or models.consult_bba or models.use_bba_rollout:
    from bba.BBA import BBABotBid
    bot = BBABotBid(None, None ,None, None, None, None, None, None)
    print(f"BBA enabled. Version {bot.version()}")    

if models.use_suitc:
    from suitc.SuitC import SuitCLib
    suitc = SuitCLib(verbose)
    print(f"SuitC enabled. Version {suitc.version()}")

if models.pimc_use_declaring or models.pimc_use_defending:
    from pimc.PIMC import BGADLL
    pimc = BGADLL(None, None, None, None, None, None, None)
    from pimc.PIMCDef import BGADefDLL
    pimcdef = BGADefDLL(None, None, None, None, None, None, None, None)
    print(f"PIMC enabled. Version {pimc.version()}")
    print(f"PIMCDef enabled. Version {pimcdef.version()}")

from ddsolver import ddsolver
dds = ddsolver.DDSolver()
print(f"DDSolver enabled. Version {dds.version()}")

if args.boards:
    filename = args.boards
    file_extension = os.path.splitext(filename)[1].lower()  
    if file_extension == '.ben':
        with open(filename, "r") as file:
            board_no.append(0) 
            lines = file.readlines()  # 
            # Loop through the lines, grouping them into objects
            for i in range(0, len(lines), 2):
                board = {
                    'deal': lines[i].strip(),      
                    'auction': lines[i+1].strip().replace('NT','N')  
                }
                boards.append(board)            
            print(f"{len(boards)} boards loaded from file")
        random = False
    if file_extension == '.pbn':
        with open(filename, "r") as file:
            lines = file.readlines()
            boards = load(lines)
            print(f"{len(boards)} boards loaded from file")
        random = False

if args.boardno:
    print(f"Starting from {args.boardno}")
    board_no[0] = args.boardno -1

if random:
    print("Playing random deals or deals from the client")

def worker(driver):
    print('worker', driver)
    asyncio.new_event_loop().run_until_complete(driver.run())


async def handler(websocket, path, board_no, seed):
    print('{} Got websocket connection'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    driver = game.Driver(models, human.WebsocketFactory(websocket, verbose), Sample.from_conf(configuration, verbose), seed, dds, verbose)
    play_only = False
    driver.human = [False, False, False, False]
    parsed_url = urlparse(path)
    query_params = parse_qs(parsed_url.query)
    deal = None
    N = query_params.get('N', [None])[0]
    if N: driver.human[0] = True
    E = query_params.get('E', [None])[0]
    if E: driver.human[1] = True
    S = query_params.get('S', [None])[0]
    if S: driver.human[2] = True
    W = query_params.get('W', [None])[0]
    if W: driver.human[3] = True
    H = query_params.get('H', [None])[0]
    if H: driver.human_declare = True
    name = query_params.get('name', [None])[0]
    if name: driver.name = name
    R = query_params.get('R', [None])[0]
    if R: driver.rotate = True
    M = query_params.get('M', [None])[0]
    if M: 
        models.matchpoint = True
    else:
         models.matchpoint = False
    P = query_params.get('P', [None])[0]
    if P == "5":
        play_only = True
    deal = query_params.get('deal', [None])[0]
    board_no_query = query_params.get('board_no')
    board_number = None
    if board_no_query is not None and board_no_query[0] != "null" and board_no_query[0] != "None":
        board_number = int(board_no_query[0]) 
    else:
        if not deal and not board_no[0] > 0:
            board_number = np.random.randint(1, 1000)

    # If deal provided in the URL
    if deal:
        if board_number == None:
            board_number = np.random.randint(1, 1000)
        np.random.seed(board_number)
        split_values = deal[1:-1].replace("'","").split(',')
        rdeal = tuple(value.strip() for value in split_values)
        driver.set_deal(board_number, *rdeal, play_only)
        print(f"Board: {board_number} {rdeal} {play_only}")
    else: 
        # If random
        if random:
            #Just take a random"
            np.random.seed(board_number)
            rdeal = game.random_deal_board(board_number)
            # example of to use a fixed deal
            # rdeal = ('AK64.8642.Q32.Q6 9.QT973.AT5.KJ94 QT532.J5.KJ974.7 J87.AK.86.AT8532', 'W None')
            print(f"Board: {board_number} {rdeal}")
            driver.set_deal(board_number, *rdeal, False)
        else:
            # Select the next from the provided inputfile
            rdeal = boards[board_no[0]]['deal']
            auction = boards[board_no[0]]['auction']
            print(f"{Fore.LIGHTBLUE_EX}Board: {board_no[0]+1} {rdeal}{Fore.RESET}")
            np.random.seed(board_no[0]+1)
            driver.set_deal(board_no[0] + 1, rdeal, auction, play_only)

    log_memory_usage()
    try:
        t_start = time.time()
        await driver.run(t_start)

        print(f'{Fore.CYAN}{datetime.datetime.now():%Y-%m-%d %H:%M:%S} Board played in {time.time() - t_start:0.1f} seconds.{Fore.RESET}')  
        if not random and len(boards) > 0:
            board_no[0] = (board_no[0] + 1) % len(boards)
        gc.collect()
        log_memory_usage()

    except (ConnectionClosedOK, ConnectionClosedError, ConnectionAbortedError):
        print('User left')
    except ValueError as e:
        print("Error in configuration - typical the models do not match the configuration.")
        handle_exception(e)
        sys.exit(1)

async def main():
    sys.stderr.write(f"{Fore.CYAN}{datetime.datetime.now():%Y-%m-%d %H:%M:%S} Listening on port: {port}{Fore.RESET}\n")
    start_server = websockets.serve(functools.partial(handler, board_no=board_no, seed=seed), "0.0.0.0", port, 
        ping_timeout=60,  # 60 seconds timeout for pings
        close_timeout=60  # 60 seconds timeout for closing the connection
        )
    try:
        await start_server
    except OSError as e:
        if e.errno == 10048:
            print("Port is already in use.  - Wait or terminate the process.")
        else:
            raise
    except Exception as e:
        print("Error starting server.")
        handle_exception(e)
        sys.exit(1)

if __name__ == "__main__":
    print(Back.BLACK)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        handle_exception(e)
        sys.exit(1)
    finally:
        loop.close()
        print(Style.RESET_ALL)