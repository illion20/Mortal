import json
import socket

import torch
from config import config
from engine import MortalEngine
from libriichi.mjai import Bot
from model import DQN, Brain


def tile_str_to_name(hai_str):
    if hai_str == "1m":
        return "Characters (1)"
    elif hai_str == "2m":
        return "Characters (2)"
    elif hai_str == "3m":
        return "Characters (3)"
    elif hai_str == "4m":
        return "Characters (4)"
    elif hai_str == "5m":
        return "Characters (5)"
    elif hai_str == "6m":
        return "Characters (6)"
    elif hai_str == "7m":
        return "Characters (7)"
    elif hai_str == "8m":
        return "Characters (8)"
    elif hai_str == "9m":
        return "Characters (9)"
    elif hai_str == "5mr":
        return "Characters (5)*"
    elif hai_str == "1p":
        return "Dots (1)"
    elif hai_str == "2p":
        return "Dots (2)"
    elif hai_str == "3p":
        return "Dots (3)"
    elif hai_str == "4p":
        return "Dots (4)"
    elif hai_str == "5p":
        return "Dots (5)"
    elif hai_str == "6p":
        return "Dots (6)"
    elif hai_str == "7p":
        return "Dots (7)"
    elif hai_str == "8p":
        return "Dots (8)"
    elif hai_str == "9p":
        return "Dots (9)"
    elif hai_str == "5pr":
        return "Dots (5)*"
    elif hai_str == "1s":
        return "Bamboo (1)"
    elif hai_str == "2s":
        return "Bamboo (2)"
    elif hai_str == "3s":
        return "Bamboo (3)"
    elif hai_str == "4s":
        return "Bamboo (4)"
    elif hai_str == "5s":
        return "Bamboo (5)"
    elif hai_str == "6s":
        return "Bamboo (6)"
    elif hai_str == "7s":
        return "Bamboo (7)"
    elif hai_str == "8s":
        return "Bamboo (8)"
    elif hai_str == "9s":
        return "Bamboo (9)"
    elif hai_str == "5sr":
        return "Bamboo (5)*"
    elif hai_str == "E":
        return "East Wind"
    elif hai_str == "S":
        return "South Wind"
    elif hai_str == "W":
        return "West Wind"
    elif hai_str == "N":
        return "North Wind"
    elif hai_str == "P":
        return "White Dragon"
    elif hai_str == "F":
        return "Green Dragon"
    elif hai_str == "C":
        return "Red Dragon"
    else:
        return "?"


def print_formatted(data):
    if data["type"] == "none":
        return "Pass"
    elif data["type"] == "pon":
        return "Pon -> " + tile_str_to_name(data["pai"]) + " : " + tile_str_to_name(data["consumed"][0]) + ", " + tile_str_to_name(data["consumed"][1])
    elif data["type"] == "chi":
        return "Chi -> " + tile_str_to_name(data["pai"]) + " : " + tile_str_to_name(data["consumed"][0]) + ", " + tile_str_to_name(data["consumed"][1])
    elif data["type"] == "ankan":
        return "Ankan -> " + tile_str_to_name(data["consumed"][0]) + ", " + tile_str_to_name(data["consumed"][1]) + ", " + tile_str_to_name(data["consumed"][2]) + ", " + tile_str_to_name(data["consumed"][3])
    elif data["type"] == "kakan":
        return "Kakan -> " + tile_str_to_name(data["pai"]) + " : " + tile_str_to_name(data["consumed"][0]) + ", " + tile_str_to_name(data["consumed"][1]) + ", " + tile_str_to_name(data["consumed"][2])
    elif data["type"] == "daiminkan":
        return "Daiminkan -> " + tile_str_to_name(data["pai"]) + " : " + tile_str_to_name(data["consumed"][0]) + ", " + tile_str_to_name(data["consumed"][1]) + ", " + tile_str_to_name(data["consumed"][2])
    elif data["type"] == "reach":
        return "Riichi"
    elif data["type"] == "hora":
        return "Hora"
    elif data["type"] == "dahai":
        return "Discard -> " + tile_str_to_name(data["pai"])
    return json.dumps(data)


def main():
    player_id = 0
    device = torch.device('cpu')
    state = torch.load(config['control']['state_file'], map_location=torch.device('cpu'))
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    num_blocks = cfg['resnet']['num_blocks']
    conv_channels = cfg['resnet']['conv_channels']

    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])

    engine = MortalEngine(
        mortal,
        dqn,
        version = version,
        is_oracle=False,
        device=device,
        enable_amp=False,
        enable_quick_eval=True,
        enable_rule_based_agari_guard=True,
        name='mortal',
    )

    bot = Bot(engine, player_id)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("127.0.0.1", 11000))
        while True:
            line = ""
            while True:
                data = s.recv(1024)
                if not data:
                    break
                line += data.decode()
                if b"\n" in data:
                    break
            response = "{\"type\": \"ignore\"}"
            if not "option" in line:
                print(line, flush=True, end='')
                bot.react(line, can_act=False)
            else:
                if reaction := bot.get_reaction():
                    response = reaction
                    print(reaction, flush=True)
                    print('[\033[91m1\033[0m] : \033[96m' +
                          print_formatted(json.loads(reaction))+'\033[0m')
            # send back response
            msg = '[' + response + ']\n'
            s.sendall(msg.encode())

    finally:
        s.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
