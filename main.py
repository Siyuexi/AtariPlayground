# general libs
import argparse
import os
import yaml
import traceback
# self-defined
import agent
import model
import wrap
import buffer
"""
***ENTRANCE HERE***
TORCH:1.9.0
GYM:0.19.0
CREATED BY SIYUEXI
2022.07.01
"""
def args():
    parser = argparse.ArgumentParser("CHOOSE YOUR GAME")
    parser.add_argument("-g","--game", type=str, default=None, help="The game that you wanna play.")
    parser.add_argument("-m","--mode", type=str, default=None, help="Training or Testing?")
    parser.add_argument("-i", "--id", type=str, default="", help="special label for a training task")
    parser.add_argument("--ddqn", type=bool, default=True, help="using ddqn structure.")
    parser.add_argument("--deepmind", type=bool, default=True, help="using deepmind wrapper.")
    parser.add_argument("--lite", type=bool, default=True, help="using lite buffer.")

    return parser

def main(args):

    # initialization
    try:
        # parsing yaml settings
        id = args.id
        game = args.game
        mode = args.mode
        ddqn = args.ddqn
        param = yaml.load(open("param.yaml"), Loader=yaml.SafeLoader)[game]
        
        # building infrastructure
        net = model.get_net(param['type'], param['n_actions'], param['n_states'])
        env = wrap.get_env(param['name'], param['noop_max'], param['skip'], param['width'], param['height'], param['n_states'], args.deepmind)
        mem = buffer.get_mem(param['memory_capacity'], param['width'], param['height'], param['n_states'], args.lite)
        hp = param['hyper_params']

    except Exception as e:
        traceback.print_exc()
        print("ILLEGAL PARAMETERS. PLEASE CHECK 'param.yaml' TO GET LEGAL PARAMETERS.")
        exit()

    # creating basic paths
    if not os.path.exists("./checkpoint/temp"):
        os.makedirs("./checkpoint/temp")
    if not os.path.exists("./log/uru"):
        os.makedirs("./log/uru")
    if not os.path.exists("./log/viz"):
        os.makedirs("./log/viz")

    # game begin
    player = agent.Player(id, net, env, mem, hp, game, mode, ddqn)
    player.execute()


if __name__ == '__main__':
    args = args().parse_args()

    # args configuration
    assert args.game is not None, "YOU HAVE NOT CHOSEN A GAME YET."
    assert args.mode is not None, "YOU HAVE NOT CHOSEN A MODE YET."
    assert args.mode == 'train' or args.mode == 'test', "ILLEGAL MODE:TRAIN OR TEST."
    
    main(args)