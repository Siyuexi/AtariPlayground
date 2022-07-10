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
    parser.add_argument("--ddqn", type=bool, default=True, help="using ddqn structure.")
    parser.add_argument("--deepmind", type=bool, default=True, help="using deepmind wrapper.")
    parser.add_argument("--lite", type=bool, default=True, help="using lite buffer")
    parser.add_argument("--train_start", type=int, default=5000, help="training starts after x epoch of experience collecting.")
    parser.add_argument("--test_start", type=int, default=10, help="testing starts at x epoch of checkpoints.")

    return parser

def main(args):

    # initialization
    try:
        # parsing yaml settings
        game = args.game
        mode = args.mode
        train_start = args.train_start
        test_start = args.test_start
        param = yaml.load(open("param.yaml"), Loader=yaml.SafeLoader)[game]
        
        # building infrastructure
        net = model.get_net(param['type'], param['n_actions'], param['n_states'], args.ddqn)
        env = wrap.get_env(param['name'], param['noop_max'], param['skip'], param['width'], param['height'], param['n_states'], param['seed'], args.deepmind)
        mem = buffer.get_mem(param['memory_capacity'], param['width'], param['height'], param['n_states'], args.lite)
        hp = param['hyper_params']

    except Exception as e:
        traceback.print_exc()
        print("ILLEGAL PARAMETERS. PLEASE CHECK 'param.yaml' TO GET LEGAL PARAMETERS.")
        exit()

    # creating basic paths
    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint/temp")
    if not os.path.exists("./log"):
        os.makedirs("./log/uru")
        os.makedirs("./log/viz")

    # game begin
    player = agent.Player(net, env, mem, hp, game, mode, train_start, test_start)
    player.execute()


if __name__ == '__main__':
    args = args().parse_args()

    # args configuration
    assert args.game is not None, "YOU HAVE NOT CHOSEN A GAME YET."
    assert args.mode is not None, "YOU HAVE NOT CHOSEN A MODE YET."
    assert args.mode == 'train' or args.mode == 'test', "ILLEGAL MODE:TRAIN OR TEST."
    
    main(args)