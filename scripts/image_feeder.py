
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("glob", type=str, help="glob-selection of files enclosed in quotes!")
parser.add_argument("nrank", type=int, help="number of mpiranks waiting to process images")
parser.add_argument("--maxProc", type=int, default=None, help="maximum number of images to process")
args = parser.parse_args()
import Pyro4

def main():

    for i_rank in range(args.nrank):
        img_monst = Pyro4.Proxy("PYRONAME:image.monster%d" % i_rank)
        img_monst.eat_images(args.glob, args.maxProc)

if __name__=="__main__":
    main()