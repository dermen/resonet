from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("logfile", type=str, help="training.log file")
    args = parser.parse_args()
    lines = open(args.logfile, 'r').readlines()

    train = []
    test = []
    train_loss = []
    test_loss = []

    for i,l in enumerate(lines):
        test_acc = train_acc = None
        if "test accuracy" in l:
            try:
                test_acc = float(lines[i+1].split(":")[-1].split("%")[0])
                test.append(test_acc)
                print(test_acc)
            except (ValueError, IndexError):
                print("failed")
                pass
        elif "train accuracy" in l:
            try:
                train_acc = float(lines[i+1].split(":")[-1].split("%")[0])
                print(train_acc, "train")
                train.append(train_acc)
            except (ValueError, IndexError):
                print("failed2")
                pass
        elif "Train loss" in l and "Test loss" in l:
            train_l = float(l.split("=")[1].split(",")[0])
            test_l = float( l.split("=")[-1] )
            print("train l, test l = %f, %f" % (train_l, test_l))
            train_loss.append(train_l)
            test_loss.append(test_l)

    import pylab as plt
    epochs = range(1, len(train_loss)+1)
    plot_kwargs = {'ms':6, 'ls':'-', 'lw': 2, }

    plt.subplot(121)
    if train:
        n = min(len(epochs), len(train), len(test))
        plt.plot(epochs[:n], train[:n],marker='o',color='C0', label="train", **plot_kwargs)
        plt.plot(epochs[:n], test[:n],marker='s',color='tomato',label="test", **plot_kwargs)
        plt.xlabel("epoch", fontsize=18)
        plt.ylabel("accuracy", fontsize=18)
        plt.gca().tick_params(labelsize=15)
        plt.gca().grid(1)
        plt.legend(prop={"size": 15})

    plt.subplot(122)
    plt.plot(epochs, train_loss,marker='o',color='C0', label="train", **plot_kwargs)
    plt.plot(epochs, test_loss,marker='s',color='tomato',label="test", **plot_kwargs)
    plt.xlabel("epoch", fontsize=18)
    plt.ylabel("loss", fontsize=18)
    plt.gca().tick_params(labelsize=15)
    plt.gca().grid(1)
    plt.legend(prop={"size": 15})

    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.99, wspace=0.33)
    w,h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches((1.3*w,h))
    plt.suptitle(args.logfile)
    plt.show()


if __name__=="__main__":
    main()
