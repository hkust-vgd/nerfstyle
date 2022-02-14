import argparse
import trainers


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_cfg')
    parser.add_argument('--mode', choices=['pretrain', 'distill', 'finetune'],
                        default='pretrain')
    parser.add_argument('--name', default='tmp')
    parser.add_argument('--ckpt-path')
    parser.add_argument('--teacher-ckpt-path')

    args, nargs = parser.parse_known_args()
    trainer = trainers.get_trainer(args, nargs)
    trainer.run()


if __name__ == '__main__':
    train()
