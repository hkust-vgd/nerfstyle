import argparse
from trainers import get_trainer


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_cfg')
    parser.add_argument('name')
    parser.add_argument('--run-dir', default='./runs')
    parser.add_argument('--mode', choices=['pretrain', 'distill', 'finetune'],
                        default='pretrain')
    parser.add_argument('--ckpt-path')
    parser.add_argument('--teacher-ckpt-path')
    parser.add_argument('--occ-map')

    args, nargs = parser.parse_known_args()
    trainer = get_trainer(args, nargs)

    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.logger.info('Training interrupted')
    finally:
        trainer.close()


if __name__ == '__main__':
    train()
