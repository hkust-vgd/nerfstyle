from config import BaseConfig
from trainers import get_trainer


def train():
    cfg, nargs = BaseConfig.read_nargs()
    trainer = get_trainer(cfg, nargs)

    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.logger.info('Training interrupted')
    finally:
        trainer.close()


if __name__ == '__main__':
    train()
