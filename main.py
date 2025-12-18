from src.train import Trainer
from src.render import GameUI

def main():
    trainer = Trainer()
    ui = GameUI(trainer)
    ui.run()

if __name__ == "__main__":
    main()
