from game import Game
from player import Player


def main(num_players,num_games=1):
    for _ in range(num_games):
        if num_players == 1:
            ai = Player()
        if num_players == 2:
            ai1 = Player()
            ai2 = Player()
            ai2.player = "O"
        game = Game()
        game.print_board()
        while True:
            if num_players == 1:
                if ai.player == game.current_player:
                    move = ai.analyse(game)
                    print(move)
                else:
                    move = input(f"player {game.current_player}:")
            elif num_players == 0:
                move = input(f"player {game.current_player}:")
            elif num_players == 2:
                if ai1.player == game.current_player:
                    move = ai1.analyse(game)
                elif ai2.player == game.current_player:
                    move = ai2.analyse(game)
            if move == "exit":
                break
            while not game.player_move(move):
                pass
            if game.win is not None:
                break
            if game.draw:
                break
        if num_players == 1:
            ai.add_game_to_history(game)
        elif num_players == 2:
            ai1.add_game_to_history(game)
            # if ai1.player == "X":
            #     ai1.player = "O"
            #     ai2.player = "X"
            # else:
            #     ai1.player = "X"
            #     ai2.player = "O"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(2,100)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
