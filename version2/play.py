from nnet import Network

def play():
    ans = input("Do you want to be player X? (yes or no)")
    if ans == "yes":
        player = "X"
    elif ans == "no":
        player == "O"
    else:
        player == "X"
    print(f"You are player {player}")

    
