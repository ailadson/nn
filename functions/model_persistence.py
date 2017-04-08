def save_model(name, nn):
    pickle.dump( nn, open( f"models/{name}.p", "wb" ) )

def prompt_save(nn):
    ans = input("Save model? d - discard | anything else - save\n")
    if ans == "d":
        print("Model Disarded\n\n")
    else:
        name = input("Enter filename. Omit extension\n")
        save_model(name, nn)
        print(f"Model Saved As: ./models/{name}.p \n\n")

def load_model(name):
    return pickle.load( open( f"./models/{name}.p", "rb" ) )
