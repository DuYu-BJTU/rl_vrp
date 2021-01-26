import os

model_list = os.listdir("saved/")
epi_turns = list()
for model_name in model_list:
    epi_turn = int(model_name.split("_")[-1].split(".")[0])
    if epi_turn > 10000:
        os.remove("saved/{}".format(model_name))
