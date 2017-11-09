from interface import AEInterface

mode = "q"

model = AEInterface(number_of_trials=1, number_of_episodes=1)

if mode=="s":
	model.execute(True, reload=False, persist=True, mode=0, filepath="logs")
elif mode=="q":
	model.execute(True, reload=False, persist=True, mode=1, filepath="logsq")