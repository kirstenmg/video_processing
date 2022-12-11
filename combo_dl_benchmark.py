from combo_dataloader import ComboDataLoader
from dali_dataloader import DaliDataLoader
import time

if __name__ == "__main__":
	combo_dl = ComboDataLoader([DaliDataLoader(8, 1)])
	count = 0
	for batch in combo_dl:
		count += 1
		print(count)
	print("batch done")
	time.sleep(20)
