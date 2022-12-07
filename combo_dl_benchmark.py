from combo_dataloader import ComboDataLoader
from dali_dataloader import DaliDataLoader

dali_dl = DaliDataLoader(8, 10)
print(dali_dl.next_batch())

combo_dl = ComboDataLoader([DaliDataLoader(8, 10)])
combo_dl.next_batch()
print("batch done")
combo_dl.shutdown()
