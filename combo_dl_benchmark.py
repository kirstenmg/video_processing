from combo_dataloader import ComboDataLoader, DataLoaderType
from dali_dataloader import DaliDataLoader
import time

# Get video paths from annotation CSV
ANNOTATION_FILE_PATH = "/home/maureen/kinetics/kinetics400/annotations/val.csv"
VIDEO_BASE_PATH = "/home/maureen/kinetics/kinetics400"
video_paths = []
with open(ANNOTATION_FILE_PATH, 'r') as annotation_file:
    for i, line in enumerate(annotation_file):
        if i != 0: # skip column headers
            line = annotation_file.readline()
            label, youtube_id, time_start, time_end, split, is_cc = line.strip().split(',')
            vpath = f'{VIDEO_BASE_PATH}/{split}/{youtube_id}_{int(time_start):06d}_{int(time_end):06d}.mp4'
            video_paths.append(vpath)

def main():
	combo_dl = ComboDataLoader([DataLoaderType.PYTORCH, DataLoaderType.PYTORCH], video_paths=video_paths)
	count = 0
	for batch in combo_dl:
		count += 1

	print("batch done")



if __name__ == "__main__":
	
	main()
