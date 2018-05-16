from get_twitter_data import read_file
from progress.bar import Bar

def check_missing(ids, file_in):
	missing = []
	bar = Bar('Processing: ' + file_in, max=len(ids))
	f = open(file_in + '_data.txt').read()
	for ID in ids:
		found_missing = False
		for tweet in ID.split(','):
			if found_missing:
				continue
			if ('"'+ tweet + '": null')  in f:
				found_missing = True
				missing.append(ID)

		bar.next()
	bar.finish()
	return missing 

if __name__ == "__main__":
	
	tuning = read_file("../data/twitter_ids.tuning.txt")
	missing_tuning = check_missing(tuning, '../data/tuning')
	val = read_file("../data/twitter_ids.validation.txt")
	missing_val= check_missing(val, "../data/validation")

	print("There are a total of {} triples".format(len(tuning) + len(val)))
	print("There are {} incomplete triples.".format(len(missing_tuning) + len(missing_val)))
	print("{} are from tuning and {} are from validation".format(len(missing_tuning), len(missing_val)))