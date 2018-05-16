import requests
import json
from time import sleep
from progress.bar import Bar

def store_data(twitter_ids, file_out):
	"""
	Gets the data and writes it to a file
	"""
	bar = Bar('Processing: ' + file_out, max=len(twitter_ids))
	with open(file_out, 'w') as f:
		for twitter_id in twitter_ids:
			data = getData(twitter_id)
			
			f.write(json.dumps(data, indent=4, separators=(',', ': '))+'\n')
			bar.next()
	bar.finish()


def getData(data):
	"""
	Performs the necessary post request to get the data for a triple
	"""
	headers={
	"Authorization" : "Bearer AAAAAAAAAAAAAAAAAAAAAEJXhgAAAAAAwe%2Bcbsk5UU7yHwoV557ISKHHwIU%3DOUTtv84OxsgS8GgK7WOq1PH948gzCPECyYr5Mq31xZIhPKT5Ii", 
	"user-agent": "AttentionMechProj", 
	"Content-Type" : "application/x-www-form-urlencoded;charset=utf-8"
	}

	form = {'id' : data, 'map':'true', 'trim_user':'true', 'include_entities' :'true'}
	r = requests.post('https://api.twitter.com/1.1/statuses/lookup.json', 
		data=form,
		headers=headers)
	
	return r.json()

def read_file(file_in):
	"""
	Parses a file
	"""
	with open(file_in) as f:
		return [line for line in f.read().split('\r\n')]


if __name__ == "__main__":
	tuning = read_file("../data/twitter_ids.tuning.txt")
	store_data(tuning, '../data/tuning_data.txt')
	val = read_file("../data/twitter_ids.validation.txt")
	store_data(val, '../data/validation_data.txt')