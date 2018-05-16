import random
import sys, getopt

"""

data_creation.py -r <randomlength(True,False)> -n <numberOfPoints> -o <outputfile>

This script creates a number of data points. The data points are binairy numbers of
length 20 to 30. These can be used for the copy task. 

arguments:
randomlength   = A boolean to check if the data points have a variable length
                 If True, length between 20 and 30. If False length is 20
                 Default is False
numberOfPoints = The number of data points to be made. Default is 500
outputfile     = The output file where the data is stored. Default is out.txt

"""

def main(argv):
    randomlength = False
    dataPoints = 500
    outputfile = 'out.txt'
    try:
        opts, args = getopt.getopt(argv,"hr:n:o:",["ranLength=","numOfPoints=","ofile="])
    except getopt.GetoptError:
        print 'data_creation.py -r <randomlength(True,False)> -n <numberOfPoints> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'data_creation.py -r <randomlength(True,False)> -n <numberOfPoints> -o <outputfile>'
            sys.exit()
        elif opt in ("-r", "--ranLength"):
            if arg:
                randomlength = arg
        elif opt in ("-n", "--numOfPoints"):
            dataPoints = int(arg)
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print 'Use random lenght is: ', randomlength
    print 'Number of data point is: ', dataPoints
    print 'Output file is: ', outputfile

    f = open(outputfile, 'w')

    for point in xrange(0,dataPoints):
        num = ""
        length = 20
        if randomlength:
            length = length + random.randint(0,10)
        for x in xrange(0,length):
            temp = str(random.randint(0,1))
            num = num + temp
        f.write(num+'\n')
    f.close

if __name__ == "__main__":
    main(sys.argv[1:])