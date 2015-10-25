import parmap

def function(a, b):

	print('a = {0} | b = {1}'.format(a, b))
	return a + b

def main():

	a = [i for i in range(10)]
	b = [i*i for i in range(10)]
	print('Initial: a = {0} | b = {1}'.format(a, b))

	arg = [(a[i], b[i]) for i in range(len(a))]
	print(arg)
	results = parmap.starmap(function, arg)
	print(results)

if __name__ == "__main__":
    main()
