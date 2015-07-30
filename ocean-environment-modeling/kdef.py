
# Define the kernel used for classification
def kerneldef(h, k):
    return h(1e-3, 1e5, 10) * k('gaussian', 
                                [h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1)])