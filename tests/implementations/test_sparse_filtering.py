import unittest
import os
import numpy as np

from ParamSklearn.implementations.SparseFiltering import SparseFiltering


class TestSparseFiltering(unittest.TestCase):
    def test_sparse_filtering(self):
        """Test sparse filtering on a simple dataset"""
        # load a few patches of image data from a file which is currently hard coded :)
	# JTS TODO: remove this hard coding
        dataset = "/home/springj/data/image_patches.npz"
        # try not to break testing if data is not available
        if (not os.path.isfile(dataset)):
            return
        patches = np.load(dataset)
        data = patches['data']
        preprocess = SparseFiltering(256, random_state = 123456)
        print("BEFORE")
        preprocess.fit(data)
	# JTS TODO: figure out a better test than this nonsense here ;)
        self.assertFalse((preprocess.W == 0).all())
        """
        # JTS: the following is only useful for visualization purposes
        # turn it on if you want to see sparse filtering in action on image data ;)
        import pylab 
        # method for eyeballing the features
        # assumes features in ROWS not columns!
        def displayData(X, example_width = False, display_cols = False):
            # compute rows, cols
            m,n = X.shape
            if not example_width:
                example_width = int(np.round(np.sqrt(n)))
            example_height = (n/example_width)
            # Compute number of items to display
            if not display_cols:
                display_cols = int(np.sqrt(m))
            display_rows = int(np.ceil(m/display_cols))
            pad = 1
            # Setup blank display
            display_array = -np.ones((pad+display_rows * (example_height+pad),
                                      pad+display_cols * (example_width+pad)))
            # Copy each example into a patch on the display array
            curr_ex = 0
            for j in range(display_rows):
                for i in range(display_cols):
                    if curr_ex>=m:
                        break
                    # Copy the patch
                    # Get the max value of the patch
                    max_val = abs(X[curr_ex,:]).max()
                    i_inds = example_width*[pad+j * (example_height+pad)+q for q in range(example_height)]
                    j_inds = [pad+i * (example_width+pad)+q
                              for q in range(example_width)
                              for nn in range(example_height)]
                    try:
                        newData = (X[curr_ex,:].reshape((example_height,example_width)))/max_val
                    except:
                        print X[curr_ex,:].shape
                        print (example_height,example_width)
                        raise
                    display_array[i_inds,j_inds] = newData.flatten()
                    curr_ex+=1
                if curr_ex>=m:
                    break
            # Display the image
            pylab.imshow(display_array,vmin=-1,vmax=1,interpolation='nearest',cmap=pylab.cm.gray)
            pylab.xticks([])
            pylab.yticks([])
        displayData(preprocess.W.T)
        pylab.show()
        #"""
        
