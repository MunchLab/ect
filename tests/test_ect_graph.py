import unittest
from ect import embed_graph, ect_graph



class TestECT(unittest.TestCase):
    def test_example_graph_ect(self):
        G = embed_graph.create_example_graph()
        num_dirs = 8
        num_thresh = 10
        myect = ect_graph.ECT(num_dirs, num_thresh)
        self.assertEqual( myect.ECT_matrix.shape, (8,10))

        r = G.get_bounding_radius()
        myect.set_bounding_radius(1.2*r)
        ecc = myect.calculateECC(G, 0)
        self.assertEqual( len(ecc), num_thresh)

    def test_check_bounding_radius(self):
        
        # make an example graph 
        G = embed_graph.create_example_graph()
        num_dirs = 8
        num_thresh = 10

        myect = ect_graph.ECT(num_dirs, num_thresh)

        # At this point, there shouldn't be a radius set 
        self.assertIs( myect.bound_radius, None)

        # Try to calculate the ECC without a radius set
        myect.calculateECC(G, 0, bound_radius=None)

        # Try to calculate the ECC with a negative radius.
        # It should throw an error.
        with self.assertRaises(ValueError):  
            myect.calculateECC(G, 0, bound_radius= -1)

        # Try to calculate the ECC with tightbbox set to True 
        # This should  work fine 
        ecc = myect.calculateECC(G, 0, bound_radius=None)
        self.assertEqual( len(ecc), num_thresh)

        # Now set the bounding radius 
        r = G.get_bounding_radius()
        myect.set_bounding_radius(1.2*r)
        ecc = myect.calculateECC(G, 0, bound_radius=None)
        self.assertEqual( len(ecc), num_thresh)

        # TODO: write a test where we check that if None is passed and the radius is set internally, it will use that one.


    


if __name__ == '__main__':
    unittest.main()