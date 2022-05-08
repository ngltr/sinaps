# -*- coding: utf-8 -*-
from sinaps.data import io
import ssl



def test_read_swc():
    ssl._create_default_https_context = ssl._create_unverified_context #fix url open error 
    filename = "http://neuromorpho.org/dableFiles/chang/CNG%20version/V247fs-MT-Untreated-56.CNG.swc"
    nrn = io.read_swc(filename)
    assert nrn.nb_nodes == 128
