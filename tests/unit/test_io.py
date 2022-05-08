# -*- coding: utf-8 -*-
from sinaps.data import io
import pytest


@pytest.mark.skip #skipping this test because ssl issue with neuromorpho website
def test_read_swc():
    filename = "http://neuromorpho.org/dableFiles/chang/CNG%20version/V247fs-MT-Untreated-56.CNG.swc"
    nrn = io.read_swc(filename)
    assert nrn.nb_nodes == 128
