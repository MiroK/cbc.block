#!/usr/bin/env python

from __future__ import absolute_import
from distutils.core import setup
setup(name = "cbc.block",
      version = "0.0.1",
      description = "Block utilities",
      author = "Joachim Berdal Haga",
      author_email = "jobh@simula.no",
      url = "http://www.launchpad.net/cbc.block/",
      packages = ["block", "block.algebraic", "block.iterative", "block.algebraic.petsc"]
)

