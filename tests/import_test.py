"""Tests for import."""

import unittest
import pytinyrenderer


class ImportTest(unittest.TestCase):

  def test_import(self):
    scene = pytinyrenderer.TinySceneRenderer()
    self.assertTrue(scene)

if __name__ == '__main__':
  unittest.main()
