from unittest import TestCase
from learning_python import Dog

class TestDog(TestCase):

    def setUp(self):
        self.dog = Dog('buddy', 'golden')

class TestRun(TestDog):

    def test_run_once(self):
        self.assertEqual(self.dog.run(5), 25)

