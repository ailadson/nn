import config
import numpy as np
import re

class TextDataSet:
    def __init__(self, filename):
        self.filename = filename
        self.num_chars = None

        self.char_to_int_map = {}
        self.int_to_char_map = {}

        self.text = ""
        self.text_as_ints = None
        self.text_as_one_hot = None
        self.num_chars = None

        self.load()

    def load(self):
        self.read()
        self.build_char_int_maps()
        self.create_one_hot_encoding()
        print("One Hot Encoding Success!")

    def read(self):
        rex = re.compile(r'\W+')
        with open(self.filename, 'r') as f:
            for line in f.readlines():
                if config.CASE_INSENSITIVE:
                    line = line.lower()
                if not config.READ_NEW_LINES:
                    line = re.sub('\n+',' ',line)
                    line = re.sub('\x0c',' ',line)
                if config.COLLAPSE_WHITESPACE:
                    line = re.sub(' +',' ',line)
                self.text += line
        self.num_chars = len(set(self.text))

    def build_char_int_maps(self):
        for char in self.text:
            if char in self.char_to_int_map: continue
            char_int = len(self.int_to_char_map)
            self.char_to_int_map[char] = char_int
            self.int_to_char_map[char_int] = char

    def create_one_hot_encoding(self):
        self.text_as_ints = [self.char_to_int_map[char] for char in self.text]
        self.text_as_one_hot = np.zeros([len(self.text), self.num_chars], dtype=np.float32)

        for i, char_as_int in enumerate(self.text_as_ints):
            self.text_as_one_hot[i, char_as_int] = 1.

    def char_to_one_hot(self, char):
        char_int = self.char_to_int_map[char]
        one_hot = np.zeros([self.num_chars], dtype=np.float32)
        one_hot[char_int] = 1.
        return one_hot

    def one_hot_to_char(self, one_hot):
        int_char = one_hot.argmax()
        return self.int_to_char_map[int_char]
