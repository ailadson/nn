import numpy as np

class FileReader:
    def __init__(self, filename):
        self.filename = filename
        self.read()
        self.build_char_to_int_maps()
        self.one_hot_encoding()
        print("One Hot Encoding Success!")

    def read(self):
        txt = ""
        with open(self.filename, 'r') as f:
            for line in f.readlines():
                txt += line
        self.txt = txt

    def build_char_to_int_maps(self):
        txt_set = set(self.txt)
        self.char_to_int = { c : i for i, c in enumerate(txt_set) }
        self.int_to_char = dict(enumerate(txt_set))
        self.num_chars = len(self.char_to_int)


    def one_hot_encoding(self):
        self.int_txt = [self.char_to_int[char] for char in self.txt]
        self.one_hot = np.zeros([len(self.txt), len(self.int_to_char)], dtype=np.float32)

        for i, int_char in enumerate(self.int_txt):
            self.one_hot[i, int_char] = 1.
