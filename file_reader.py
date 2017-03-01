class FileReader:
    @staticmethod
    def run():
        label_data = FileReader.read_file('dataset/labels.txt', FileReader.label_conversion)
        review_data = FileReader.read_file('dataset/reviews.txt', FileReader.review_conversion)
        label_data = label_data[:1000]
        review_data = review_data[:1000]
        return (label_data, review_data)

    @staticmethod
    def read_file(filename, cb):
        data =[]

        with open(filename, 'r') as f:
            for line in f.readlines():
                data.append(cb(line.strip()))

        return data

    @staticmethod
    def label_conversion(label):
        return 1 if label == "positive" else 0

    @staticmethod
    def review_conversion(review):
        return review.upper().split(" ")
