from classify import classify_csv
import os

if __name__ == "__main__":

    input_file = os.path.join(os.path.dirname(__file__), 'assets', 'test.csv')
    output_file = os.path.join(os.path.dirname(__file__), 'assets', 'output.csv')

    classify_csv(input_file, output_file)
    print(f"Classified logs saved to {output_file}")