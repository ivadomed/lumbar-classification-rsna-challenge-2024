import os
import argparse


def generate_directory_structure(root_dir, indent=''):
    items = os.listdir(root_dir)
    items.sort()

    for i, item in enumerate(items):
        path = os.path.join(root_dir, item)
        if i == len(items) - 1:
            connector = '└──'
            new_indent = indent + '    '
        else:
            connector = '├──'
            new_indent = indent + '│   '

        print(indent + connector + item)

        if os.path.isdir(path):
            generate_directory_structure(path, new_indent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate directory structure')
    parser.add_argument('-dir', type=str, help='The root directory to generate structure for')

    args = parser.parse_args()

    if os.path.isdir(args.dir):
        generate_directory_structure(args.dir)
    else:
        print(f"The provided path '{args.directory}' is not a valid directory.")
