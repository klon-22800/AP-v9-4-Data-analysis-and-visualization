import os
import csv

from typing import List

from progress.bar import IncrementalBar


def get_absolute_paths(num_mark: int, folder_name: str) -> List[str]:
    """
    The function gets absolute paths to files and returns a list with absolute paths

    Args:
        num_mark (int): class num
        folder_name (str): path of initial dataset

    Returns:
        List[str]: list of absolute path
    """
    absolute_path = os.path.abspath(f'{folder_name}')
    class_path = os.path.join(absolute_path, str(num_mark))
    names = os.listdir(class_path)
    absolute_paths = []
    for name in names:
        absolute_paths.append(os.path.join(class_path, name))
    return absolute_paths


def get_relative_paths(num_mark: int, folder_name: str) -> List[str]:
    """
    The function gets absolute paths to files and returns a list with relative paths
    Args:
        num_mark (int): class num
        folder_name (str): folder path with files

    Returns:
        List[str]: list of relative paths
    """
    relative_path = os.path.relpath(f'{folder_name}')
    class_path = os.path.join(relative_path, str(num_mark))
    names = os.listdir(class_path)
    relative_paths = []
    for name in names:
        relative_paths.append(os.path.join(class_path, name))
    return relative_paths


def make_csv(folder_name: str) -> None:
    """
    The function writes data to a csv file in the following format: absolute path, relative path, class label

    Args:
        folder_name (str): folder path with initial dataset
    """
    bar = IncrementalBar('Writting csv', max=5000)
    f = open("paths.csv", 'w')
    writer = csv.writer(f, delimiter=',', lineterminator='\n')
    for i in range(1, 6):
        absolute_paths = get_absolute_paths(i, f'{folder_name}')
        relative_paths = get_relative_paths(i, f'{folder_name}')
        for absolute_path, relative_path in zip(absolute_paths, relative_paths):
            bar.next()
            writer.writerow([absolute_path, relative_path, str(i)])


def main() -> None:
    make_csv('dataset')


if __name__ == '__main__':
    main()
