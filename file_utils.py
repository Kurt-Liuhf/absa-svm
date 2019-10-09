import os
import shutil


def read_as_list(file_path):
    return [l.strip() for l in open(file_path, 'r').readlines()]


def write_list(file_path, list2write):
    with open(file_path, 'w') as f:
        f.write("\n".join(list2write))


def remove_dirs(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)

    print('%s removed...' % dir)


def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    print('%s created...' % dir)


def clear_folder(folder_path):
    os.removedirs(folder_path)
    os.makedirs(folder_path)



if __name__ == '__main__':
    l = ['this', 'is', 'a', 'test', '']
    write_list('test.txt', list2write=l)
    print(read_as_list('test.txt'))

    make_dirs('a/b/c')
    make_dirs('a/b/d')
    remove_dirs('a/b/c')

