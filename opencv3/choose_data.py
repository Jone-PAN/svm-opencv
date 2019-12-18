import os
import shutil
import random


def main():
    random.seed(123456)
    wd = os.getcwd()
    src_path = os.path.join(wd, 'src_data')
    dst_path = os.path.join(wd, 'data')

    for _, dirs, _ in os.walk(src_path):
        for d in dirs:
            if d == []:
                continue
            files = os.listdir(os.path.join(src_path, d))
            files = [f for f in files if f.endswith('.jpg')]
            files = random.sample(files, min(len(files), 1000))
            for f in files:
                file_path = os.path.join(dst_path, d)
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                src_file = os.path.join(src_path, d, f)
                dst_file = os.path.join(dst_path, d, f)
                shutil.copy2(src_file, dst_file)


if __name__ == '__main__':
    main()
