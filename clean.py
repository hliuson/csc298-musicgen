import music21
import os

def main():
    #crawl through all files in the directory and ensure they are valid midi files using music21
    root = "./data/lmd_full"
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(".mid"):
                try:
                    score = music21.converter.parse(os.path.join(root, file))
                    notes = score.flat.getElementsByClass('Note')
                except:
                    os.remove(os.path.join(root, file))
                    print("removed " + os.path.join(root, file))


if __name__ == "__main__":
    main()