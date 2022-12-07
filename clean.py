import music21
import os

def main():
    #crawl through all files in the directory and ensure they are valid midi files using music21
    root = "/scratch/dchien/data/lmd_full"
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(".mid"):
                print("Trying now")
                try:
                    score = music21.converter.parse(os.path.join(root, file))
                    print(score)
                    notes = score.flat.getElementsByClass('Note')
                    print(notes)
                except:
                    print("Could not parse the notes!")
                    os.remove(os.path.join(root, file))
                    print("removed " + os.path.join(root, file))


if __name__ == "__main__":
    main()