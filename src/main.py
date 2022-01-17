import sys
import dvc.api
import train

def check_data_version():    
    remote_ver = len(dvc.api.read('data-registry/train.csv', repo='../'))
    with open("./version.txt") as file:
        used_ver = file.readlines()[0].split(',')[0]
    
    return int(used_ver) == int(remote_ver)

def retrain():
    train.main()

def main():
    if check_data_version() == True:
        print("There's no change in the database")
        sys.exit(0)

    else:
        print("New data is available")
        print("Doing retraining.......")
        retrain()
        print("Done!")

if __name__ == "__main__":
    main()
    
    