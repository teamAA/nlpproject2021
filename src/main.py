import sys
import os
import dvc.api
import pickle
import train
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# def check_data_version():    
#     remote_ver = len(dvc.api.read('data-registry/train.csv', repo='../'))
#     with open("./version.txt") as file:
#         used_ver = file.readlines()[0].split(',')[0]
    
#     return int(used_ver) == int(remote_ver)

def retrain():
    return train.main()

def main():
    # if check_data_version() == True:
    #     print("There's no change in the database")
    #     sys.exit(0)

    print("New data is available")
    print("Doing retraining.......")
    #credentials = os.environ.get('GDRIVE_CREDENTIALS_DATA')
    model = retrain()
    #print(credentials)

    upload_model(model)

def upload_model(model):
    pickle.dump(model, open("../model/model.pkl", 'wb'))

    gauth = GoogleAuth()       
    credentials = os.environ['CREDENTIALS']
    gauth.LoadCredentialsFile(credentials)
    drive = GoogleDrive(gauth)  
    
    upload_file = '../model/model.pkl'
    gfile = drive.CreateFile({'parents': [{'id': '1BnBxVTuQ8Otab2h7aQFdXI14Ou3uIVVk'}]})
    
    gfile.SetContentFile(upload_file)
    gfile.Upload()

if __name__ == "__main__":
    main()
    
    
    
    
    